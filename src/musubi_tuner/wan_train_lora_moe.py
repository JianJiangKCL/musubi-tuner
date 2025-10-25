
import argparse
import os
import json
from pathlib import Path
from typing import Optional, Dict, List
import torch
import torch.nn as nn
from tqdm import tqdm
from accelerate import Accelerator

from musubi_tuner.wan_train_network import WanNetworkTrainer, logger, setup_parser_common, wan_setup_parser, read_config_from_file
from musubi_tuner.networks.lora_wan_moe import create_wan_lora_moe, WanLoRAMoENetwork
from musubi_tuner.losses.lora_moe_losses import LoRAMoECombinedLoss
from musubi_tuner.dataset.image_video_dataset import ItemInfo


class WanLoRAMoETrainer(WanNetworkTrainer):
    """
    Extended WAN trainer with LoRA-MoE support.
    """

    def __init__(self):
        super().__init__()
        self.lora_moe_network: Optional[WanLoRAMoENetwork] = None
        self.lora_moe_loss: Optional[LoRAMoECombinedLoss] = None
        self.training_stage = "stage_a"

        # Instrument detector/classifier (optional, set externally)
        self.instrument_classifier = None

        # ROI detector (optional, set externally)
        self.roi_detector = None

    def handle_model_specific_args(self, args):
        """Handle LoRA-MoE specific arguments."""
        super().handle_model_specific_args(args)

        # Training stage
        self.training_stage = getattr(args, "training_stage", "stage_a")
        logger.info(f"LoRA-MoE Training Stage: {self.training_stage}")

        # LoRA-MoE config
        self.lora_moe_config = {
            "lora_dim": getattr(args, "lora_dim", 4),
            "alpha": getattr(args, "lora_alpha", 1.0),
            "num_experts": getattr(args, "num_experts", 4),
            "expert_names": getattr(args, "expert_names", None),
            "use_base_lora": getattr(args, "use_base_lora", True),
            "dropout": getattr(args, "lora_dropout", 0.0),
            "rank_dropout": getattr(args, "lora_rank_dropout", 0.0),
            "module_dropout": getattr(args, "lora_module_dropout", 0.0),
        }

        # Router config
        self.router_config = {
            "routing_mode": getattr(args, "routing_mode", "learned"),  # Default to learned for stage_b
            "top_k": getattr(args, "router_top_k", 2),
            "temperature": getattr(args, "router_temperature", 0.7),
            "ema_beta": getattr(args, "router_ema_beta", 0.9),
            "learnable_hidden_dim": getattr(args, "router_hidden_dim", 64),
            "input_dim": getattr(args, "router_input_dim", 512),
            "use_text_conditioning": getattr(args, "use_text_conditioning", False),
        }

        # If we only use instrument logits (num_experts) as input, align router input dim to num_experts to avoid matmul shape mismatch
        if not self.router_config["use_text_conditioning"]:
            self.router_config["input_dim"] = self.lora_moe_config["num_experts"]

        # Router training flag (for stage_b)
        self.train_router_in_stage_b = getattr(args, "train_router", True)  # Default: train router in stage_b
        self.use_teacher_guidance = getattr(args, "use_teacher_guidance", True)  # Use rule-based as teacher
        self.teacher_kl_weight = getattr(args, "teacher_kl_weight", 1.0)

        # Target blocks
        target_blocks_str = getattr(args, "target_blocks", "last_8")
        if target_blocks_str.startswith("["):
            # Parse as list
            import ast
            self.target_blocks = ast.literal_eval(target_blocks_str)
        else:
            self.target_blocks = target_blocks_str

        # Projection ranks
        self.projection_ranks = {
            "q": getattr(args, "rank_q", 8),
            "k": getattr(args, "rank_k", 4),
            "v": getattr(args, "rank_v", 4),
            "o": getattr(args, "rank_o", 4),
            "ffn": getattr(args, "rank_ffn", 4),
        }

        # Loss weights
        self.loss_weights = {
            "base_diffusion": getattr(args, "weight_base_diffusion", 1.0),
            "roi_recon": getattr(args, "weight_roi_recon", 3.0),
            "identity": getattr(args, "weight_identity", 0.5),
            "temporal": getattr(args, "weight_temporal", 0.5),
            "routing_entropy": getattr(args, "weight_routing_entropy", 0.01),
            "routing_load_balance": getattr(args, "weight_routing_load_balance", 0.05),
        }

        # Instrument data config
        self.instrument_data_path = getattr(args, "instrument_data_path", None)
        self.use_roi_loss = getattr(args, "use_roi_loss", True)
        self.use_identity_loss = getattr(args, "use_identity_loss", False)  # Requires classifier
        self.use_temporal_loss = getattr(args, "use_temporal_loss", False)  # Requires flow model

        # Pretrained weights
        self.base_lora_weights = getattr(args, "base_lora_weights", None)
        self.lora_moe_weights = getattr(args, "lora_moe_weights", None)

    def setup_lora_moe(self, model: nn.Module):
        """
        Setup LoRA-MoE network on the model.
        """
        logger.info("Setting up LoRA-MoE network...")

        # Create LoRA-MoE network
        self.lora_moe_network = create_wan_lora_moe(
            model=model,
            lora_config=self.lora_moe_config,
            router_config=self.router_config,
            target_blocks=self.target_blocks,
        )

        # Load pretrained weights if provided
        if self.lora_moe_weights is not None:
            logger.info(f"Loading LoRA-MoE weights from {self.lora_moe_weights}")
            self.lora_moe_network.load_lora_moe_weights(self.lora_moe_weights)
        elif self.base_lora_weights is not None and self.training_stage == "stage_b":
            # Load base LoRA from stage A
            logger.info(f"Loading base LoRA weights from {self.base_lora_weights}")
            # Note: This assumes stage A saved compatible weights
            # You may need to implement a separate loading function
            self.lora_moe_network.load_lora_moe_weights(self.base_lora_weights)

        # Prepare for training stage
        train_router = self.train_router_in_stage_b if self.training_stage == "stage_b" else False
        self.lora_moe_network.prepare_for_training_stage(self.training_stage, train_router=train_router)

        logger.info(f"LoRA-MoE network ready for {self.training_stage}")
        if self.training_stage == "stage_b" and train_router:
            logger.info(f"  Router training: ENABLED (mode: {self.router_config['routing_mode']})")
            if self.use_teacher_guidance:
                logger.info(f"  Teacher guidance: ENABLED (KL weight: {self.teacher_kl_weight})")
        else:
            logger.info(f"  Router training: DISABLED (using rule-based routing)")

    def setup_loss_function(self):
        """
        Setup LoRA-MoE combined loss function.
        """
        logger.info("Setting up LoRA-MoE loss function...")

        # Create loss modules
        from musubi_tuner.losses.lora_moe_losses import (
            ROIWeightedLoss,
            InstrumentIdentityLoss,
            TemporalConsistencyLoss,
            RoutingRegularizationLoss,
        )

        roi_loss = ROIWeightedLoss(
            roi_weight=self.loss_weights["roi_recon"]
        ) if self.use_roi_loss else None

        identity_loss = InstrumentIdentityLoss(
            classifier=self.instrument_classifier,
            roi_weight=1.0,
            background_weight=0.25,
        ) if self.use_identity_loss else None

        temporal_loss = TemporalConsistencyLoss(
            flow_estimator=None,  # Optional: add flow estimator
            consistency_metric="l1",
            roi_only=True,
        ) if self.use_temporal_loss else None

        routing_loss = RoutingRegularizationLoss(
            entropy_weight=self.loss_weights["routing_entropy"],
            load_balance_weight=self.loss_weights["routing_load_balance"],
            teacher_kl_weight=self.teacher_kl_weight,
            num_experts=self.lora_moe_config["num_experts"],
        )

        # Combined loss
        self.lora_moe_loss = LoRAMoECombinedLoss(
            base_diffusion_weight=self.loss_weights["base_diffusion"],
            roi_recon_weight=self.loss_weights["roi_recon"],
            identity_weight=self.loss_weights["identity"],
            temporal_weight=self.loss_weights["temporal"],
            roi_loss=roi_loss,
            identity_loss=identity_loss,
            temporal_loss=temporal_loss,
            routing_loss=routing_loss,
        )

        logger.info("LoRA-MoE loss function ready")

    def get_instrument_logits_from_batch(self, batch: Dict) -> Optional[torch.Tensor]:
        """
        Extract or predict instrument logits for routing.

        Args:
            batch: Training batch dict

        Returns:
            instrument_logits: [B, num_instruments] or None
        """
        # Option 1: Pre-computed logits from dataset
        if "instrument_logits" in batch:
            return batch["instrument_logits"]

        # Option 2: Pre-computed labels (convert to one-hot)
        if "instrument_labels" in batch:
            labels = batch["instrument_labels"]  # [B]
            B = labels.shape[0]
            logits = torch.zeros(B, self.lora_moe_config["num_experts"], device=labels.device)
            logits.scatter_(1, labels.unsqueeze(1), 1.0)
            return logits

        # Option 3: Run classifier on latents/frames (requires instrument_classifier)
        if self.instrument_classifier is not None:
            # Extract middle frame or random frame
            latents = batch.get("latents", None)
            if latents is not None:
                # For simplicity, use middle frame
                # In practice, you'd decode latents to pixels and run classifier
                # This is a placeholder
                logger.warning("Instrument classification from latents not implemented, using uniform distribution")

        # Fallback: uniform distribution
        B = batch["latents"].shape[0] if "latents" in batch else 1
        num_experts = self.lora_moe_config["num_experts"]
        return torch.ones(B, num_experts, device=batch["latents"].device) / num_experts

    def get_roi_mask_from_batch(self, batch: Dict) -> Optional[torch.Tensor]:
        """
        Extract or predict ROI mask for loss weighting.

        Args:
            batch: Training batch dict

        Returns:
            roi_mask: [B, 1, F, H, W] or None
        """
        # Option 1: Pre-computed ROI masks from dataset
        if "roi_mask" in batch:
            return batch["roi_mask"]

        # Option 2: Run ROI detector (requires roi_detector)
        if self.roi_detector is not None:
            # Placeholder
            logger.warning("ROI detection not implemented, using None")

        # Fallback: no ROI mask
        return None

    def train_batch(
        self,
        batch: Dict,
        accelerator: Accelerator,
        global_step: int,
    ) -> Dict[str, float]:
        """
        Train on a single batch with LoRA-MoE.

        Args:
            batch: Training batch
            accelerator: Accelerator instance
            global_step: Global training step

        Returns:
            loss_dict: Dictionary of loss components
        """
        # Get instrument logits for routing
        instrument_logits = self.get_instrument_logits_from_batch(batch)

        # Compute teacher gates (rule-based) if using teacher guidance
        teacher_gates = None
        if self.training_stage == "stage_b" and self.use_teacher_guidance and self.train_router_in_stage_b:
            # Create a temporary rule-based router for teacher guidance
            from musubi_tuner.networks.lora_moe import InstrumentRouter
            teacher_router = InstrumentRouter(
                num_experts=self.lora_moe_config["num_experts"],
                routing_mode="rule_based",
                top_k=self.router_config["top_k"],
                temperature=self.router_config["temperature"],
            ).to(instrument_logits.device)

            with torch.no_grad():
                teacher_gates = teacher_router(
                    instrument_logits=instrument_logits,
                    apply_topk=True,
                    apply_ema=False,
                )

        # Set routing gates (learned router if training, else rule-based)
        self.lora_moe_network.set_routing_gates(
            instrument_logits=instrument_logits,
            text_embeddings=None,  # Optional: add text conditioning
        )

        # Get ROI mask
        roi_mask = self.get_roi_mask_from_batch(batch)

        # Get timestep from batch
        timestep = batch.get("timestep", 0.5)  # Normalized timestep
        self.lora_moe_network.set_timestep(timestep)

        # Forward pass through model
        # This depends on your existing training loop structure
        # Placeholder for model prediction
        model_pred = batch.get("model_pred")  # You'll get this from your model forward
        target = batch.get("target")          # Target noise

        # Get routing gates for loss (current learned gates)
        routing_gates = self.lora_moe_network.router.ema_gates.unsqueeze(0)

        # Compute loss
        total_loss, loss_dict = self.lora_moe_loss(
            model_pred=model_pred,
            target=target,
            generated_frames=None,  # Optional: decode latents for identity/temporal loss
            roi_mask=roi_mask,
            instrument_labels=batch.get("instrument_labels"),
            routing_gates=routing_gates,
            teacher_gates=teacher_gates,  # Rule-based teacher for KL guidance
            stage=self.training_stage,
        )

        return total_loss, loss_dict

    # Override: load transformer then setup LoRA-MoE and loss
    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        model = super().load_transformer(
            accelerator,
            args,
            dit_path,
            attn_mode,
            split_attn,
            loading_device,
            dit_weight_dtype,
        )

        # Ensure LoRA-MoE is applied and loss is ready
        if self.lora_moe_network is None:
            self.setup_lora_moe(model)
        if self.lora_moe_loss is None:
            self.setup_loss_function()

        # Avoid double-counting base diffusion: parent loop already computes weighted MSE
        try:
            self.lora_moe_loss.base_diffusion_weight = 0.0
        except Exception:
            pass

        # Ensure router is on the same device as accelerator
        try:
            if self.lora_moe_network is not None and hasattr(self.lora_moe_network, "router"):
                self.lora_moe_network.router.to(accelerator.device)
        except Exception:
            pass

        # If WAN high/low training is enabled, convert the inactive (high-noise) state dict
        # into the MoE-wrapped keyspace so that weight swapping works without key mismatches.
        try:
            if getattr(self, "high_low_training", False) and getattr(self, "dit_inactive_state_dict", None) is not None:
                # Build a MoE-compatible state dict: start from the current MoE model state,
                # then overwrite base DiT weights using the inactive (plain) high-noise state dict,
                # mapping projection keys into org_module.* for wrapped modules.
                moe_sd = model.state_dict()
                plain_sd = self.dit_inactive_state_dict

                def map_plain_key_to_moe_key(k: str) -> str:
                    # Map projection weights/biases for self_attn/cross_attn and ffn.0 to org_module.*
                    # Example: blocks.32.self_attn.q.weight -> blocks.32.self_attn.q.org_module.weight
                    #          blocks.37.ffn.0.weight      -> blocks.37.ffn.0.org_module.weight
                    parts = k.split(".")
                    # attn projections
                    if len(parts) >= 5 and parts[0] == "blocks" and parts[2] in ("self_attn", "cross_attn") and parts[3] in ("q", "k", "v", "o") and parts[4] in ("weight", "bias"):
                        return ".".join(parts[:4] + ["org_module", parts[4]])
                    # ffn first linear layer is indexed as ffn.0
                    if len(parts) >= 5 and parts[0] == "blocks" and parts[2] == "ffn" and parts[3] == "0" and parts[4] in ("weight", "bias"):
                        return ".".join(parts[:4] + ["org_module", parts[4]])
                    return k

                # Start from current MoE state (keeps LoRA-MoE params). Overwrite base weights from plain_sd.
                moe_compatible_sd = {}
                for mk, mv in moe_sd.items():
                    moe_compatible_sd[mk] = mv

                for pk, pv in plain_sd.items():
                    mk = map_plain_key_to_moe_key(pk)
                    if mk in moe_compatible_sd:
                        try:
                            moe_compatible_sd[mk] = pv.to(moe_compatible_sd[mk].dtype)
                        except Exception:
                            # Fallback without dtype cast
                            moe_compatible_sd[mk] = pv

                # Replace inactive state dict with the MoE-compatible version
                self.dit_inactive_state_dict = moe_compatible_sd
        except Exception as e:
            logger.warning(f"Failed to convert inactive state dict for high/low swap to MoE format: {e}")

        return model

    # Hook: compute extra MoE losses (routing entropy, load balance, teacher KL, etc.)
    def compute_extra_loss(
        self,
        batch: Dict,
        timesteps: torch.Tensor,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        transformer: nn.Module,
        accelerator: Accelerator,
        args: argparse.Namespace,
    ) -> torch.Tensor:
        if self.lora_moe_network is None or self.lora_moe_loss is None:
            return torch.tensor(0.0, device=accelerator.device, dtype=model_pred.dtype)

        # Prepare routing inputs
        instrument_logits = self.get_instrument_logits_from_batch(batch)
        if isinstance(instrument_logits, torch.Tensor):
            instrument_logits = instrument_logits.to(device=accelerator.device)

        # Teacher gates (rule-based) for learned router guidance
        teacher_gates = None
        if self.training_stage in ["stage_b", "stage_c"] and self.use_teacher_guidance and self.train_router_in_stage_b:
            from musubi_tuner.networks.lora_moe import InstrumentRouter
            teacher_router = InstrumentRouter(
                num_experts=self.lora_moe_config["num_experts"],
                routing_mode="rule_based",
                top_k=self.router_config["top_k"],
                temperature=self.router_config["temperature"],
            ).to(accelerator.device)
            with torch.no_grad():
                teacher_gates = teacher_router(
                    instrument_logits=instrument_logits,
                    apply_topk=True,
                    apply_ema=False,
                )

        # Set current routing for modules
        self.lora_moe_network.set_routing_gates(
            instrument_logits=instrument_logits,
            text_embeddings=None,
        )

        # Set normalized timestep (use batch mean)
        if isinstance(timesteps, torch.Tensor):
            t_norm = (timesteps.float().mean() / 1000.0).clamp(0.0, 1.0).item()
        else:
            t_norm = 0.5
        self.lora_moe_network.set_timestep(t_norm)

        # ROI mask if available
        roi_mask = self.get_roi_mask_from_batch(batch)
        if isinstance(roi_mask, torch.Tensor):
            roi_mask = roi_mask.to(device=accelerator.device)

        # Gates for loss (use current router outputs)
        # Ensure router is on device
        try:
            self.lora_moe_network.router.to(accelerator.device)
        except Exception:
            pass

        # Use differentiable gates (no top-k) for loss to allow gradients to flow into router MLP
        gates_for_loss = self.lora_moe_network.router(
            instrument_logits=instrument_logits,
            text_embeddings=None,
            apply_topk=False,
            apply_ema=False,
        )

        # Compute extra loss components (routing, roi/identity/temporal if enabled)
        extra_total, loss_dict = self.lora_moe_loss(
            model_pred=model_pred,
            target=target,
            generated_frames=None,
            roi_mask=roi_mask,
            instrument_labels=batch.get("instrument_labels"),
            routing_gates=gates_for_loss,
            teacher_gates=teacher_gates,
            stage=self.training_stage,
        )
        # Cache breakdown logs for external logging
        try:
            logs = {"loss/moe_total": float(extra_total.detach().item())}
            if isinstance(loss_dict, dict):
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        logs[f"loss/moe/{k}"] = float(v.detach().item())
                    else:
                        # assume numeric
                        logs[f"loss/moe/{k}"] = float(v)
            self._last_moe_loss_logs = logs
        except Exception:
            self._last_moe_loss_logs = {"loss/moe_total": float(extra_total.detach().item())}

        return extra_total

    def get_extra_loss_logs(self) -> Dict[str, float]:
        try:
            return getattr(self, "_last_moe_loss_logs", {})
        except Exception:
            return {}

    def save_checkpoint(self, save_dir: str, global_step: int):
        """Save LoRA-MoE checkpoint."""
        save_path = os.path.join(save_dir, f"lora_moe_step_{global_step}.safetensors")
        self.lora_moe_network.save_lora_moe_weights(save_path)

        # Save training state
        state_path = os.path.join(save_dir, f"training_state_step_{global_step}.json")
        state = {
            "global_step": global_step,
            "training_stage": self.training_stage,
            "lora_moe_config": self.lora_moe_config,
            "router_config": self.router_config,
            "loss_weights": self.loss_weights,
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved checkpoint to {save_path}")


def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser for LoRA-MoE training."""
    # Base/common arguments
    parser = setup_parser_common()
    parser = wan_setup_parser(parser)
    parser.description = "Train WAN with LoRA-MoE"

    # LoRA-MoE specific arguments
    parser.add_argument("--training_stage", type=str, default="stage_a",
                      choices=["stage_a", "stage_b", "stage_c"],
                      help="Training stage: stage_a (base LoRA), stage_b (experts), stage_c (router)")

    # LoRA config
    parser.add_argument("--lora_dim", type=int, default=4, help="Base LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha scaling")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of expert LoRAs")
    parser.add_argument("--expert_names", type=str, nargs="+", default=None,
                      help="Expert names (default: Scissors, Hook/Electrocautery, Suction, Other)")
    parser.add_argument("--use_base_lora", action="store_true", default=True,
                      help="Use base LoRA (shared adaptation)")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--lora_rank_dropout", type=float, default=0.0, help="LoRA rank dropout")
    parser.add_argument("--lora_module_dropout", type=float, default=0.0, help="LoRA module dropout")

    # Projection-specific ranks
    parser.add_argument("--rank_q", type=int, default=8, help="Rank for Q projection")
    parser.add_argument("--rank_k", type=int, default=4, help="Rank for K projection")
    parser.add_argument("--rank_v", type=int, default=4, help="Rank for V projection")
    parser.add_argument("--rank_o", type=int, default=4, help="Rank for O projection")
    parser.add_argument("--rank_ffn", type=int, default=4, help="Rank for FFN")

    # Target blocks
    parser.add_argument("--target_blocks", type=str, default="last_8",
                      help="Which blocks to adapt: 'last_8', 'last_6', or list like '[32,33,34,35,36,37,38,39]'")

    # Router config
    parser.add_argument("--routing_mode", type=str, default="learned",
                      choices=["rule_based", "learned"],
                      help="Routing mode (default: learned for stage_b)")
    parser.add_argument("--router_top_k", type=int, default=2, help="Top-k routing")
    parser.add_argument("--router_temperature", type=float, default=0.7, help="Router temperature")
    parser.add_argument("--router_ema_beta", type=float, default=0.9, help="Router EMA beta")
    parser.add_argument("--router_hidden_dim", type=int, default=64, help="Router MLP hidden dimension")
    parser.add_argument("--router_input_dim", type=int, default=512, help="Router input dimension")
    parser.add_argument("--use_text_conditioning", action="store_true", default=False,
                      help="Use text embeddings for routing")

    # Router training (for stage_b)
    parser.add_argument("--train_router", action="store_true", default=True,
                      help="Train router in stage_b (default: True for simultaneous training)")
    parser.add_argument("--use_teacher_guidance", action="store_true", default=True,
                      help="Use rule-based router as teacher for learned router")
    parser.add_argument("--teacher_kl_weight", type=float, default=1.0,
                      help="KL divergence weight for teacher guidance")

    # Loss weights
    parser.add_argument("--weight_base_diffusion", type=float, default=1.0, help="Base diffusion loss weight")
    parser.add_argument("--weight_roi_recon", type=float, default=3.0, help="ROI reconstruction weight")
    parser.add_argument("--weight_identity", type=float, default=0.5, help="Identity preservation weight")
    parser.add_argument("--weight_temporal", type=float, default=0.5, help="Temporal consistency weight")
    parser.add_argument("--weight_routing_entropy", type=float, default=0.01, help="Routing entropy weight")
    parser.add_argument("--weight_routing_load_balance", type=float, default=0.05, help="Routing load balance weight")

    # Data config
    parser.add_argument("--instrument_data_path", type=str, default=None,
                      help="Path to instrument annotations/labels")
    parser.add_argument("--use_roi_loss", action="store_true", default=True,
                      help="Use ROI-weighted loss")
    parser.add_argument("--use_identity_loss", action="store_true", default=False,
                      help="Use identity preservation loss (requires classifier)")
    parser.add_argument("--use_temporal_loss", action="store_true", default=False,
                      help="Use temporal consistency loss (requires flow model)")

    # Pretrained weights
    parser.add_argument("--base_lora_weights", type=str, default=None,
                      help="Path to base LoRA weights (for stage B)")
    parser.add_argument("--lora_moe_weights", type=str, default=None,
                      help="Path to LoRA-MoE weights (for resuming or stage C)")

    return parser


def main():
    """Main training entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    # Load from config file if provided and set defaults similar to WAN trainer
    args = read_config_from_file(args, parser)
    # Prefer wandb by default unless explicitly set
    if getattr(args, "log_with", None) is None:
        args.log_with = "wandb"
    args.dit_dtype = None  # automatically detected
    if getattr(args, "mixed_precision", None) is None:
        args.mixed_precision = "bf16"  # default to bf16 for WAN2.2 bf16 weights
    if getattr(args, "vae_dtype", None) is None:
        args.vae_dtype = "bfloat16"  # default for VAE

    # Create and run trainer
    trainer = WanLoRAMoETrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
