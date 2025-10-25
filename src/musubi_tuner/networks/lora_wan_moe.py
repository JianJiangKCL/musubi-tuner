"""
WAN-specific LoRA-MoE integration.
Extends lora_wan.py to support Mixture of Experts for instrument-specific adaptation.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
import re

from .lora_moe import LoRAMoEModule, InstrumentRouter, LoRAMoENetwork


# Target modules for LoRA-MoE (same as standard WAN LoRA)
WAN_MOE_TARGET_REPLACE_MODULES = ["WanAttentionBlock"]

# Exclusion patterns (avoid LoRA on embeddings, norms, etc.)
WAN_MOE_EXCLUDE_PATTERNS = [
    r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*"
]


class WanLoRAMoENetwork(LoRAMoENetwork):
    """
    WAN-specific LoRA-MoE network manager.

    Handles:
    - Selective layer adaptation (top 6-8 DiT blocks)
    - Projection-specific ranks (Q=8, K/V/O=4, FFN=4)
    - Integration with WAN2.2 dual-model architecture
    - Timestep-selective expert application
    """

    def __init__(
        self,
        model: nn.Module,
        lora_config: Dict,
        router_config: Optional[Dict] = None,
        target_block_indices: Optional[List[int]] = None,
        projection_ranks: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            model: WanModel instance
            lora_config: LoRA configuration dict with keys:
                - lora_dim: Base LoRA rank (default: 4)
                - alpha: Scaling factor (default: 1.0)
                - num_experts: Number of experts (default: 4)
                - expert_names: List of expert names
                - use_base_lora: Whether to use base LoRA (default: True)
                - dropout: Dropout rate (default: 0.0)
                - rank_dropout: Rank dropout rate (default: 0.0)
                - module_dropout: Module dropout rate (default: 0.0)
            router_config: Router configuration dict
            target_block_indices: Indices of DiT blocks to adapt (e.g., [32, 33, ..., 39] for last 8)
                                 If None, adapts last 8 blocks
            projection_ranks: Dict mapping projection names to ranks, e.g., {"q": 8, "k": 4, "v": 4, "o": 4, "ffn": 4}
        """
        # Extract config
        self.lora_dim = lora_config.get("lora_dim", 4)
        self.alpha = lora_config.get("alpha", 1.0)
        self.num_experts = lora_config.get("num_experts", 4)
        self.expert_names = lora_config.get("expert_names", [
            "Scissors", "Hook/Electrocautery", "Suction", "Other"
        ])
        self.use_base_lora = lora_config.get("use_base_lora", True)
        self.dropout = lora_config.get("dropout", 0.0)
        self.rank_dropout = lora_config.get("rank_dropout", 0.0)
        self.module_dropout = lora_config.get("module_dropout", 0.0)

        # Target blocks (default: last 8 blocks for 40-layer model)
        self.target_block_indices = target_block_indices
        if target_block_indices is None:
            # Assume 40-layer model, adapt last 8
            self.target_block_indices = list(range(32, 40))

        # Projection-specific ranks
        self.projection_ranks = projection_ranks or {
            "q": 8,  # Query: most impactful
            "k": 4,
            "v": 4,
            "o": 4,  # Output projection
            "ffn": 4,  # FFN (only last 2-3 blocks)
        }

        # FFN target blocks (last 2-3 blocks)
        self.ffn_target_indices = self.target_block_indices[-3:]

        # Initialize router
        router_config = router_config or {}
        self.router = InstrumentRouter(
            num_experts=self.num_experts,
            **router_config
        )

        # Store model reference
        self.model = model
        self.lora_moe_modules: List[LoRAMoEModule] = []

        # Module name to rank mapping
        self.module_ranks: Dict[str, int] = {}

    def _should_exclude(self, module_name: str) -> bool:
        """Check if module should be excluded from LoRA."""
        for pattern in WAN_MOE_EXCLUDE_PATTERNS:
            if re.match(pattern, module_name):
                return True
        return False

    def _get_rank_for_projection(self, module_name: str) -> int:
        """Get LoRA rank for specific projection based on name."""
        # Determine projection type from module name
        if ".q." in module_name or module_name.endswith(".q"):
            return self.projection_ranks.get("q", self.lora_dim)
        elif ".k." in module_name or module_name.endswith(".k"):
            return self.projection_ranks.get("k", self.lora_dim)
        elif ".v." in module_name or module_name.endswith(".v"):
            return self.projection_ranks.get("v", self.lora_dim)
        elif ".o." in module_name or module_name.endswith(".o"):
            return self.projection_ranks.get("o", self.lora_dim)
        elif "ffn" in module_name:
            return self.projection_ranks.get("ffn", self.lora_dim)
        else:
            return self.lora_dim

    def apply_lora_moe(self):
        """
        Apply LoRA-MoE to target DiT blocks in WAN model.

        Strategy:
        1. Target only specific block indices (last 6-8 blocks)
        2. Apply to Q/K/V/O projections in self-attn and cross-attn
        3. Apply to FFN first linear in last 2-3 blocks
        4. Use projection-specific ranks
        """
        # Access model blocks
        if not hasattr(self.model, 'blocks'):
            raise ValueError("Model must have 'blocks' attribute (list of WanAttentionBlock)")

        blocks = self.model.blocks
        total_blocks = len(blocks)

        print(f"Total DiT blocks: {total_blocks}")
        print(f"Target block indices: {self.target_block_indices}")
        print(f"FFN target indices: {self.ffn_target_indices}")

        for block_idx in self.target_block_indices:
            if block_idx >= total_blocks:
                print(f"Warning: block index {block_idx} out of range (total: {total_blocks})")
                continue

            block = blocks[block_idx]
            block_name = f"blocks.{block_idx}"

            # Self-attention projections
            if hasattr(block, 'attn1') or hasattr(block, 'self_attn'):
                attn = getattr(block, 'attn1', None) or getattr(block, 'self_attn', None)
                for proj_name in ['q', 'k', 'v', 'o']:
                    if hasattr(attn, proj_name):
                        proj_module = getattr(attn, proj_name)
                        if isinstance(proj_module, nn.Linear):
                            full_name = f"{block_name}.attn1.{proj_name}"
                            if not self._should_exclude(full_name):
                                self._replace_with_lora_moe(
                                    parent=attn,
                                    child_name=proj_name,
                                    child_module=proj_module,
                                    module_name=full_name,
                                )

            # Cross-attention projections
            if hasattr(block, 'attn2') or hasattr(block, 'cross_attn'):
                attn = getattr(block, 'attn2', None) or getattr(block, 'cross_attn', None)
                for proj_name in ['q', 'k', 'v', 'o']:
                    if hasattr(attn, proj_name):
                        proj_module = getattr(attn, proj_name)
                        if isinstance(proj_module, nn.Linear):
                            full_name = f"{block_name}.attn2.{proj_name}"
                            if not self._should_exclude(full_name):
                                self._replace_with_lora_moe(
                                    parent=attn,
                                    child_name=proj_name,
                                    child_module=proj_module,
                                    module_name=full_name,
                                )

            # FFN first linear (only in last 2-3 blocks)
            if block_idx in self.ffn_target_indices:
                if hasattr(block, 'ffn') or hasattr(block, 'mlp'):
                    ffn = getattr(block, 'ffn', None) or getattr(block, 'mlp', None)

                    # FFN typically has structure: Linear -> Activation -> Linear
                    # We target the first Linear
                    if hasattr(ffn, 'fc1') or hasattr(ffn, '0'):
                        ffn_linear = getattr(ffn, 'fc1', None) or getattr(ffn, '0', None)
                        if isinstance(ffn_linear, nn.Linear):
                            full_name = f"{block_name}.ffn.fc1"
                            if not self._should_exclude(full_name):
                                self._replace_with_lora_moe(
                                    parent=ffn,
                                    child_name='fc1' if hasattr(ffn, 'fc1') else '0',
                                    child_module=ffn_linear,
                                    module_name=full_name,
                                )

        print(f"Applied LoRA-MoE to {len(self.lora_moe_modules)} projections across {len(self.target_block_indices)} blocks")

        # Print parameter count
        total_params = sum(
            sum(p.numel() for p in module.parameters() if p.requires_grad)
            for module in self.lora_moe_modules
        )
        print(f"Total trainable LoRA-MoE parameters: {total_params / 1e6:.2f}M")

    def _replace_with_lora_moe(
        self,
        parent: nn.Module,
        child_name: str,
        child_module: nn.Module,
        module_name: str,
    ):
        """Replace a module with LoRA-MoE wrapper."""
        # Get rank for this projection
        rank = self._get_rank_for_projection(module_name)

        # Create LoRA-MoE module
        lora_moe = LoRAMoEModule(
            org_module=child_module,
            lora_dim=rank,
            alpha=self.alpha,
            num_experts=self.num_experts,
            expert_names=self.expert_names,
            dropout=self.dropout,
            rank_dropout=self.rank_dropout,
            module_dropout=self.module_dropout,
            use_base_lora=self.use_base_lora,
        )

        # Align device/dtype with original child module
        try:
            dev = child_module.weight.device
            dt = child_module.weight.dtype
            lora_moe.to(device=dev, dtype=dt)
        except Exception:
            pass

        # Replace
        setattr(parent, child_name, lora_moe)
        self.lora_moe_modules.append(lora_moe)
        self.module_ranks[module_name] = rank

        print(f"  Replaced {module_name} with LoRA-MoE (rank={rank})")

    def prepare_for_training_stage(self, stage: str, train_router: bool = False):
        """
        Prepare network for specific training stage.

        Stages:
        - "stage_a": Base LoRA only (freeze experts and router)
        - "stage_b": Train experts + router simultaneously (freeze base LoRA)
                    This is the recommended 2-stage approach after vanilla LoRA training
        - "stage_b_experts_only": Train experts only with fixed rule-based routing (freeze base)
        - "stage_c": Train router only (freeze base + experts) - rarely used

        Args:
            stage: Training stage identifier
            train_router: If True in stage_b, enables router training (default for stage_b)
        """
        if stage == "stage_a":
            # Stage A: Train base LoRA only (this is vanilla LoRA training)
            print("Stage A: Training base LoRA (freezing experts and router)")
            for module in self.lora_moe_modules:
                # Enable base LoRA gradients
                if module.base_lora_down is not None:
                    for param in module.base_lora_down.parameters():
                        param.requires_grad = True
                    for param in module.base_lora_up.parameters():
                        param.requires_grad = True

                # Freeze experts
                for expert_down in module.expert_lora_down:
                    for param in expert_down.parameters():
                        param.requires_grad = False
                for expert_up in module.expert_lora_up:
                    for param in expert_up.parameters():
                        param.requires_grad = False

                # Freeze timestep gate
                if module.timestep_gate is not None:
                    for param in module.timestep_gate.parameters():
                        param.requires_grad = False

            # Freeze router
            for param in self.router.parameters():
                param.requires_grad = False

        elif stage == "stage_b":
            # Stage B: Train experts + router simultaneously (RECOMMENDED)
            # This is the main MoE training stage after vanilla LoRA
            print("Stage B: Training expert LoRAs + Router simultaneously (freezing base LoRA)")
            for module in self.lora_moe_modules:
                # Freeze base LoRA (use pretrained vanilla LoRA)
                if module.base_lora_down is not None:
                    for param in module.base_lora_down.parameters():
                        param.requires_grad = False
                    for param in module.base_lora_up.parameters():
                        param.requires_grad = False

                # Enable expert gradients
                for expert_down in module.expert_lora_down:
                    for param in expert_down.parameters():
                        param.requires_grad = True
                for expert_up in module.expert_lora_up:
                    for param in expert_up.parameters():
                        param.requires_grad = True

                # Enable timestep gate
                if module.timestep_gate is not None:
                    for param in module.timestep_gate.parameters():
                        param.requires_grad = True

            # Enable router (default behavior for stage_b)
            if train_router or self.router.routing_mode == "learned":
                print("  - Router training: ENABLED")
                for param in self.router.parameters():
                    param.requires_grad = True
            else:
                print("  - Router training: DISABLED (using rule-based routing)")
                for param in self.router.parameters():
                    param.requires_grad = False

        elif stage == "stage_b_experts_only":
            # Alternative: Train experts only with rule-based routing
            print("Stage B (Experts Only): Training expert LoRAs with fixed routing (freezing base)")
            for module in self.lora_moe_modules:
                # Freeze base LoRA
                if module.base_lora_down is not None:
                    for param in module.base_lora_down.parameters():
                        param.requires_grad = False
                    for param in module.base_lora_up.parameters():
                        param.requires_grad = False

                # Enable expert gradients
                for expert_down in module.expert_lora_down:
                    for param in expert_down.parameters():
                        param.requires_grad = True
                for expert_up in module.expert_lora_up:
                    for param in expert_up.parameters():
                        param.requires_grad = True

                # Enable timestep gate
                if module.timestep_gate is not None:
                    for param in module.timestep_gate.parameters():
                        param.requires_grad = True

            # Freeze router (use rule-based)
            for param in self.router.parameters():
                param.requires_grad = False

        elif stage == "stage_c":
            # Stage C: Train router only (rarely used, only if you want to fine-tune router later)
            print("Stage C: Training router only (freezing all LoRAs)")
            for module in self.lora_moe_modules:
                # Freeze everything
                for param in module.parameters():
                    param.requires_grad = False

            # Enable router
            for param in self.router.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters in LoRA-MoE modules and router."""
        params = []

        # LoRA-MoE module params
        for module in self.lora_moe_modules:
            params.extend([p for p in module.parameters() if p.requires_grad])

        # Router params
        params.extend([p for p in self.router.parameters() if p.requires_grad])

        return params

    def save_lora_moe_weights(self, save_path: str):
        """Save only LoRA-MoE weights (not full model)."""
        state_dict = {}

        # Save LoRA-MoE module weights
        for i, module in enumerate(self.lora_moe_modules):
            state_dict[f"lora_moe_{i}"] = module.state_dict()

        # Save router weights
        state_dict["router"] = self.router.state_dict()

        # Save config
        state_dict["config"] = {
            "lora_dim": self.lora_dim,
            "alpha": self.alpha,
            "num_experts": self.num_experts,
            "expert_names": self.expert_names,
            "target_block_indices": self.target_block_indices,
            "projection_ranks": self.projection_ranks,
            "module_ranks": self.module_ranks,
        }

        torch.save(state_dict, save_path)
        print(f"Saved LoRA-MoE weights to {save_path}")

    def load_lora_moe_weights(self, load_path: str):
        """Load LoRA-MoE weights."""
        # Support safetensors and PyTorch 2.6+ weights_only behavior
        if load_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(load_path)
        else:
            try:
                state_dict = torch.load(load_path, map_location="cpu", weights_only=False)
            except TypeError:
                # Older PyTorch without weights_only
                state_dict = torch.load(load_path, map_location="cpu")

        # Load LoRA-MoE modules
        for i, module in enumerate(self.lora_moe_modules):
            if f"lora_moe_{i}" in state_dict:
                module.load_state_dict(state_dict[f"lora_moe_{i}"])

        # Load router
        if "router" in state_dict:
            self.router.load_state_dict(state_dict["router"])

        print(f"Loaded LoRA-MoE weights from {load_path}")


def create_wan_lora_moe(
    model: nn.Module,
    lora_config: Optional[Dict] = None,
    router_config: Optional[Dict] = None,
    target_blocks: str = "last_8",  # "last_8", "last_6", or list of indices
) -> WanLoRAMoENetwork:
    """
    Factory function to create WAN LoRA-MoE network.

    Args:
        model: WanModel instance
        lora_config: LoRA configuration (see WanLoRAMoENetwork.__init__)
        router_config: Router configuration
        target_blocks: Which blocks to adapt
            - "last_8": Last 8 blocks (indices 32-39 for 40-layer model)
            - "last_6": Last 6 blocks (indices 34-39)
            - List[int]: Custom indices

    Returns:
        WanLoRAMoENetwork instance with LoRA-MoE applied
    """
    # Default configs
    if lora_config is None:
        lora_config = {
            "lora_dim": 4,
            "alpha": 1.0,
            "num_experts": 4,
            "expert_names": ["Scissors", "Hook/Electrocautery", "Suction", "Other"],
            "use_base_lora": True,
            "dropout": 0.0,
            "rank_dropout": 0.0,
            "module_dropout": 0.0,
        }

    if router_config is None:
        router_config = {
            "routing_mode": "rule_based",
            "top_k": 2,
            "temperature": 0.7,
            "ema_beta": 0.9,
        }

    # Determine target block indices
    if isinstance(target_blocks, str):
        if target_blocks == "last_8":
            target_block_indices = list(range(32, 40))
        elif target_blocks == "last_6":
            target_block_indices = list(range(34, 40))
        else:
            raise ValueError(f"Unknown target_blocks preset: {target_blocks}")
    else:
        target_block_indices = target_blocks

    # Create network
    lora_moe_net = WanLoRAMoENetwork(
        model=model,
        lora_config=lora_config,
        router_config=router_config,
        target_block_indices=target_block_indices,
    )

    # Apply LoRA-MoE
    lora_moe_net.apply_lora_moe()

    return lora_moe_net
