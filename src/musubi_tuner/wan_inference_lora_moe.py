"""
Inference script for LoRA-MoE adapted WAN2.2 model.

Supports:
- Instrument-specific video generation
- Automatic routing based on text prompts or instrument hints
- Composition of base LoRA + expert LoRAs
- Timestep-selective expert application

Usage:
    python wan_inference_lora_moe.py \\
        --task t2v-A14B \\
        --lora_moe_weights outputs/lora_moe_stage_b/lora_moe_final.safetensors \\
        --prompt "抓钳抓取组织。电凝钩切割。" \\
        --instrument_hint "Hook/Electrocautery" \\
        --output_path output.mp4 \\
        --alpha_base 0.7 \\
        --alpha_expert 1.0
"""

import argparse
import torch
import json as _json
import os as _os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging
import os
from safetensors.torch import load_file

from musubi_tuner.wan.configs import WAN_CONFIGS
from musubi_tuner.wan.modules.model4infer import load_wan_model
from musubi_tuner.wan.modules.vae import WanVAE
from musubi_tuner.wan.modules.t5 import T5EncoderModel
from musubi_tuner.wan.modules.clip import CLIPModel
from musubi_tuner.networks.lora_wan_moe import WanLoRAMoENetwork
from musubi_tuner.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.hv_generate_video import save_videos_grid
import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LoRAMoEInference:
    """
    Inference wrapper for LoRA-MoE adapted WAN2.2.
    """

    def __init__(
        self,
        task: str = "i2v-A14B",
        dit_path: str = None,
        dit_high_noise_path: str = None,
        vae_path: str = None,
        t5_path: str = None,
        clip_path: str = None,
        lora_moe_weights: str = None,
        alpha_base: float = 0.7,
        alpha_expert: float = 1.0,
        device: str = "cuda",
        lora_moe_config_file: Optional[str] = None,
        timestep_boundary: float = 0.9,
    ):
        """
        Args:
            task: WAN task name (e.g., "i2v-A14B")
            dit_path: Path to DiT model weights
            dit_high_noise_path: Path to high noise DiT weights
            vae_path: Path to VAE weights
            t5_path: Path to T5 encoder weights
            clip_path: Path to CLIP model
            lora_moe_weights: Path to LoRA-MoE weights
            alpha_base: Scaling for base LoRA
            alpha_expert: Scaling for expert LoRAs
            device: Device to run inference on
        """
        self.task = task
        self.config = WAN_CONFIGS[task]
        self.device = torch.device(device)
        self.alpha_base = alpha_base
        self.alpha_expert = alpha_expert
        self.timestep_boundary = timestep_boundary

        # Load DiT model
        logger.info(f"Loading WAN DiT model for task: {task}")
        self.dit = load_wan_model(
            config=self.config,
            device=self.device,
            dit_path=dit_path,
            attn_mode="torch",
            split_attn=False,
            loading_device=self.device,
            dit_weight_dtype=torch.bfloat16,
            fp8_scaled=False,
        )

        # WAN 2.2 dual-DiT support
        self.dit_low = self.dit
        self.dit_high = None
        self.dit_active = self.dit_low
        if dit_high_noise_path:
            logger.info(f"Loading high-noise DiT model for WAN 2.2 from {dit_high_noise_path}")
            self.dit_high = load_wan_model(
                config=self.config,
                device=self.device,
                dit_path=dit_high_noise_path,
                attn_mode="torch",
                split_attn=False,
                loading_device=self.device,
                dit_weight_dtype=torch.bfloat16,
                fp8_scaled=False,
            )
            # Start with high-noise model at the beginning of sampling (t near 1.0)
            self.dit_active = self.dit_high

        # Load VAE
        logger.info(f"Loading VAE from {vae_path}")
        self.vae = WanVAE(vae_path=vae_path, device=self.device, dtype=torch.bfloat16)

        # Load T5
        logger.info(f"Loading T5 encoder from {t5_path}")
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=self.device,
            weight_path=t5_path,
        )

        # Load CLIP (for i2v)
        if clip_path and "i2v" in task:
            # allow directory input by appending expected checkpoint filename
            if os.path.isdir(clip_path):
                weight_path = os.path.join(clip_path, self.config.clip_checkpoint)
            else:
                weight_path = clip_path
            if not os.path.exists(weight_path):
                logger.warning(f"CLIP weights not found at {weight_path}; proceeding without CLIP.")
                self.clip = None
            else:
                logger.info(f"Loading CLIP from {weight_path}")
                self.clip = CLIPModel(
                    dtype=self.config.clip_dtype,
                    device=self.device,
                    weight_path=weight_path,
                )
        else:
            self.clip = None

        # Create LoRA-MoE network
        self.lora_moe_network = None
        if lora_moe_weights:
            logger.info(f"Preparing LoRA-MoE from {lora_moe_weights}")
            checkpoint = None  # network loader will be used

            # Base defaults, will be overrided by config yaml
            lora_config = {
                "lora_dim": 4,
                "alpha": 1.0,
                "num_experts": 6,
                "expert_names": ["bipolar", "clipper", "grasper", "hook", "irrigator", "scissors"],
                # Stage1 base LoRA merged into DiT; disable base_lora
                "use_base_lora": False,
                "dropout": 0.0,
                "rank_dropout": 0.0,
                "module_dropout": 0.0,
                # Optional fields for inference application
                "target_blocks": "last_8",
                "projection_ranks": {"q": 8, "k": 4, "v": 4, "o": 4, "ffn": 4},
            }

            router_config = {
                "routing_mode": "learned",
                "top_k": 2,
                "temperature": 0.7,
                "learnable_hidden_dim": 64,
                "input_dim": None,
            }

            # Discover default config file if not provided
            if not lora_moe_config_file:
                candidate_paths = [
                    os.environ.get("LORA_MOE_CONFIG_FILE", None),
                    os.path.join(os.getcwd(), "Lap", "config", "lora_moe.yaml"),
                    "/mnt/cfs/jj/proj/musubi-tuner/Lap/config/lora_moe.yaml",
                ]
                for _p in candidate_paths:
                    if _p and os.path.exists(_p):
                        lora_moe_config_file = _p
                        break

            # Optional: override from external config file
            if lora_moe_config_file:
                def _load_structured_config(path: str) -> dict:
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "r") as f:
                        if ext in [".yaml", ".yml"]:
                            import yaml  # type: ignore
                            return yaml.safe_load(f) or {}
                        elif ext == ".toml":
                            try:
                                import tomllib
                                return tomllib.loads(f.read())
                            except Exception:
                                import tomli  # type: ignore
                                f.seek(0)
                                return tomli.loads(f.read())
                        else:
                            import json as _json
                            return _json.load(f)

                try:
                    cfg = _load_structured_config(lora_moe_config_file)
                    lora_cfg = cfg.get("lora_moe", {})
                    router_cfg = cfg.get("router", {})
                    if isinstance(lora_cfg, dict):
                        # Allow target_blocks and projection_ranks to override
                        lora_config.update({k: v for k, v in lora_cfg.items() if v is not None})
                    if isinstance(router_cfg, dict):
                        router_config.update({k: v for k, v in router_cfg.items() if v is not None})
                    logger.info(f"Loaded LoRA-MoE inference config from {lora_moe_config_file}")
                except Exception as e:
                    logger.warning(f"Failed to load LoRA-MoE inference config '{lora_moe_config_file}': {e}")

            # Optional: override from weights metadata/config (takes highest priority to avoid rank mismatch)
            # Supports safetensors metadata['config'] (json string) or torch checkpoint with ['config']
            try:
                from safetensors import safe_open as _safe_open
                try:
                    with _safe_open(lora_moe_weights, framework="pt", device="cpu") as f:
                        md = f.metadata()
                        if md and "config" in md:
                            try:
                                cfg_from_weights = _json.loads(md["config"]) or {}
                            except Exception:
                                cfg_from_weights = {}
                        else:
                            cfg_from_weights = {}
                except Exception:
                    cfg_from_weights = {}

                if not cfg_from_weights:
                    # try torch.load
                    try:
                        st = torch.load(lora_moe_weights, map_location="cpu")
                        cfg_from_weights = st.get("config", {}) if isinstance(st, dict) else {}
                    except Exception:
                        cfg_from_weights = {}

                if isinstance(cfg_from_weights, dict) and cfg_from_weights:
                    # Only pick keys we know
                    for k in ["lora_dim", "alpha", "num_experts", "expert_names", "projection_ranks", "target_block_indices"]:
                        if k in cfg_from_weights and cfg_from_weights[k] is not None:
                            if k == "target_block_indices":
                                # translate indices back to preset if matches last_8/last_6
                                tbi = cfg_from_weights[k]
                                if tbi == list(range(32, 40)):
                                    lora_config["target_blocks"] = "last_8"
                                elif tbi == list(range(34, 40)):
                                    lora_config["target_blocks"] = "last_6"
                                else:
                                    lora_config["target_blocks"] = tbi
                            else:
                                lora_config[k] = cfg_from_weights[k]
                    logger.info("Applied LoRA-MoE config from weights metadata to avoid rank mismatch")
            except Exception as e:
                logger.warning(f"Could not read config from weights: {e}")

            # Determine target blocks
            target_blocks = lora_config.get("target_blocks", "last_8")
            if isinstance(target_blocks, str):
                if target_blocks == "last_8":
                    target_block_indices = list(range(32, 40))
                elif target_blocks == "last_6":
                    target_block_indices = list(range(34, 40))
                else:
                    raise ValueError(f"Unknown target_blocks preset: {target_blocks}")
            else:
                target_block_indices = target_blocks

            # Ensure router input dim matches current experts (no text conditioning path)
            try:
                router_config["input_dim"] = lora_config.get("num_experts", 6)
            except Exception:
                router_config["input_dim"] = 6

            if True:
                # Pass projection ranks if provided
                projection_ranks = lora_config.get("projection_ranks", None)

                if self.dit_high is not None:
                    # Dual models: apply LoRA to both
                    self.lora_moe_network_low = WanLoRAMoENetwork(
                        model=self.dit_low,
                        lora_config=lora_config,
                        router_config=router_config,
                        target_block_indices=target_block_indices,
                        projection_ranks=projection_ranks,
                    )
                    self.lora_moe_network_low.apply_lora_moe()
                    # Ensure router on device
                    try:
                        self.lora_moe_network_low.router.to(self.device)
                    except Exception:
                        pass
                    # Load LoRA-MoE weights via network loader (supports safetensors/torch)
                    try:
                        self.lora_moe_network_low.load_lora_moe_weights(lora_moe_weights)
                        logger.info("Loaded LoRA-MoE weights into low-noise model")
                    except Exception as e:
                        logger.error(f"Failed to load LoRA-MoE into low-noise model: {e}")

                    self.lora_moe_network_high = WanLoRAMoENetwork(
                        model=self.dit_high,
                        lora_config=lora_config,
                        router_config=router_config,
                        target_block_indices=target_block_indices,
                        projection_ranks=projection_ranks,
                    )
                    self.lora_moe_network_high.apply_lora_moe()
                    try:
                        self.lora_moe_network_high.router.to(self.device)
                    except Exception:
                        pass
                    try:
                        self.lora_moe_network_high.load_lora_moe_weights(lora_moe_weights)
                        logger.info("Loaded LoRA-MoE weights into high-noise model")
                    except Exception as e:
                        logger.error(f"Failed to load LoRA-MoE into high-noise model: {e}")

                    self.lora_moe_network = None
                else:
                    # Single model
                    self.lora_moe_network = WanLoRAMoENetwork(
                        model=self.dit,
                        lora_config=lora_config,
                        router_config=router_config,
                        target_block_indices=target_block_indices,
                        projection_ranks=projection_ranks,
                    )
                    self.lora_moe_network.apply_lora_moe()
                    try:
                        self.lora_moe_network.router.to(self.device)
                    except Exception:
                        pass
                    try:
                        self.lora_moe_network.load_lora_moe_weights(lora_moe_weights)
                        logger.info("Loaded LoRA-MoE weights into model")
                    except Exception as e:
                        logger.error(f"Failed to load LoRA-MoE weights: {e}")
                    self._set_lora_scales(alpha_base, alpha_expert)
            else:
                self.lora_moe_network = None
        else:
            logger.warning("No LoRA-MoE weights provided, using base model")

    def _load_lora_moe_weights(self, state_dict: dict):
        """Load LoRA-MoE weights from checkpoint."""
        # Load state dict into model
        # The state_dict should contain keys like:
        # - lora_moe.blocks.32.attn1.q.base_lora_down.weight
        # - lora_moe.blocks.32.attn1.q.expert_lora_down.0.weight
        # - router.router_mlp.0.weight

        # For simplicity, we use load_state_dict with strict=False
        # to handle partial matches
        missing, unexpected = self.dit.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded LoRA-MoE weights: {len(state_dict)} keys")
        if missing:
            logger.debug(f"Missing keys: {len(missing)}")
        if unexpected:
            logger.debug(f"Unexpected keys: {len(unexpected)}")

    def _set_lora_scales(self, alpha_base: float, alpha_expert: float):
        """Set LoRA scaling factors for inference."""
        if self.lora_moe_network is None:
            return

        for module in self.lora_moe_network.lora_moe_modules:
            # Scale base LoRA
            if module.base_lora_down is not None:
                module.scale = alpha_base / module.lora_dim

            # Note: Expert scaling is applied via gates, so we use alpha_expert
            # when computing the final mixture
            # This is already handled in the forward pass with routing gates

    def infer_instrument_from_prompt(self, prompt: str) -> str:
        """
        Infer likely instrument from Chinese prompt text.

        Args:
            prompt: Text prompt in Chinese

        Returns:
            instrument: Predicted instrument name
        """
        # Simple keyword matching for surgical instruments
        instrument_keywords = {
            "bipolar": ["双极电凝钳", "双极电凝", "双极", "电凝钳"],
            "clipper": ["钛夹钳", "夹钳", "夹闭", "夹"],
            "grasper": ["抓钳", "抓取", "镊子"],
            "hook": ["电凝钩", "电凝", "钩"],
            "irrigator": ["吸引器", "吸引", "吸"],
            "scissors": ["剪刀", "剪切"],
        }

        # Count matches
        scores = {inst: 0 for inst in instrument_keywords.keys()}
        for inst, keywords in instrument_keywords.items():
            for keyword in keywords:
                if keyword in prompt:
                    scores[inst] += 1

        # Get max score
        max_inst = max(scores.items(), key=lambda x: x[1])

        if max_inst[1] == 0:
            # No match, default to grasper
            return "grasper"

        return max_inst[0]

    def create_instrument_logits(
        self,
        instrument_hint: Optional[str] = None,
        prompt: Optional[str] = None,
        confidence: float = 0.8,
    ) -> torch.Tensor:
        """
        Create instrument logits for routing.

        Args:
            instrument_hint: Explicit instrument name (overrides prompt)
            prompt: Text prompt (used if no hint)
            confidence: Confidence level for predicted instrument

        Returns:
            logits: [num_experts] tensor
        """
        expert_names = self.lora_moe_network.expert_names if self.lora_moe_network is not None else [
            "bipolar", "clipper", "grasper", "hook", "irrigator", "scissors"
        ]

        # Determine instrument
        if instrument_hint is not None:
            instrument = instrument_hint
        elif prompt is not None:
            instrument = self.infer_instrument_from_prompt(prompt)
        else:
            # Default to grasper
            instrument = "grasper"

        # Find expert index
        if instrument in expert_names:
            expert_idx = expert_names.index(instrument)
        else:
            expert_idx = expert_names.index("grasper")

        # Create logits (peaked at predicted expert)
        logits = torch.ones(len(expert_names)) * (1 - confidence) / (len(expert_names) - 1)
        logits[expert_idx] = confidence

        logger.info(f"Routing to instrument: {instrument} (index {expert_idx})")

        return logits.to(self.device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        instrument_hint: Optional[str] = None,
        image_path: Optional[str] = None,
        num_frames: int = 81,
        video_size: tuple = (256, 256),
        num_steps: int = 40,
        guidance_scale: float = 7.5,
        flow_shift: float = 7.0,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate video with LoRA-MoE.

        Args:
            prompt: Text prompt
            instrument_hint: Explicit instrument hint for routing (e.g., "Scissors", "Hook/Electrocautery")
            image_path: Path to start image for i2v (required for i2v task)
            num_frames: Number of frames to generate
            video_size: Video size (height, width)
            num_steps: Number of diffusion steps
            guidance_scale: Guidance scale for CFG
            flow_shift: Flow shift parameter
            seed: Random seed
            output_path: Path to save video (optional)

        Returns:
            video: Generated video as numpy array [F, H, W, C]
        """
        # Set seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info("=== LoRA-MoE Video Generation ===")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Instrument: {instrument_hint or 'auto-detected'}")
        logger.info(f"Frames: {num_frames}, Steps: {num_steps}, CFG: {guidance_scale}")
        logger.info(f"Seed: {seed}")

        # Create instrument logits for routing
        instrument_logits = self.create_instrument_logits(
            instrument_hint=instrument_hint,
            prompt=prompt,
        )

        # Set routing gates on available networks
        if self.lora_moe_network is not None:
            self.lora_moe_network.set_routing_gates(
                instrument_logits=instrument_logits.unsqueeze(0),
                text_embeddings=None,
            )
        if hasattr(self, "lora_moe_network_low") and self.lora_moe_network_low is not None:
            self.lora_moe_network_low.set_routing_gates(
                instrument_logits=instrument_logits.unsqueeze(0),
                text_embeddings=None,
            )
        if hasattr(self, "lora_moe_network_high") and self.lora_moe_network_high is not None:
            self.lora_moe_network_high.set_routing_gates(
                instrument_logits=instrument_logits.unsqueeze(0),
                text_embeddings=None,
            )

        # Encode text prompt
        logger.info("Encoding text prompt...")
        context = self.text_encoder([prompt], self.device)
        context_null = self.text_encoder([""], self.device)

        # Calculate latent dimensions
        height, width = video_size
        lat_f = (num_frames - 1) // self.config.vae_stride[0] + 1
        lat_h = height // self.config.vae_stride[1]
        lat_w = width // self.config.vae_stride[2]
        latent_shape = (1, 16, lat_f, lat_h, lat_w)

        # Handle image conditioning for i2v (prepare y and optional clip_fea)
        clip_fea = None
        y = None
        if "i2v" in self.task and image_path:
            logger.info(f"Loading start image from {image_path}")
            from PIL import Image
            import torchvision.transforms.functional as TF

            pil_image = Image.open(image_path).convert("RGB")
            pil_image = pil_image.resize((width, height))
            # [-1, 1] normalized tensor, CHW
            img_tensor = TF.to_tensor(pil_image).to(self.device).sub_(0.5).div_(0.5)
            # CFHW
            img_cf = img_tensor.unsqueeze(1)

            # encode image to latent space with VAE, padding frames-1 with zeros
            padding_frames = num_frames - 1
            if padding_frames > 0:
                pad_zeros = torch.zeros(3, padding_frames, height, width, device=self.device, dtype=img_cf.dtype)
                img_cf = torch.cat([img_cf, pad_zeros], dim=1)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                y_latent = self.vae.encode([img_cf])[0]

            # mask channels (4) + image latent (16) => total 36 channels expected by model
            msk = torch.zeros(4, lat_f, lat_h, lat_w, device=self.device, dtype=y_latent.dtype)
            msk[:, 0] = 1
            y = torch.cat([msk, y_latent], dim=0)

            # Optional CLIP image features
            if self.clip is not None:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    clip_fea = self.clip.visual([img_tensor.unsqueeze(1)])
            else:
                logger.warning("CLIP weights not provided; proceeding without CLIP image conditioning.")

        # Initialize noise
        logger.info("Initializing noise...")
        latents = torch.randn(latent_shape, generator=generator, device=self.device, dtype=torch.bfloat16)

        # Setup scheduler
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
        )
        scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = scheduler.timesteps

        # Denoising loop
        logger.info("Starting denoising loop...")
        seq_len = lat_f * lat_h * lat_w // (self.config.patch_size[1] * self.config.patch_size[2])

        for i, t in enumerate(timesteps):
            # Set timestep for LoRA-MoE (normalized to [0, 1])
            timestep_normalized = 1.0 - (i / len(timesteps))
            if self.lora_moe_network is not None:
                self.lora_moe_network.set_timestep(timestep_normalized)
            if hasattr(self, "lora_moe_network_low") and self.lora_moe_network_low is not None:
                self.lora_moe_network_low.set_timestep(timestep_normalized)
            if hasattr(self, "lora_moe_network_high") and self.lora_moe_network_high is not None:
                self.lora_moe_network_high.set_timestep(timestep_normalized)

            # Switch active DiT based on timestep for WAN 2.2 dual models
            if self.dit_high is not None and self.dit_low is not None:
                use_high = timestep_normalized >= self.timestep_boundary
                desired = self.dit_high if use_high else self.dit_low
                if self.dit_active is not desired:
                    self.dit_active = desired

            # Prepare model inputs
            arg_c = {"context": context, "seq_len": seq_len, "y": [y] if y is not None else None, "clip_fea": clip_fea}
            arg_null = {"context": context_null, "seq_len": seq_len, "y": [y] if y is not None else None, "clip_fea": clip_fea}

            # Model prediction with CFG
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                # Prepare latent for model: list of [C, F, H, W]
                latent_for_model = latents.squeeze(0)  # [16, F, H, W]
                # Conditional prediction
                noise_pred_cond = self.dit_active([latent_for_model], t=t.unsqueeze(0), **arg_c)[0]
                # Unconditional prediction
                noise_pred_uncond = self.dit_active([latent_for_model], t=t.unsqueeze(0), **arg_null)[0]

            # Classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Scheduler step
            # Scheduler expects batch dimension
            latents = scheduler.step(noise_pred.unsqueeze(0), t, latents).prev_sample

            if (i + 1) % 10 == 0:
                logger.info(f"Denoising step {i+1}/{len(timesteps)}")

        # Decode latents to video
        logger.info("Decoding latents to video...")
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            video_list = self.vae.decode(latents)
            video = video_list[0]

        # Convert to numpy [F, H, W, C]
        video = video.squeeze(0).permute(1, 2, 3, 0).cpu().float().numpy()
        video = ((video + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        # Save video
        if output_path:
            logger.info(f"Saving video to {output_path}")
            # Convert back to tensor format for save_videos_grid: [B, C, F, H, W]
            video_tensor = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0 * 2 - 1
            save_videos_grid(video_tensor, output_path, fps=16)

        logger.info("Generation complete!")
        return video


def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser."""
    parser = argparse.ArgumentParser(
        description="LoRA-MoE Inference for WAN2.2 - Instrument-specific video generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python wan_inference_lora_moe.py \\
    --task i2v-A14B \\
    --dit outputs/merged_low_noise.safetensors \\
    --vae /path/to/Wan2.1_VAE.pth \\
    --t5 /path/to/models_t5_umt5-xxl-enc-bf16.pth \\
    --clip /path/to/clip-vit-large-patch14 \\
    --lora_moe_weights outputs/lora_moe_stage2/lora_moe_final.safetensors \\
    --prompt "抓钳抓取组织。电凝钩切割。" \\
    --instrument_hint "Hook/Electrocautery" \\
    --image_path start_frame.png \\
    --output_path output_video.mp4
        """
    )

    # Model config
    parser.add_argument("--task", type=str, default="i2v-A14B",
                      choices=list(WAN_CONFIGS.keys()),
                      help="WAN task name (default: i2v-A14B)")
    parser.add_argument("--dit", type=str, required=True,
                      help="Path to DiT weights")
    parser.add_argument("--dit_high_noise", type=str, default=None,
                      help="Path to high-noise DiT weights (WAN2.2, optional)")
    parser.add_argument("--vae", type=str, required=True,
                      help="Path to VAE weights")
    parser.add_argument("--t5", type=str, required=True,
                      help="Path to T5 encoder model")
    parser.add_argument("--clip", type=str, default=None,
                      help="Path to CLIP model (required for i2v tasks)")

    # LoRA-MoE config
    parser.add_argument("--lora_moe_weights", type=str, required=True,
                      help="Path to trained LoRA-MoE weights (.safetensors)")
    parser.add_argument("--alpha_base", type=float, default=0.7,
                      help="Scaling factor for base LoRA (default: 0.7)")
    parser.add_argument("--alpha_expert", type=float, default=1.0,
                      help="Scaling factor for expert LoRAs (default: 1.0)")
    parser.add_argument("--lora_moe_config_file", type=str, default=None,
                      help="Path to LoRA-MoE config file (YAML/TOML/JSON) for inference")

    # Generation config
    parser.add_argument("--prompt", type=str, required=True,
                      help="Text prompt (in Chinese)")
    parser.add_argument("--instrument_hint", type=str, default=None,
                      choices=["bipolar", "clipper", "grasper", "hook", "irrigator", "scissors"],
                      help="Explicit instrument hint for expert routing (optional)")
    parser.add_argument("--image_path", type=str, default=None,
                      help="Path to start image for i2v (required for i2v tasks)")
    parser.add_argument("--num_frames", type=int, default=81,
                      help="Number of frames to generate (default: 81)")
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256],
                      help="Video size [height, width] (default: 256 256)")
    parser.add_argument("--num_steps", type=int, default=40,
                      help="Number of diffusion steps (default: 40)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                      help="Classifier-free guidance scale (default: 7.5)")
    parser.add_argument("--flow_shift", type=float, default=7.0,
                      help="Flow matching shift parameter (default: 7.0)")
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed (optional, random if not specified)")

    # Output
    parser.add_argument("--output_path", type=str, default="output.mp4",
                      help="Output video path (default: output.mp4)")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run inference on (default: cuda)")

    return parser


def main():
    """Main inference entry point."""
    # Workaround for cusolver internal error: prefer MAGMA backend where available
    try:
        import torch.backends.cuda
        torch.backends.cuda.preferred_linalg_library("magma")
        _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass
    parser = setup_parser()
    args = parser.parse_args()

    # Create inference wrapper
    logger.info("Initializing LoRA-MoE inference...")
    inference = LoRAMoEInference(
        task=args.task,
        dit_path=args.dit,
        dit_high_noise_path=args.dit_high_noise,
        vae_path=args.vae,
        t5_path=args.t5,
        clip_path=args.clip,
        lora_moe_weights=args.lora_moe_weights,
        alpha_base=args.alpha_base,
        alpha_expert=args.alpha_expert,
        device=args.device,
        lora_moe_config_file=args.lora_moe_config_file,
    )

    # Generate
    video = inference.generate(
        prompt=args.prompt,
        instrument_hint=args.instrument_hint,
        image_path=args.image_path,
        num_frames=args.num_frames,
        video_size=args.video_size,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        flow_shift=args.flow_shift,
        seed=args.seed,
        output_path=args.output_path,
    )

    logger.info(f"✓ Generation complete! Video saved to {args.output_path}")


if __name__ == "__main__":
    main()
