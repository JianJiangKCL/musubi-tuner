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

        # Load DiT model
        logger.info(f"Loading WAN DiT model for task: {task}")
        self.dit = load_wan_model(
            checkpoint_path=dit_path,
            config=self.config,
            device=self.device,
            dtype=torch.bfloat16,
        )

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
            logger.info(f"Loading CLIP from {clip_path}")
            self.clip = CLIPModel(
                dtype=self.config.clip_dtype,
                device=self.device,
                weight_path=clip_path,
            )
        else:
            self.clip = None

        # Create LoRA-MoE network
        self.lora_moe_network = None
        if lora_moe_weights:
            logger.info(f"Loading LoRA-MoE weights from {lora_moe_weights}")
            # Load checkpoint to get config
            checkpoint = load_file(lora_moe_weights)

            # Parse config from checkpoint metadata or use defaults
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

            router_config = {
                "routing_mode": "learned",
                "top_k": 2,
                "temperature": 0.7,
                "learnable_hidden_dim": 64,
            }

            # Target last 8 blocks for WAN2.2 (40-layer model)
            target_block_indices = list(range(32, 40))

            # Create LoRA-MoE network
            self.lora_moe_network = WanLoRAMoENetwork(
                model=self.dit,
                lora_config=lora_config,
                router_config=router_config,
                target_block_indices=target_block_indices,
            )

            # Apply LoRA-MoE to model
            self.lora_moe_network.apply_lora_moe()

            # Load weights
            logger.info("Loading LoRA-MoE state dict")
            self._load_lora_moe_weights(checkpoint)

            # Set alpha scales
            self._set_lora_scales(alpha_base, alpha_expert)
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
            "Scissors": ["剪刀", "剪切"],
            "Hook/Electrocautery": ["电凝钩", "双极电凝", "电凝", "钩"],
            "Suction": ["吸引", "吸"],
            "Other": ["抓钳", "戳卡", "镊子"],
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
            # No match, use Other
            return "Other"

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
        expert_names = ["Scissors", "Hook/Electrocautery", "Suction", "Other"]

        # Determine instrument
        if instrument_hint is not None:
            instrument = instrument_hint
        elif prompt is not None:
            instrument = self.infer_instrument_from_prompt(prompt)
        else:
            # Default to Other
            instrument = "Other"

        # Find expert index
        if instrument in expert_names:
            expert_idx = expert_names.index(instrument)
        else:
            expert_idx = expert_names.index("Other")

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

        # Set routing gates
        if self.lora_moe_network is not None:
            self.lora_moe_network.set_routing_gates(
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

        # Handle image conditioning for i2v
        if "i2v" in self.task and image_path:
            logger.info(f"Loading start image from {image_path}")
            from PIL import Image
            import torchvision.transforms.functional as TF

            image = Image.open(image_path).convert("RGB")
            image = image.resize((width, height))
            image_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device) * 2 - 1

            # Encode image with CLIP
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                clip_embedding = self.clip.encode_image(image_tensor)

            # Add clip embedding to context
            # This is task-specific, adjust based on your model architecture
            context = torch.cat([context, clip_embedding], dim=1)
            context_null = torch.cat([context_null, torch.zeros_like(clip_embedding)], dim=1)

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
            if self.lora_moe_network is not None:
                timestep_normalized = 1.0 - (i / len(timesteps))
                self.lora_moe_network.set_timestep(timestep_normalized)

            # Expand latents for CFG
            latent_model_input = torch.cat([latents, latents], dim=0)

            # Prepare model inputs
            arg_c = {"context": context, "seq_len": seq_len}
            arg_null = {"context": context_null, "seq_len": seq_len}

            # Model prediction with CFG
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                # Conditional prediction
                noise_pred_cond = self.dit(latents, timestep=t.unsqueeze(0), **arg_c)
                # Unconditional prediction
                noise_pred_uncond = self.dit(latents, timestep=t.unsqueeze(0), **arg_null)

            # Classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            if (i + 1) % 10 == 0:
                logger.info(f"Denoising step {i+1}/{len(timesteps)}")

        # Decode latents to video
        logger.info("Decoding latents to video...")
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            video = self.vae.decode(latents)

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

    # Generation config
    parser.add_argument("--prompt", type=str, required=True,
                      help="Text prompt (in Chinese)")
    parser.add_argument("--instrument_hint", type=str, default=None,
                      choices=["Scissors", "Hook/Electrocautery", "Suction", "Other"],
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
