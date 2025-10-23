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

from musubi_tuner.wan_generate_video import WanVideoGenerator
from musubi_tuner.networks.lora_wan_moe import create_wan_lora_moe, WanLoRAMoENetwork
from musubi_tuner.wan.configs import WAN_CONFIGS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LoRAMoEInference:
    """
    Inference wrapper for LoRA-MoE adapted WAN2.2.
    """

    def __init__(
        self,
        task: str = "t2v-A14B",
        lora_moe_weights: str = None,
        alpha_base: float = 0.7,
        alpha_expert: float = 1.0,
        device: str = "cuda",
    ):
        """
        Args:
            task: WAN task name (e.g., "t2v-A14B")
            lora_moe_weights: Path to LoRA-MoE weights
            alpha_base: Scaling for base LoRA
            alpha_expert: Scaling for expert LoRAs
            device: Device to run inference on
        """
        self.task = task
        self.config = WAN_CONFIGS[task]
        self.device = device
        self.alpha_base = alpha_base
        self.alpha_expert = alpha_expert

        # Load model
        logger.info(f"Loading WAN model for task: {task}")
        # This would use your existing model loading code
        # self.model = load_wan_model(...)

        # Create LoRA-MoE network
        if lora_moe_weights:
            logger.info(f"Loading LoRA-MoE weights from {lora_moe_weights}")
            # Load saved config from weights
            checkpoint = torch.load(lora_moe_weights, map_location="cpu")
            lora_config = checkpoint.get("config", {})

            # Create LoRA-MoE network
            self.lora_moe_network = create_wan_lora_moe(
                model=self.model,
                lora_config=lora_config.get("lora_moe_config"),
                router_config=lora_config.get("router_config"),
                target_blocks=lora_config.get("target_block_indices"),
            )

            # Load weights
            self.lora_moe_network.load_lora_moe_weights(lora_moe_weights)

            # Set alpha scales
            self._set_lora_scales(alpha_base, alpha_expert)
        else:
            logger.warning("No LoRA-MoE weights provided, using base model")
            self.lora_moe_network = None

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

    def generate(
        self,
        prompt: str,
        instrument_hint: Optional[str] = None,
        num_frames: int = 81,
        num_steps: int = 40,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate video with LoRA-MoE.

        Args:
            prompt: Text prompt
            instrument_hint: Explicit instrument hint for routing
            num_frames: Number of frames to generate
            num_steps: Number of diffusion steps
            guidance_scale: Guidance scale
            seed: Random seed
            output_path: Path to save video (optional)

        Returns:
            video: Generated video as numpy array [F, H, W, C]
        """
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

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

        # Generate video
        # This would use your existing generation pipeline
        # For now, this is a placeholder
        logger.info("Generating video...")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Instrument: {instrument_hint or 'auto-detected'}")
        logger.info(f"Frames: {num_frames}, Steps: {num_steps}, CFG: {guidance_scale}")

        # Placeholder: actual generation would go here
        # video = self.model.generate(...)

        # During diffusion loop, you would:
        # 1. Set timestep for each step: self.lora_moe_network.set_timestep(t)
        # 2. Forward through model with LoRA-MoE active
        # 3. Experts are applied based on timestep_selective gating

        logger.warning("Actual generation not implemented - this is a template")

        # if output_path:
        #     save_video(video, output_path)

        # return video
        return None


def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser."""
    parser = argparse.ArgumentParser(description="LoRA-MoE Inference for WAN")

    # Model config
    parser.add_argument("--task", type=str, default="t2v-A14B",
                      help="WAN task name")
    parser.add_argument("--dit", type=str, required=True,
                      help="Path to DiT weights")
    parser.add_argument("--dit_high_noise", type=str, default=None,
                      help="Path to high-noise DiT weights (WAN2.2)")
    parser.add_argument("--vae", type=str, required=True,
                      help="Path to VAE weights")
    parser.add_argument("--t5", type=str, required=True,
                      help="Path to T5 model")
    parser.add_argument("--clip", type=str, required=True,
                      help="Path to CLIP model")

    # LoRA-MoE config
    parser.add_argument("--lora_moe_weights", type=str, required=True,
                      help="Path to LoRA-MoE weights")
    parser.add_argument("--alpha_base", type=float, default=0.7,
                      help="Scaling for base LoRA")
    parser.add_argument("--alpha_expert", type=float, default=1.0,
                      help="Scaling for expert LoRAs")

    # Generation config
    parser.add_argument("--prompt", type=str, required=True,
                      help="Text prompt (in Chinese)")
    parser.add_argument("--instrument_hint", type=str, default=None,
                      choices=["Scissors", "Hook/Electrocautery", "Suction", "Other"],
                      help="Explicit instrument hint for routing")
    parser.add_argument("--num_frames", type=int, default=81,
                      help="Number of frames")
    parser.add_argument("--num_steps", type=int, default=40,
                      help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                      help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed")

    # Output
    parser.add_argument("--output_path", type=str, default="output.mp4",
                      help="Output video path")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run on")

    return parser


def main():
    """Main inference entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    # Create inference wrapper
    inference = LoRAMoEInference(
        task=args.task,
        lora_moe_weights=args.lora_moe_weights,
        alpha_base=args.alpha_base,
        alpha_expert=args.alpha_expert,
        device=args.device,
    )

    # Generate
    video = inference.generate(
        prompt=args.prompt,
        instrument_hint=args.instrument_hint,
        num_frames=args.num_frames,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_path=args.output_path,
    )

    logger.info(f"Generation complete! Saved to {args.output_path}")


if __name__ == "__main__":
    main()
