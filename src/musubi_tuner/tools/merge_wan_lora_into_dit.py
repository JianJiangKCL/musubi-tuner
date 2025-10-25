import argparse
import logging
import torch
from safetensors.torch import load_file

from musubi_tuner.networks import lora
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file
from musubi_tuner.wan.configs import WAN_CONFIGS
from musubi_tuner.wan.modules.model import load_wan_model


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge Stage-1 LoRA into WAN DiT checkpoint")
    parser.add_argument("--task", type=str, default="i2v-A14B", choices=list(WAN_CONFIGS.keys()), help="WAN task key")
    parser.add_argument("--dit", type=str, required=True, help="Path to DiT checkpoint (first shard or single file)")
    parser.add_argument("--lora_weight", type=str, required=True, help="Stage-1 LoRA weights (.safetensors)")
    parser.add_argument(
        "--lora_multiplier", type=float, default=1.0, help="Multiplier when merging LoRA into base DiT"
    )
    parser.add_argument("--save_merged_model", type=str, required=True, help="Output path for merged DiT safetensors")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu")
    logger.info(f"Loading WAN DiT for task={args.task} from: {args.dit}")
    config = WAN_CONFIGS[args.task]
    transformer = load_wan_model(
        config=config,
        device=device,
        dit_path=args.dit,
        attn_mode="torch",
        split_attn=False,
        loading_device="cpu",
        dit_weight_dtype=torch.bfloat16,
        fp8_scaled=False,
    )
    transformer.eval()

    logger.info(f"Loading LoRA weights: {args.lora_weight} (multiplier={args.lora_multiplier})")
    weights_sd = load_file(args.lora_weight)
    network = lora.create_arch_network_from_weights(args.lora_multiplier, weights_sd, unet=transformer, for_inference=True)

    logger.info("Merging LoRA weights into DiT")
    network.merge_to(None, transformer, weights_sd, dtype=torch.bfloat16, device=device, non_blocking=True)

    logger.info(f"Saving merged model to: {args.save_merged_model}")
    mem_eff_save_file(transformer.state_dict(), args.save_merged_model)
    logger.info("Merged model saved")


if __name__ == "__main__":
    main()


