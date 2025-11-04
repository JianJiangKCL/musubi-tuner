#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


DEFAULT_DIT_LOW = \
    "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors"
DEFAULT_DIT_HIGH = \
    "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA into both low/high-noise WAN DiT checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "lora_path",
        type=str,
        help="Path to Stage-1 LoRA weights (.safetensors)",
    )
    parser.add_argument(
        "--dit_low",
        type=str,
        default=DEFAULT_DIT_LOW,
        help="Low-noise DiT checkpoint (first shard or single file)",
    )
    parser.add_argument(
        "--dit_high",
        type=str,
        default=DEFAULT_DIT_HIGH,
        help="High-noise DiT checkpoint (first shard or single file)",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="LoRA multiplier for both merges",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/mnt/cfs/jj/proj/musubi-tuner/outputs",
        help="Directory to write merged checkpoints",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="v2",
        help="Suffix tag for output filenames",
    )
    return parser.parse_args()


def run_merge(dit_path: str, lora_path: str, out_path: str, multiplier: float) -> None:
    script_path = os.path.join(os.path.dirname(__file__), "merge_wan_lora_into_dit.py")
    cmd = [
        sys.executable,
        script_path,
        "--task",
        "i2v-A14B",
        "--dit",
        dit_path,
        "--lora_weight",
        lora_path,
        "--lora_multiplier",
        str(multiplier),
        "--save_merged_model",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    low_out = os.path.join(args.out_dir, f"merged_low_noise_{args.tag}.safetensors")
    high_out = os.path.join(args.out_dir, f"merged_high_noise_{args.tag}.safetensors")

    run_merge(args.dit_low, args.lora_path, low_out, args.multiplier)
    run_merge(args.dit_high, args.lora_path, high_out, args.multiplier)


if __name__ == "__main__":
    main()


