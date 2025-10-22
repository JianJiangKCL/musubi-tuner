import argparse
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description="Run two I2V generations with a given LoRA weight")
    parser.add_argument("--lora_weight", type=str, required=True, help="Path to LoRA weight (.safetensors)")
    args = parser.parse_args()

    script_path = "/mnt/cfs/jj/musubi-tuner/src/musubi_tuner/wan_generate_video.py"

    # Fixed config paths and parameters (mirrors scripts/test.sh)
    cmd_base = [
        sys.executable,
        script_path,
        "--dit",
        "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
        "--dit_high_noise",
        "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
        "--lazy_loading",
        "--t5",
        "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
        "--task",
        "i2v-A14B",
        "--prompt",
        "A hand peels a realistic 3D sticker of a car off a flat surface，revealing the flat background underneath and emphasizing the illusion of depth.",
        "--video_size",
        "320",
        "480",
        "--video_length",
        "81",
        "--vae",
        "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/Wan2.1_VAE.pth",
        "--vae_cache_cpu",
        "--save_path",
        "/mnt/cfs/jj/musubi-tuner/test_i2v_outputs/",
        "--attn_mode",
        "flash2",
        "--lora_weight_high_noise",
        args.lora_weight,
        "--lora_weight",
        args.lora_weight,
    ]

    # Two inputs to process in a single model loading via batch mode
    # Each image gets its own appropriate prompt
    prompts_and_images = [
        (
            "A hand peels a realistic 3D sticker of a car off a flat surface，revealing the flat background underneath and emphasizing the illusion of depth.",
            "/mnt/cfs/jj/musubi-tuner/test_i2v_outputs/test_car.jpg"
        ),
        (
            "A hand peels a realistic 3D sticker of a car off a flat surface，revealing the flat background underneath and emphasizing the illusion of depth.",
            "/mnt/cfs/jj/musubi-tuner/datasets/peel_it/first_frames/1.jpg"
        ),
    ]

    # Build a temporary prompts file with one line per input image
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as tf:
        for prompt, img in prompts_and_images:
            # Use short option --i to set per-line image path (parsed by parse_prompt_line)
            tf.write(f"{prompt} --i {img}\n")
        prompts_file = tf.name

    # Single invocation using batch mode; model is loaded once inside wan_generate_video
    cmd = cmd_base + ["--from_file", prompts_file]
    print(f"\n[Batch Run] Executing: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


