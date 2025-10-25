# LoRA-MoE 推理指南 / LoRA-MoE Inference Guide

## 概述 / Overview

本指南介绍如何使用训练好的 WAN2.2 LoRA-MoE 模型进行手术器械特定的视频生成。

This guide explains how to use trained WAN2.2 LoRA-MoE models for instrument-specific surgical video generation.

## 快速开始 / Quick Start

### 1. 准备模型权重 / Prepare Model Weights

确保已完成 Stage 2 训练,并有以下文件:

Ensure Stage 2 training is complete and you have:

```bash
# LoRA-MoE weights from Stage 2
outputs/lora_moe_stage2/lora_moe_final.safetensors

# Base WAN2.2 model weights
outputs/merged_low_noise.safetensors
/path/to/Wan2.1_VAE.pth
/path/to/models_t5_umt5-xxl-enc-bf16.pth
/path/to/clip-vit-large-patch14
```

### 2. 配置推理脚本 / Configure Inference Script

编辑 `Lap/scripts/infer_lora_moe.sh`:

Edit `Lap/scripts/infer_lora_moe.sh`:

```bash
# 修改路径 / Modify paths
PROJECT_ROOT="/your/path/to/musubi-tuner"
CKPT_ROOT="/your/path/to/Wan2.2-I2V-A14B"

# 设置提示词 / Set prompt
PROMPT="抓钳抓取组织。电凝钩切割血管。"

# 选择器械类型 / Choose instrument
INSTRUMENT_HINT="Hook/Electrocautery"  # Options: Scissors, Hook/Electrocautery, Suction, Other

# 输入图像 (i2v 任务需要) / Input image (required for i2v)
IMAGE_PATH="/path/to/start_frame.png"
```

### 3. 运行推理 / Run Inference

```bash
cd /path/to/musubi-tuner
bash Lap/scripts/infer_lora_moe.sh
```

## Python API 使用 / Python API Usage

也可以直接使用 Python API:

You can also use the Python API directly:

```python
from musubi_tuner.wan_inference_lora_moe import LoRAMoEInference

# Initialize
inference = LoRAMoEInference(
    task="i2v-A14B",
    dit_path="outputs/merged_low_noise.safetensors",
    vae_path="/path/to/Wan2.1_VAE.pth",
    t5_path="/path/to/models_t5_umt5-xxl-enc-bf16.pth",
    clip_path="/path/to/clip-vit-large-patch14",
    lora_moe_weights="outputs/lora_moe_stage2/lora_moe_final.safetensors",
    alpha_base=0.7,
    alpha_expert=1.0,
    device="cuda",
)

# Generate video
video = inference.generate(
    prompt="抓钳抓取组织。电凝钩切割血管。",
    instrument_hint="Hook/Electrocautery",
    image_path="/path/to/start_frame.png",
    num_frames=81,
    video_size=(256, 256),
    num_steps=40,
    guidance_scale=7.5,
    flow_shift=7.0,
    seed=42,
    output_path="output.mp4",
)
```

## 命令行使用 / Command Line Usage

```bash
python -m musubi_tuner.wan_inference_lora_moe \
    --task i2v-A14B \
    --dit outputs/merged_low_noise.safetensors \
    --vae /path/to/Wan2.1_VAE.pth \
    --t5 /path/to/models_t5_umt5-xxl-enc-bf16.pth \
    --clip /path/to/clip-vit-large-patch14 \
    --lora_moe_weights outputs/lora_moe_stage2/lora_moe_final.safetensors \
    --prompt "抓钳抓取组织。电凝钩切割。" \
    --instrument_hint "Hook/Electrocautery" \
    --image_path start_frame.png \
    --num_frames 81 \
    --video_size 256 256 \
    --num_steps 40 \
    --guidance_scale 7.5 \
    --flow_shift 7.0 \
    --alpha_base 0.7 \
    --alpha_expert 1.0 \
    --seed 42 \
    --output_path output_video.mp4
```

## 参数说明 / Parameter Description

### 器械类型 / Instrument Types

LoRA-MoE 支持 4 种器械专家:

LoRA-MoE supports 4 instrument experts:

- **Scissors** (剪刀): 剪切、分离组织
- **Hook/Electrocautery** (电凝钩/双极电凝): 电凝止血、切割
- **Suction** (吸引): 吸引血液、烟雾
- **Other** (其他): 抓钳、戳卡等其他器械

### 关键参数 / Key Parameters

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--instrument_hint` | None | 显式指定器械类型,不指定则从提示词自动推断 |
| `--alpha_base` | 0.7 | Base LoRA 强度,控制整体风格适配 |
| `--alpha_expert` | 1.0 | Expert LoRA 强度,控制器械特定特征 |
| `--num_steps` | 40 | 扩散步数,越多质量越高但速度越慢 |
| `--guidance_scale` | 7.5 | CFG 强度,控制提示词遵循程度 |
| `--flow_shift` | 7.0 | Flow matching 参数,影响生成质量 |
| `--seed` | random | 随机种子,固定可复现生成结果 |

## 调优建议 / Tuning Tips

### 1. 器械路由 / Instrument Routing

- **显式指定**: 使用 `--instrument_hint` 精确控制专家选择
- **自动检测**: 留空让模型从提示词推断器械类型
- **多器械场景**: 提示词包含多种器械时,模型会自动混合多个专家

### 2. LoRA 强度调节 / LoRA Strength Adjustment

```bash
# 强化器械特定特征 / Enhance instrument-specific features
--alpha_base 0.5 --alpha_expert 1.2

# 平衡风格和特征 / Balance style and features
--alpha_base 0.7 --alpha_expert 1.0

# 减弱适配效果 / Reduce adaptation effect
--alpha_base 0.3 --alpha_expert 0.5
```

### 3. 生成质量 / Generation Quality

```bash
# 高质量 (慢) / High quality (slow)
--num_steps 50 --guidance_scale 8.0

# 平衡质量与速度 / Balanced
--num_steps 40 --guidance_scale 7.5

# 快速预览 / Fast preview
--num_steps 20 --guidance_scale 6.0
```

## 示例 / Examples

### 示例 1: 剪刀剪切 / Example 1: Scissors Cutting

```bash
python -m musubi_tuner.wan_inference_lora_moe \
    --task i2v-A14B \
    --dit outputs/merged_low_noise.safetensors \
    --vae /path/to/Wan2.1_VAE.pth \
    --t5 /path/to/models_t5_umt5-xxl-enc-bf16.pth \
    --clip /path/to/clip-vit-large-patch14 \
    --lora_moe_weights outputs/lora_moe_stage2/lora_moe_final.safetensors \
    --prompt "剪刀剪切缝线" \
    --instrument_hint "Scissors" \
    --image_path start.png \
    --output_path scissors_cutting.mp4
```

### 示例 2: 电凝钩止血 / Example 2: Hook Cauterization

```bash
python -m musubi_tuner.wan_inference_lora_moe \
    --task i2v-A14B \
    --dit outputs/merged_low_noise.safetensors \
    --vae /path/to/Wan2.1_VAE.pth \
    --t5 /path/to/models_t5_umt5-xxl-enc-bf16.pth \
    --clip /path/to/clip-vit-large-patch14 \
    --lora_moe_weights outputs/lora_moe_stage2/lora_moe_final.safetensors \
    --prompt "电凝钩电凝止血" \
    --instrument_hint "Hook/Electrocautery" \
    --image_path start.png \
    --output_path hook_cautery.mp4
```

### 示例 3: 多器械场景 / Example 3: Multi-instrument Scene

```bash
python -m musubi_tuner.wan_inference_lora_moe \
    --task i2v-A14B \
    --dit outputs/merged_low_noise.safetensors \
    --vae /path/to/Wan2.1_VAE.pth \
    --t5 /path/to/models_t5_umt5-xxl-enc-bf16.pth \
    --clip /path/to/clip-vit-large-patch14 \
    --lora_moe_weights outputs/lora_moe_stage2/lora_moe_final.safetensors \
    --prompt "抓钳抓取组织,剪刀剪切,吸引清理视野" \
    --output_path multi_instrument.mp4
    # No instrument_hint: auto-detect and mix experts
```

## 故障排除 / Troubleshooting

### 问题 1: CUDA OOM

```bash
# 减少帧数 / Reduce frames
--num_frames 41

# 降低分辨率 / Reduce resolution
--video_size 128 128

# 减少步数 / Reduce steps
--num_steps 30
```

### 问题 2: 生成质量差

```bash
# 增加步数 / Increase steps
--num_steps 50

# 调整 CFG / Adjust CFG
--guidance_scale 8.0

# 调整 LoRA 强度 / Adjust LoRA strength
--alpha_expert 1.2
```

### 问题 3: 器械特征不明显

```bash
# 显式指定器械 / Explicitly specify instrument
--instrument_hint "Hook/Electrocautery"

# 增强专家 LoRA / Strengthen expert LoRA
--alpha_expert 1.5

# 在提示词中强调器械 / Emphasize instrument in prompt
--prompt "电凝钩电凝止血,电凝钩切割组织"
```

## 性能参考 / Performance Reference

| 配置 | VRAM | 时间 (81帧) |
|------|------|-------------|
| 256x256, 40 steps | ~16GB | ~3-5 min |
| 256x256, 30 steps | ~14GB | ~2-3 min |
| 512x512, 40 steps | ~24GB | ~8-12 min |

*测试环境: NVIDIA A100 40GB*

## 相关文档 / Related Documentation

- [Stage 2 Training Guide](./train_lora_moe_stage2.sh) - LoRA-MoE 训练脚本
- [WAN 2.2 Documentation](../../docs/WAN22_TRAINING.md) - WAN 2.2 训练文档
- [LoRA-MoE Architecture](../../src/musubi_tuner/networks/lora_moe.py) - 模型架构实现

## 引用 / Citation

如果使用本代码,请引用:

If you use this code, please cite:

```bibtex
@software{wan22_lora_moe,
  title={WAN2.2 LoRA-MoE for Surgical Video Generation},
  author={Your Name},
  year={2025},
  url={https://github.com/JianJiangKCL/musubi-tuner}
}
```
