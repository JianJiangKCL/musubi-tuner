# LoRA-MoE for WAN2.2: Surgical Video Adaptation Guide

## Overview

This guide explains how to use the **LoRA-of-Experts (LoRA-MoE)** system for adapting WAN2.2 to surgical video generation with instrument-specific specialization.

### Key Features

- **Mixture of LoRA Experts**: 4 instrument-specific LoRA adapters (Scissors, Hook/Electrocautery, Suction, Other)
- **Base LoRA**: Shared general adaptation across all instruments
- **Intelligent Routing**: Automatic or manual instrument-aware routing
- **Low-Data Efficient**: Designed for 160 video clips with minimal overfitting
- **Timestep-Selective**: Experts activate during shape-critical diffusion timesteps
- **ROI-Focused**: Emphasizes surgical tip regions for better instrument identity

---

## Architecture

```
WAN2.2 DiT Block (Last 8 blocks: 32-39)
├── Self-Attention (Q/K/V/O projections)
│   └── LoRA-MoE: Base LoRA + 4 Expert LoRAs
├── Cross-Attention (Q/K/V/O projections)
│   └── LoRA-MoE: Base LoRA + 4 Expert LoRAs
└── FFN (First linear, last 3 blocks only)
    └── LoRA-MoE: Base LoRA + 4 Expert LoRAs

Composition per projection:
y = W(x) + α_base * ΔW_base(x) + Σ_e g_e * α_e * ΔW_e(x)

Where:
- W: Frozen base weights
- ΔW_base: Shared base LoRA adaptation
- ΔW_e: Expert LoRA for instrument e
- g_e: Routing gate for expert e (sum to 1)
- α: Scaling factors
```

### Projection-Specific Ranks

| Projection | Rank | Rationale |
|------------|------|-----------|
| Query (Q)  | 8    | Most impactful for attention patterns |
| Key (K)    | 4    | Standard rank |
| Value (V)  | 4    | Standard rank |
| Output (O) | 4    | Standard rank |
| FFN        | 4    | Applied only to last 2-3 blocks |

### Expert Families

| Expert ID | Instrument Family | Chinese | Examples |
|-----------|-------------------|---------|----------|
| 0 | Scissors | 剪刀 | Surgical scissors |
| 1 | Hook/Electrocautery | 电凝钩/双极电凝 | Electrocautery hook, bipolar cautery |
| 2 | Suction | 吸引 | Suction devices |
| 3 | Other | 其他 | Graspers (抓钳), pushrods (戳卡), etc. |

---

## Training Pipeline

### Stage A: Base LoRA Training

**Goal**: General adaptation to surgical video domain.

**Configuration**: `configs/lora_moe_stage_a.toml`

**Training**:
```bash
python src/musubi_tuner/wan_train_lora_moe.py \
    --config configs/lora_moe_stage_a.toml \
    --dit models/wan2.2/wan_t2v_A14B.safetensors \
    --dit_high_noise models/wan2.2/wan_t2v_A14B_high_noise.safetensors \
    --vae models/wan2.2/vae.safetensors \
    --t5 models/t5-v1_1-xxl \
    --clip models/clip-vit-large-patch14 \
    --train_data_dir data/surgical_videos \
    --instrument_data_path Lap/preprocessing/filtered_clips_processed.jsonl \
    --output_dir outputs/lora_moe_stage_a
```

**What happens**:
- Base LoRA trains on all videos (160 clips)
- Experts are initialized but frozen
- No routing regularization
- Loss: Base diffusion + ROI-weighted reconstruction
- Output: `lora_moe_stage_a/lora_moe_final.safetensors`

**Duration**: ~10 epochs, 2-3 hours on 1xA100

---

### Stage B: Expert LoRA Training

**Goal**: Instrument-specific specialization.

**Configuration**: `configs/lora_moe_stage_b.toml`

**Prerequisites**:
- Stage A checkpoint
- Instrument labels in dataset (from `filtered_clips_processed.jsonl`)
- Optional: Instrument classifier for identity loss
- Optional: ROI detector for masks

**Training**:
```bash
python src/musubi_tuner/wan_train_lora_moe.py \
    --config configs/lora_moe_stage_b.toml \
    --dit models/wan2.2/wan_t2v_A14B.safetensors \
    --dit_high_noise models/wan2.2/wan_t2v_A14B_high_noise.safetensors \
    --vae models/wan2.2/vae.safetensors \
    --t5 models/t5-v1_1-xxl \
    --clip models/clip-vit-large-patch14 \
    --base_lora_weights outputs/lora_moe_stage_a/lora_moe_final.safetensors \
    --train_data_dir data/surgical_videos \
    --instrument_data_path Lap/preprocessing/filtered_clips_processed.jsonl \
    --output_dir outputs/lora_moe_stage_b
```

**What happens**:
- Base LoRA frozen
- 4 expert LoRAs train independently
- Rule-based routing from instrument labels
- Top-2 routing with EMA smoothing
- Loss: Base + ROI + Identity + Temporal + Routing regularization
- Output: `lora_moe_stage_b/lora_moe_final.safetensors`

**Duration**: ~15 epochs, 3-5 hours on 1xA100

---

### Stage C: Learnable Router (Optional)

**Goal**: Replace rule-based router with learned MLP.

**Configuration**: `configs/lora_moe_stage_c.toml`

**Prerequisites**:
- Stage B checkpoint
- Text embeddings from prompts

**Training**:
```bash
python src/musubi_tuner/wan_train_lora_moe.py \
    --config configs/lora_moe_stage_c.toml \
    --dit models/wan2.2/wan_t2v_A14B.safetensors \
    --dit_high_noise models/wan2.2/wan_t2v_A14B_high_noise.safetensors \
    --vae models/wan2.2/vae.safetensors \
    --t5 models/t5-v1_1-xxl \
    --clip models/clip-vit-large-patch14 \
    --lora_moe_weights outputs/lora_moe_stage_b/lora_moe_final.safetensors \
    --train_data_dir data/surgical_videos \
    --instrument_data_path Lap/preprocessing/filtered_clips_processed.jsonl \
    --output_dir outputs/lora_moe_stage_c
```

**What happens**:
- All LoRAs frozen
- Tiny MLP router (64 hidden dim) trains
- Teacher-student guidance from rule-based router
- KL divergence loss + entropy + load balancing
- Output: `lora_moe_stage_c/lora_moe_final.safetensors`

**Duration**: ~5 epochs, 1 hour on 1xA100

---

## Inference

### Basic Usage

```bash
python src/musubi_tuner/wan_inference_lora_moe.py \
    --task t2v-A14B \
    --dit models/wan2.2/wan_t2v_A14B.safetensors \
    --dit_high_noise models/wan2.2/wan_t2v_A14B_high_noise.safetensors \
    --vae models/wan2.2/vae.safetensors \
    --t5 models/t5-v1_1-xxl \
    --clip models/clip-vit-large-patch14 \
    --lora_moe_weights outputs/lora_moe_stage_b/lora_moe_final.safetensors \
    --prompt "电凝钩切割组织。吸引止血。" \
    --instrument_hint "Hook/Electrocautery" \
    --output_path output_hook.mp4 \
    --alpha_base 0.7 \
    --alpha_expert 1.0
```

### Parameters

- `--alpha_base`: Scaling for base LoRA (0.5-1.0, default: 0.7)
- `--alpha_expert`: Scaling for expert LoRAs (0.8-1.2, default: 1.0)
- `--instrument_hint`: Manual routing override
  - `Scissors`
  - `Hook/Electrocautery`
  - `Suction`
  - `Other`
- If no `--instrument_hint`, auto-detects from Chinese prompt keywords

### Routing Behavior

1. **Auto-routing** (no hint):
   - Extracts instrument keywords from prompt
   - Maps to expert family
   - Falls back to `Other` if ambiguous

2. **Manual routing** (with hint):
   - Forces specific expert
   - Useful for controlled generation

3. **Top-2 soft routing**:
   - Primary expert: 60-80% weight
   - Secondary expert: 20-40% weight
   - Prevents over-specialization

4. **Timestep selectivity**:
   - Early timesteps (t < 0.2): Experts off
   - Mid timesteps (0.2 ≤ t < 0.6): Experts at 50%
   - Late timesteps (t ≥ 0.6): Experts at 100%

---

## Data Preparation

### Required Format

`filtered_clips_processed.jsonl`:
```json
{
  "video_path": "/path/to/clip.mp4",
  "caption": "抓钳抓取组织。电凝钩切割。",
  "duration": 4.5,
  "source": "cholec02",
  "instrument_label": 1  // Optional: 0=Scissors, 1=Hook, 2=Suction, 3=Other
}
```

### Preprocessing Script

Use existing script:
```bash
python Lap/preprocessing/filter_clips_by_duration4all.py \
    --input_dir data/raw_videos \
    --output_jsonl data/filtered_clips_processed.jsonl \
    --min_duration 2.0 \
    --max_duration 20.0
```

### Instrument Label Generation

If labels are missing, you can:

1. **Manual annotation**: Annotate 160 clips manually
2. **Keyword-based**: Extract from captions
   ```python
   def get_instrument_label(caption):
       if "剪刀" in caption or "剪切" in caption:
           return 0  # Scissors
       elif "电凝" in caption or "钩" in caption:
           return 1  # Hook/Electrocautery
       elif "吸引" in caption:
           return 2  # Suction
       else:
           return 3  # Other
   ```
3. **Classifier-based**: Train a simple instrument classifier on frames

---

## Loss Functions

### 1. Base Diffusion Loss
Standard flow-matching MSE loss:
```
L_diffusion = MSE(model_pred, target_noise)
```

### 2. ROI-Weighted Reconstruction
Emphasizes tip regions:
```
L_roi = Σ w(x,y) * MSE(pred, target)
where w(x,y) = 1.0 (background), 3.0-5.0 (ROI)
```

### 3. Identity Preservation
Maintains instrument class identity:
```
L_identity = CE(classifier(generated_frame), instrument_label)
```
- Applied only in ROI regions
- Requires frozen instrument classifier
- Weight: 0.5

### 4. Temporal Consistency
Ensures smooth motion:
```
L_temporal = Σ_t L1(warp(frame_t, flow_t→t+1), frame_t+1)
```
- Optional: uses optical flow
- Applied to ROI patches
- Weight: 0.5

### 5. Routing Regularization

**Entropy penalty** (prevents collapse):
```
L_entropy = max(0, target_entropy - H(gates))
where H(gates) = -Σ g_e log(g_e)
```

**Load balancing** (uniform usage):
```
L_balance = MSE(mean(gates), uniform)
```

**Teacher KL** (for stage C):
```
L_teacher = KL(learned_gates || rule_based_gates)
```

### Combined Loss

```
L_total = L_diffusion + 3.0 * L_roi + 0.5 * L_identity + 0.5 * L_temporal
          + 0.01 * L_entropy + 0.05 * L_balance
```

---

## Hyperparameters

### Recommended Defaults

| Parameter | Stage A | Stage B | Stage C |
|-----------|---------|---------|---------|
| Learning Rate | 1e-4 | 5e-5 | 1e-3 |
| Batch Size | 1 | 1 | 2 |
| Grad Accum | 8 | 8 | 4 |
| Epochs | 10 | 15 | 5 |
| LR Scheduler | constant_warmup | cosine | cosine |
| Rank Dropout | 0.0 | 0.05 | 0.0 |
| Timestep Range | [0, 1] | [0.2, 1.0] | [0, 1] |

### Memory Optimization

For 80GB A100:
- Batch size: 1-2
- Gradient checkpointing: enabled
- Offload inactive DiT: enabled
- Mixed precision: bf16

For 24GB RTX 4090:
- Batch size: 1
- Gradient accumulation: 16
- Block swap: 10 blocks
- Offload inactive DiT: enabled

---

## Evaluation Metrics

### 1. Tip PCK (Percentage of Correct Keypoints)
Measures tip localization accuracy:
```
PCK@threshold = % of predicted tips within threshold pixels of GT
```

### 2. Tip IoU
Intersection over Union for tip bounding boxes:
```
IoU = Area(pred ∩ GT) / Area(pred ∪ GT)
```

### 3. Identity Accuracy
Instrument classification accuracy on generated tips:
```
Accuracy = % of generated tips classified correctly
```

### 4. Temporal Stability
Warped PSNR/LPIPS in ROI:
```
PSNR_temporal = mean_t PSNR(warp(frame_t, flow), frame_t+1)
```

### 5. Video Quality
- FVD (Fréchet Video Distance)
- KVD (Kernel Video Distance)
- Human evaluation (tip realism, motion smoothness)

---

## Ablation Studies

Test these configurations to validate design:

1. **Base vs Base+MoE**
   - Base LoRA only
   - Base + 4 experts

2. **Number of experts**
   - E ∈ {2, 3, 4, 5}

3. **Routing strategy**
   - Top-1 vs Top-2
   - Temperature: {0.5, 0.7, 1.0}

4. **Timestep gating**
   - Always on
   - Timestep-selective (0.2-1.0)

5. **ROI weighting**
   - Uniform (1.0 everywhere)
   - ROI-weighted (3.0x)

6. **Target blocks**
   - Last 4 blocks
   - Last 6 blocks
   - Last 8 blocks

---

## Troubleshooting

### Expert Collapse
**Symptom**: All gates route to one expert

**Solutions**:
- Increase `weight_routing_load_balance` (0.05 → 0.1)
- Increase entropy weight (0.01 → 0.05)
- Cap max gate at 0.9
- Ensure balanced sampling across instruments

### Router Noise
**Symptom**: Gates fluctuate wildly across frames

**Solutions**:
- Increase EMA beta (0.9 → 0.95)
- Use clip-level gates instead of frame-level
- Lower router temperature (0.7 → 0.5)

### Overfitting
**Symptom**: Training loss low, validation loss high

**Solutions**:
- Reduce LoRA ranks (r=4 → r=2)
- Increase rank dropout (0.0 → 0.1)
- Add stronger data augmentation
- Early stop based on validation ROI metrics
- Freeze norm layers

### Low Identity Accuracy
**Symptom**: Generated tips don't match instrument

**Solutions**:
- Increase `weight_identity` (0.5 → 1.0)
- Increase `weight_roi_recon` (3.0 → 5.0)
- Check instrument classifier quality
- Improve ROI masks

---

## File Structure

```
musubi-tuner/
├── src/musubi_tuner/
│   ├── networks/
│   │   ├── lora_moe.py              # Core LoRA-MoE modules
│   │   └── lora_wan_moe.py          # WAN-specific integration
│   ├── losses/
│   │   └── lora_moe_losses.py       # Loss functions
│   ├── wan_train_lora_moe.py        # Training script
│   └── wan_inference_lora_moe.py    # Inference script
├── configs/
│   ├── lora_moe_stage_a.toml        # Stage A config
│   ├── lora_moe_stage_b.toml        # Stage B config
│   └── lora_moe_stage_c.toml        # Stage C config
├── docs/
│   └── lora_moe_guide.md            # This guide
└── Lap/preprocessing/
    └── filter_clips_by_duration4all.py  # Data preprocessing
```

---

## Citation

If you use this LoRA-MoE implementation, please cite:

```bibtex
@misc{wan_lora_moe_2024,
  title={LoRA-MoE for Surgical Video Generation: Instrument-Specific Adaptation with Mixture of Low-Rank Experts},
  author={Your Name},
  year={2024},
  note={Implementation for WAN2.2 video diffusion model}
}
```

---

## Support

For issues and questions:
- GitHub Issues: [musubi-tuner/issues](https://github.com/your-repo/musubi-tuner/issues)
- Documentation: [docs/](docs/)

---

## License

Same license as musubi-tuner base repository.
