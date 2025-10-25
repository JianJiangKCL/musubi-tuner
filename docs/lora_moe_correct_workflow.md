# LoRA-MoE Correct 2-Stage Workflow for WAN2.2

## Overview

This document describes the **CORRECT** 2-stage training pipeline for LoRA-MoE adaptation of WAN2.2 for surgical video generation.

**Key principle**: Stage 1 vanilla LoRA is **merged into DiT weights** before Stage 2, NOT kept as a separate "base_lora" component.

---

## Architecture After Each Stage

### After Stage 1: Merged DiT
```
DiT_merged = DiT_pretrained + α × vanilla_LoRA
```

### After Stage 2: DiT + Expert LoRAs
```
Output = DiT_merged(x) + Σ [gate_e × α_expert × ΔW_e(x)]
       = [DiT_pretrained(x) + α × vanilla_LoRA(x)] + expert_mixture(x)
```

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     STAGE 1: Vanilla LoRA                   │
├─────────────────────────────────────────────────────────────┤
│  Goal: Learn basic surgical video domain knowledge         │
│  Method: Standard LoRA training on WAN2.2                  │
│  Duration: ~10 epochs, 2-3 hours on A100                   │
│                                                             │
│  What's trained:                                            │
│    ✓ LoRA on last 8 DiT blocks (32-39)                     │
│    ✓ Q/K/V/O projections in attention (ranks: 8/4/4/4)    │
│    ✓ FFN first linear in last 3 blocks (rank: 4)          │
│                                                             │
│  Output: vanilla_lora.safetensors                          │
│                                                             │
│  Training command:                                          │
│    python -m musubi_tuner.wan_train_network \              │
│      --task i2v-A14B \                                      │
│      --network_module networks.lora_wan \                  │
│      --network_dim 4 --network_alpha 1.0 \                 │
│      --rank_q 8 --rank_kvo 4 \                             │
│      ...                                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              MERGE STEP (Critical!)                         │
├─────────────────────────────────────────────────────────────┤
│  Merge vanilla LoRA weights into BOTH DiT checkpoints      │
│                                                             │
│  Command:                                                   │
│    python -m musubi_tuner.tools.merge_wan_lora_into_dit \  │
│      --task i2v-A14B \                                      │
│      --dit wan_t2v_A14B.safetensors \                       │
│      --lora_weight vanilla_lora.safetensors \               │
│      --lora_multiplier 1.0 \                                │
│      --save_merged_model merged_low_noise.safetensors       │
│                                                             │
│    python -m musubi_tuner.tools.merge_wan_lora_into_dit \  │
│      --task i2v-A14B \                                      │
│      --dit wan_t2v_A14B_high_noise.safetensors \            │
│      --lora_weight vanilla_lora.safetensors \               │
│      --lora_multiplier 1.0 \                                │
│      --save_merged_model merged_high_noise.safetensors      │
│                                                             │
│  Output:                                                    │
│    ✓ merged_low_noise.safetensors                          │
│    ✓ merged_high_noise.safetensors                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│         STAGE 2: Expert LoRAs + Learned Router              │
├─────────────────────────────────────────────────────────────┤
│  Goal: Instrument-specific specialization + routing        │
│  Method: Train experts & router from merged DiT            │
│  Duration: ~20 epochs, 5-7 hours on A100                   │
│                                                             │
│  What's trained:                                            │
│    ✓ 4 Expert LoRAs (Scissors, Hook, Suction, Other)       │
│    ✓ Learnable MLP router (64 hidden dim)                  │
│    ✓ Timestep-selective gates                              │
│    ✗ NO base_lora (vanilla LoRA already in DiT!)           │
│                                                             │
│  Training features:                                         │
│    • Rule-based teacher guidance for router                │
│    • ROI-weighted loss (3-5× on instrument tips)           │
│    • Routing regularization (entropy + load balance)       │
│    • Top-2 soft routing with EMA smoothing                 │
│                                                             │
│  Training command:                                          │
│    python -m musubi_tuner.wan_train_lora_moe \             │
│      --task i2v-A14B \                                      │
│      --training_stage stage_b \                            │
│      --dit merged_low_noise.safetensors \                  │
│      --dit_high_noise merged_high_noise.safetensors \      │
│      --no_use_base_lora \                                  │
│      --lora_dim 4 --lora_alpha 1.0 \                       │
│      --num_experts 4 \                                      │
│      --routing_mode learned \                              │
│      --train_router --use_teacher_guidance \               │
│      ...                                                    │
│                                                             │
│  Output: lora_moe_stage2.safetensors                       │
│    Contains:                                                │
│      - 4 × expert_lora_{down,up} weights                   │
│      - router MLP weights                                  │
│      - timestep gates                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Why This Workflow is Correct

### ❌ Wrong Approach (Old Documentation)
```
Stage 1 → vanilla_lora.safetensors
           ↓ (load as base_lora in LoRA-MoE)
Stage 2 → Train experts + router with frozen base_lora
```

**Problem**: This requires converting vanilla LoRA weights to LoRA-MoE format, which is complex and error-prone.

### ✅ Correct Approach (This Document)
```
Stage 1 → vanilla_lora.safetensors
           ↓ (merge into DiT weights)
         merged_dit.safetensors
           ↓ (use as frozen base)
Stage 2 → Train ONLY experts + router (no base_lora)
```

**Advantages**:
1. **Simpler**: No weight format conversion needed
2. **Standard practice**: Same as SD/SDXL LoRA fine-tuning
3. **Flexible**: Can adjust vanilla LoRA strength via `--lora_multiplier` during merge
4. **Clean separation**: Stage 1 knowledge is permanently in DiT, Stage 2 only adds specialization

---

## Forward Pass Details

### Stage 1 Inference (Vanilla LoRA)
```python
output = DiT(x) + α × vanilla_LoRA(x)
```

### Stage 2 Training (LoRA-MoE with use_base_lora=False)
```python
# Frozen base (includes merged vanilla LoRA)
base_out = DiT_merged(x)  # = DiT(x) + α × vanilla_LoRA(x)

# Only train expert LoRAs
expert_out = 0.0
for i, expert in enumerate(expert_loras):
    gate = routing_gates[i]
    expert_out += gate * α_expert * expert(x)

# Apply timestep weighting (stronger at late diffusion steps)
expert_out *= timestep_weight(t)

output = base_out + expert_out
```

**Key**: `base_lora` is NOT used when `--no_use_base_lora` is set.

---

## Parameter Configuration

### Stage 1: Vanilla LoRA
```bash
--network_module networks.lora_wan
--network_dim 4
--network_alpha 1.0
--rank_q 8       # Query gets higher rank (most important)
--rank_kvo 4     # K/V/O get lower rank
```

### Stage 2: Expert LoRAs
```bash
--no_use_base_lora   # CRITICAL: Do NOT create base_lora component
--lora_dim 4
--lora_alpha 1.0
--num_experts 4
--rank_q 8
--rank_k 4
--rank_v 4
--rank_o 4
--rank_ffn 4
```

**Important**: Ranks must match between Stage 1 and Stage 2 for consistency.

---

## Inference

### With Vanilla LoRA Only (Stage 1)
```bash
python -m musubi_tuner.wan_generate_video \
  --dit wan_t2v_A14B.safetensors \
  --lora_weight vanilla_lora.safetensors \
  --lora_multiplier 1.0 \
  ...
```

### With LoRA-MoE (Stage 2)
```bash
python -m musubi_tuner.wan_inference_lora_moe \
  --dit merged_low_noise.safetensors \
  --dit_high_noise merged_high_noise.safetensors \
  --lora_moe_weights lora_moe_stage2.safetensors \
  --instrument_hint "scissors" \
  --alpha_base 0.0 \      # No base_lora in this workflow
  --alpha_expert 1.0 \
  ...
```

**Note**: `--alpha_base` is ignored when `use_base_lora=False` during training.

---

## File Structure

```
outputs/
├── vanilla_lora_stage1/
│   └── vanilla_lora.safetensors          # Stage 1 output
│
├── merged_dits/
│   ├── merged_low_noise.safetensors      # Merged DiT (low noise)
│   └── merged_high_noise.safetensors     # Merged DiT (high noise)
│
└── lora_moe_stage2/
    ├── lora_moe_stage2.safetensors       # Final LoRA-MoE weights
    ├── checkpoint-epoch-10.safetensors   # Intermediate checkpoints
    └── logs/
        └── events.out.tfevents.*         # TensorBoard logs
```

---

## Common Issues

### Issue 1: "RuntimeError: Expected tensor for argument #1 'input' to have one of the following scalar types: Half, BFloat16"
**Cause**: Trying to load vanilla LoRA weights directly into LoRA-MoE structure.
**Solution**: Use `--no_use_base_lora` and merged DiT instead.

### Issue 2: "KeyError: 'lora_moe_0' not found in checkpoint"
**Cause**: Trying to load vanilla LoRA checkpoint as LoRA-MoE weights.
**Solution**: Use merged DiT + train from scratch for Stage 2.

### Issue 3: Poor Stage 2 performance
**Possible causes**:
1. Vanilla LoRA (Stage 1) was undertrained → Retrain Stage 1
2. Wrong `--lora_multiplier` during merge → Try 0.8-1.2 range
3. Learning rate too high in Stage 2 → Use 5e-5 (lower than Stage 1)

---

## Summary

| Aspect | Stage 1 | Merge Step | Stage 2 |
|--------|---------|------------|---------|
| **Input** | Pretrained DiT | DiT + vanilla LoRA | Merged DiT |
| **Trains** | vanilla LoRA | N/A (offline merge) | Expert LoRAs + router |
| **Output** | `.safetensors` (LoRA) | `.safetensors` (DiT) | `.safetensors` (LoRA-MoE) |
| **use_base_lora** | N/A | N/A | **False** |
| **base_lora_weights** | N/A | N/A | **Not used** |

**Key Takeaway**: The "base" knowledge from Stage 1 lives in the **merged DiT weights**, not in a separate `base_lora` component within the LoRA-MoE structure.
