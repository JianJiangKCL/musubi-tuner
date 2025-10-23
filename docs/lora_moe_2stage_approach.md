# LoRA-MoE 2-Stage Training Approach

## Overview

This document explains the **recommended 2-stage training pipeline** for LoRA-MoE adaptation of WAN2.2 for surgical video generation.

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     STAGE 1: Vanilla LoRA                   │
│                    (Already done by user)                   │
├─────────────────────────────────────────────────────────────┤
│  Goal: Learn basic surgical video knowledge                │
│  Method: Standard LoRA training on WAN2.2                  │
│  Duration: ~10 epochs, 2-3 hours                           │
│  Output: vanilla_lora/lora_final.safetensors               │
│                                                             │
│  What's trained:                                            │
│    ✓ Base LoRA on last 8 DiT blocks                        │
│    ✓ Q/K/V/O projections in attention                      │
│    ✓ FFN first linear in last 3 blocks                     │
│    ✓ Ranks: Q=8, K/V/O=4, FFN=4                            │
│                                                             │
│  Training command:                                          │
│    python wan_train_network.py --task t2v-A14B ...         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│         STAGE 2: Expert LoRAs + Router (Merged)             │
│                  (This is the MoE stage)                    │
├─────────────────────────────────────────────────────────────┤
│  Goal: Instrument-specific specialization + routing        │
│  Method: Train experts & router simultaneously             │
│  Duration: ~20 epochs, 5-7 hours                           │
│  Output: lora_moe_stage_b/lora_moe_final.safetensors       │
│                                                             │
│  What's trained:                                            │
│    ✓ 4 Expert LoRAs (Scissors, Hook, Suction, Other)       │
│    ✓ Learnable MLP router (64 hidden dim)                  │
│    ✓ Timestep-selective gates                              │
│    ✗ Base LoRA FROZEN (from Stage 1)                       │
│                                                             │
│  Training features:                                         │
│    • Rule-based teacher guidance for router                │
│    • ROI-weighted loss (3-5× on tips)                      │
│    • Routing regularization (entropy + load balance)       │
│    • Top-2 soft routing with EMA smoothing                 │
│                                                             │
│  Training command:                                          │
│    python wan_train_lora_moe.py --task t2v-A14B            │
│      --training_stage stage_b                              │
│      --base_lora_weights vanilla_lora/lora_final.safetensors│
│      --train_router --use_teacher_guidance ...             │
└─────────────────────────────────────────────────────────────┘
```

---

## Why 2 Stages?

### Stage 1: Vanilla LoRA (General Knowledge)

**Purpose**: Learn basic surgical video distribution

- Adapts WAN2.2 to surgical video domain
- Learns common patterns across all instruments
- Provides stable initialization for Stage 2
- **Already completed by user** ✓

**Analogy**: Learning general surgical skills before specializing

### Stage 2: Experts + Router (Specialization)

**Purpose**: Instrument-specific adaptation with intelligent routing

- **Expert LoRAs**: 4 specialists for different instrument families
- **Router**: Learns when to activate which expert
- **Frozen Base**: Preserves general knowledge from Stage 1
- **Co-training**: Experts and router adapt together

**Analogy**: Surgical residents specializing in different techniques

---

## Why Merge Stage B + C?

### Original 3-Stage Approach (Not Recommended)

```
Stage A: Base LoRA only
  ↓
Stage B: Expert LoRAs only (fixed rule-based routing)
  ↓
Stage C: Router only (fixed experts)
```

**Problems**:
- Router trained on frozen experts → may not match actual expert capabilities
- Experts trained without knowing how they'll be routed
- Longer total training time (3 separate stages)
- Router may overfit to rule-based teacher

### New 2-Stage Approach (Recommended)

```
Stage 1: Vanilla LoRA (user already did this)
  ↓
Stage 2: Expert LoRAs + Router simultaneously
```

**Advantages**:
- ✅ Experts and router **co-adapt** during training
- ✅ Router learns based on actual expert specializations
- ✅ **Faster**: One stage instead of two (B + C merged)
- ✅ **More stable**: Teacher guidance prevents routing collapse
- ✅ **Better performance**: Joint optimization

---

## Architecture Details

### Stage 1 Output (Vanilla LoRA)

```
WAN2.2 Frozen Base
    ↓
+ Base LoRA (Trainable) → Learns general surgical knowledge
    ↓
= Adapted Model for Surgical Videos
```

**Parameters**: ~8-12M trainable
**Used as**: Frozen base in Stage 2

### Stage 2 Architecture (LoRA-MoE)

```
WAN2.2 Frozen Base
    ↓
+ Base LoRA (FROZEN from Stage 1) → General knowledge
    ↓
+ Expert LoRAs (Trainable) ─┬─ Expert 0: Scissors
                            ├─ Expert 1: Hook/Electrocautery
                            ├─ Expert 2: Suction
                            └─ Expert 3: Other
    ↓
Gated by Learnable Router (Trainable)
    ↓
= Instrument-Specific Adapted Model
```

**Composition per projection**:
```
y = W(x) + α_base·ΔW_base(x) + Σ_e g_e·α_e·ΔW_e(x)
           ^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^
           Frozen (Stage 1)    Trained (Stage 2)
```

**Parameters**: ~15-20M trainable (experts + router)

---

## Training Configuration

### Stage 1: Vanilla LoRA (User's Config)

```toml
[training]
num_train_epochs = 10
learning_rate = 1e-4
batch_size = 1
gradient_accumulation_steps = 8

[lora]
lora_dim = 4
lora_alpha = 1.0
target_blocks = "last_8"
rank_q = 8
rank_k = 4
rank_v = 4
rank_o = 4
rank_ffn = 4
```

**Output**: `vanilla_lora/lora_final.safetensors`

### Stage 2: Experts + Router (Merged)

```toml
[training]
num_train_epochs = 20  # Longer for joint training
learning_rate = 5e-5   # Lower than Stage 1
batch_size = 1
gradient_accumulation_steps = 8

[lora_moe]
# Same ranks as Stage 1
lora_dim = 4
num_experts = 4
use_base_lora = true  # Load from Stage 1

[router]
routing_mode = "learned"
train_router = true  # ENABLE simultaneous training
use_teacher_guidance = true  # Rule-based teacher
teacher_kl_weight = 1.0

[loss]
weight_routing_entropy = 0.02
weight_routing_load_balance = 0.1
weight_roi_recon = 3.0
```

**Input**: `vanilla_lora/lora_final.safetensors` (frozen as base)
**Output**: `lora_moe_stage_b/lora_moe_final.safetensors`

---

## Loss Function (Stage 2)

### Combined Loss

```
L_total = L_diffusion             # Standard flow matching
        + 3.0 * L_roi              # ROI-weighted reconstruction
        + 0.5 * L_identity         # Instrument identity (optional)
        + 0.5 * L_temporal         # Temporal consistency (optional)
        + 0.02 * L_entropy         # Routing entropy (prevent collapse)
        + 0.1 * L_balance          # Load balancing (uniform expert usage)
        + 1.0 * L_teacher_kl       # KL from learned to rule-based router
```

### Teacher-Student Guidance

**Teacher**: Rule-based router (softmax over instrument classifier logits)
**Student**: Learned MLP router
**Loss**: KL divergence `KL(learned || rule_based)`

**Why?**
- Prevents routing collapse (all gates → one expert)
- Provides stable initialization for learned router
- Allows learned router to deviate when beneficial
- Soft guidance (not hard constraint)

---

## Training Process (Stage 2)

### Epoch-by-Epoch

**Epochs 1-5**: Bootstrap phase
- Experts learn basic specializations
- Router closely follows rule-based teacher
- High KL guidance weight

**Epochs 6-15**: Specialization phase
- Experts diverge for instrument-specific features
- Router learns to activate appropriate experts
- Routing entropy decreases naturally

**Epochs 16-20**: Refinement phase
- Fine-tune expert boundaries
- Stabilize routing decisions
- ROI metrics converge

### Monitoring

Track these metrics:
- **Diffusion loss**: Should decrease steadily
- **ROI loss**: Should decrease faster than diffusion loss
- **Routing entropy**: Should stabilize (not collapse to 0)
- **Expert usage**: Should be roughly uniform (20-30% each)
- **Teacher KL**: Should decrease as router learns

### Early Stopping

Stop if:
- ROI loss plateaus for 5 epochs
- Routing entropy drops below 0.5 (collapse warning)
- Expert usage becomes imbalanced (>70% on one expert)

---

## Inference

### Composition

```python
# Load LoRA-MoE checkpoint
checkpoint = torch.load("lora_moe_final.safetensors")

# Contains:
# - base_lora_* : Frozen vanilla LoRA from Stage 1
# - expert_lora_* : 4 expert LoRAs
# - router_* : Learned MLP router

# Forward pass
y = W(x) + α_base·base_lora(x) + Σ_e gate_e·α_e·expert_lora_e(x)
    ^^^^^   ^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Frozen  Frozen (Stage 1)     Learned (Stage 2)
```

### Routing

```python
# Get instrument logits (from classifier or prompt)
instrument_logits = classifier(frame)  # [B, num_classes]

# Compute gates via learned router
gates = router_mlp(instrument_logits)  # [B, 4]
gates = softmax(gates / temperature)
gates = top_k(gates, k=2)  # Keep top-2

# Example: "Hook" → [0.0, 0.7, 0.3, 0.0]
#          gates for [Scissors, Hook, Suction, Other]
```

### Scaling

- `α_base = 0.7`: Reduce vanilla LoRA influence slightly
- `α_expert = 1.0`: Full expert influence
- Tune based on validation metrics

---

## File Structure

```
outputs/
├── vanilla_lora/                # Stage 1 (user's output)
│   └── lora_final.safetensors   # Frozen as base in Stage 2
│
└── lora_moe_stage_b/            # Stage 2 (merged experts + router)
    ├── lora_moe_final.safetensors  # Final checkpoint
    ├── lora_moe_step_500.safetensors
    └── training_state_final.json

configs/
├── vanilla_lora.toml            # Stage 1 config (user's)
└── lora_moe_stage_b_merged.toml # Stage 2 config (merged)

examples/
└── train_lora_moe_2stage.sh     # Full 2-stage pipeline script
```

---

## Command Summary

### Stage 1 (User Already Did This)

```bash
python wan_train_network.py \
    --task t2v-A14B \
    --dit models/wan2.2/wan_t2v_A14B.safetensors \
    --output_dir outputs/vanilla_lora \
    --lora_dim 4 --lora_alpha 1.0 \
    --target_blocks last_8 \
    --num_train_epochs 10
```

### Stage 2 (Expert LoRAs + Router)

```bash
python wan_train_lora_moe.py \
    --config configs/lora_moe_stage_b_merged.toml \
    --task t2v-A14B \
    --training_stage stage_b \
    --base_lora_weights outputs/vanilla_lora/lora_final.safetensors \
    --train_router \
    --use_teacher_guidance \
    --routing_mode learned \
    --num_train_epochs 20
```

### Inference

```bash
python wan_inference_lora_moe.py \
    --lora_moe_weights outputs/lora_moe_stage_b/lora_moe_final.safetensors \
    --prompt "电凝钩切割组织" \
    --instrument_hint "Hook/Electrocautery" \
    --alpha_base 0.7 --alpha_expert 1.0 \
    --output_path output.mp4
```

---

## Advantages Over 3-Stage Approach

| Aspect | 3-Stage (Old) | 2-Stage (New) | Improvement |
|--------|---------------|---------------|-------------|
| **Training Time** | 10 + 15 + 5 = 30 epochs | 10 + 20 = 30 epochs | Same total, but simpler |
| **Joint Optimization** | ❌ Sequential | ✅ Simultaneous | Better expert-router fit |
| **Router Stability** | ⚠️ May collapse | ✅ Teacher-guided | More robust |
| **Code Complexity** | 3 configs + 3 scripts | 2 configs + 2 scripts | Simpler |
| **Expert Quality** | Good | **Better** | Co-adapted with router |
| **Routing Quality** | Good | **Better** | Learns from actual experts |

---

## FAQ

### Q: Why not train base LoRA + experts + router all together in 1 stage?

**A**: Too unstable. Base LoRA needs to learn general patterns first. If trained jointly, experts might learn redundant features instead of specializing.

### Q: Can I use rule-based routing instead of learned router?

**A**: Yes! Set `routing_mode = "rule_based"` and `train_router = false`. This uses instrument classifier logits directly. Simpler but less adaptive.

### Q: What if I don't have instrument labels?

**A**: Options:
1. Extract from Chinese captions (keyword matching)
2. Train a simple instrument classifier
3. Use rule-based routing with uniform fallback

### Q: How much VRAM needed?

**A**:
- 80GB A100: Batch size 1-2, no offloading
- 40GB A100: Batch size 1, offload inactive DiT
- 24GB 4090: Batch size 1, grad accum 16, block swap + offload

### Q: Can I fine-tune the router later (Stage 3)?

**A**: Yes, but rarely needed. If routing is suboptimal, you can:
```bash
python wan_train_lora_moe.py \
    --training_stage stage_c \
    --lora_moe_weights outputs/lora_moe_stage_b/lora_moe_final.safetensors \
    --num_train_epochs 5
```

---

## Summary

**Recommended Pipeline**:
1. ✅ **Stage 1**: Vanilla LoRA (user already did this)
2. ✅ **Stage 2**: Expert LoRAs + Router (train simultaneously)

**Key Insight**: Co-training experts and router produces better specialization and routing than training them separately.

**Total Training Time**: ~7-10 hours on 1xA100
**Total Parameters**: <20M trainable (efficient for 160 videos)

---

**Next Steps**: See `examples/train_lora_moe_2stage.sh` for the complete training script.
