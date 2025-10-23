#!/bin/bash
# Simplified 2-Stage LoRA-MoE Training Pipeline for WAN2.2
# Stage 1: Vanilla LoRA (already done by user)
# Stage 2: Expert LoRAs + Router (simultaneous training)

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== 2-Stage LoRA-MoE Training Pipeline for WAN2.2 ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Model paths (adjust to your setup)
DIT_PATH="./models/wan2.2/wan_t2v_A14B.safetensors"
DIT_HIGH_NOISE_PATH="./models/wan2.2/wan_t2v_A14B_high_noise.safetensors"
VAE_PATH="./models/wan2.2/vae.safetensors"
T5_PATH="./models/t5-v1_1-xxl"
CLIP_PATH="./models/clip-vit-large-patch14"

# Data paths
TRAIN_DATA_DIR="./data/surgical_videos"
INSTRUMENT_DATA_PATH="./Lap/preprocessing/filtered_clips_processed.jsonl"

# Output directories
VANILLA_LORA_OUTPUT="./outputs/vanilla_lora"  # Stage 1 output (already done)
STAGE_B_OUTPUT="./outputs/lora_moe_stage_b"   # Stage 2 output

# ============================================================================
# STAGE 1: Vanilla LoRA Training (ALREADY DONE BY USER)
# ============================================================================
echo "=== Stage 1: Vanilla LoRA Training ==="
echo "Status: ✓ COMPLETED (by user)"
echo "Output: $VANILLA_LORA_OUTPUT/lora_final.safetensors"
echo ""
echo "This stage trains basic LoRA adaptation on surgical videos."
echo "You have already completed this step with standard WAN LoRA training."
echo ""

# Check if vanilla LoRA weights exist
if [ ! -f "$VANILLA_LORA_OUTPUT/lora_final.safetensors" ]; then
    echo "ERROR: Vanilla LoRA weights not found at $VANILLA_LORA_OUTPUT/lora_final.safetensors"
    echo "Please ensure you have completed Stage 1 vanilla LoRA training first."
    echo ""
    echo "To train vanilla LoRA (Stage 1), use:"
    echo "  python src/musubi_tuner/wan_train_network.py \\"
    echo "    --task t2v-A14B \\"
    echo "    --dit $DIT_PATH \\"
    echo "    --dit_high_noise $DIT_HIGH_NOISE_PATH \\"
    echo "    --vae $VAE_PATH \\"
    echo "    --t5 $T5_PATH \\"
    echo "    --clip $CLIP_PATH \\"
    echo "    --train_data_dir $TRAIN_DATA_DIR \\"
    echo "    --output_dir $VANILLA_LORA_OUTPUT \\"
    echo "    --lora_dim 4 \\"
    echo "    --lora_alpha 1.0 \\"
    echo "    --target_blocks last_8 \\"
    echo "    --num_train_epochs 10"
    exit 1
fi

echo "Found vanilla LoRA weights: $VANILLA_LORA_OUTPUT/lora_final.safetensors"
echo "Proceeding to Stage 2..."
echo ""

# ============================================================================
# STAGE 2: Expert LoRAs + Router Training (SIMULTANEOUS)
# ============================================================================
echo "=== Stage 2: Expert LoRAs + Router Training (Simultaneous) ==="
echo "Goal: Train 4 instrument-specific experts + learnable router"
echo "Duration: ~5-7 hours on 1xA100 (20 epochs)"
echo ""

python src/musubi_tuner/wan_train_lora_moe.py \
    --config configs/lora_moe_stage_b_merged.toml \
    --task t2v-A14B \
    --training_stage stage_b \
    --dit "$DIT_PATH" \
    --dit_high_noise "$DIT_HIGH_NOISE_PATH" \
    --vae "$VAE_PATH" \
    --t5 "$T5_PATH" \
    --clip "$CLIP_PATH" \
    --base_lora_weights "$VANILLA_LORA_OUTPUT/lora_final.safetensors" \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --instrument_data_path "$INSTRUMENT_DATA_PATH" \
    --output_dir "$STAGE_B_OUTPUT" \
    --logging_dir "./logs/stage_b" \
    --num_train_epochs 20 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --offload_inactive_dit \
    --routing_mode learned \
    --train_router \
    --use_teacher_guidance \
    --teacher_kl_weight 1.0 \
    --use_roi_loss \
    --weight_roi_recon 3.0 \
    --weight_routing_entropy 0.02 \
    --weight_routing_load_balance 0.1 \
    --save_steps 500 \
    --logging_steps 10 \
    --seed 42

echo ""
echo "Stage 2 complete! Checkpoint: $STAGE_B_OUTPUT/lora_moe_final.safetensors"
echo ""

# ============================================================================
# Inference Example
# ============================================================================
echo "=== Inference Example ==="
echo "Generating surgical video with LoRA-MoE"
echo ""

FINAL_CHECKPOINT="$STAGE_B_OUTPUT/lora_moe_final.safetensors"

python src/musubi_tuner/wan_inference_lora_moe.py \
    --task t2v-A14B \
    --dit "$DIT_PATH" \
    --dit_high_noise "$DIT_HIGH_NOISE_PATH" \
    --vae "$VAE_PATH" \
    --t5 "$T5_PATH" \
    --clip "$CLIP_PATH" \
    --lora_moe_weights "$FINAL_CHECKPOINT" \
    --prompt "电凝钩切割组织。吸引止血。抓钳抓取。" \
    --instrument_hint "Hook/Electrocautery" \
    --num_frames 81 \
    --num_steps 40 \
    --guidance_scale 7.5 \
    --alpha_base 0.7 \
    --alpha_expert 1.0 \
    --output_path outputs/demo_hook_2stage.mp4 \
    --seed 42

echo ""
echo "=== 2-Stage Pipeline Complete! ==="
echo "Final weights: $FINAL_CHECKPOINT"
echo "Demo video: outputs/demo_hook_2stage.mp4"
echo ""
echo "Summary:"
echo "  Stage 1 (Vanilla LoRA): ✓ Completed (by user)"
echo "  Stage 2 (Experts + Router): ✓ Completed"
echo ""
echo "What was trained:"
echo "  - 4 Instrument-specific expert LoRAs (Scissors, Hook, Suction, Other)"
echo "  - Learnable MLP router with rule-based teacher guidance"
echo "  - Frozen base LoRA from Stage 1 (shared adaptation)"
echo ""
echo "Next steps:"
echo "  1. Evaluate with metrics (tip PCK/IoU, identity accuracy)"
echo "  2. Generate videos for all 4 instrument types"
echo "  3. Compare against vanilla LoRA baseline"
echo "  4. Run ablation studies (see docs/lora_moe_guide.md)"
