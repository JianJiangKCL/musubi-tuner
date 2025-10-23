#!/bin/bash
# Complete LoRA-MoE Training Pipeline for WAN2.2
# This script demonstrates the full 3-stage training process

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== LoRA-MoE Training Pipeline for WAN2.2 ==="
echo "Project root: $PROJECT_ROOT"

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
STAGE_A_OUTPUT="./outputs/lora_moe_stage_a"
STAGE_B_OUTPUT="./outputs/lora_moe_stage_b"
STAGE_C_OUTPUT="./outputs/lora_moe_stage_c"

# ============================================================================
# STAGE A: Base LoRA Training
# ============================================================================
echo ""
echo "=== Stage A: Base LoRA Training ==="
echo "Goal: General adaptation to surgical video domain"
echo "Duration: ~10 epochs, 2-3 hours on 1xA100"
echo ""

python src/musubi_tuner/wan_train_lora_moe.py \
    --config configs/lora_moe_stage_a.toml \
    --task t2v-A14B \
    --training_stage stage_a \
    --dit "$DIT_PATH" \
    --dit_high_noise "$DIT_HIGH_NOISE_PATH" \
    --vae "$VAE_PATH" \
    --t5 "$T5_PATH" \
    --clip "$CLIP_PATH" \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --instrument_data_path "$INSTRUMENT_DATA_PATH" \
    --output_dir "$STAGE_A_OUTPUT" \
    --logging_dir "./logs/stage_a" \
    --num_train_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --offload_inactive_dit \
    --save_steps 500 \
    --logging_steps 10 \
    --seed 42

echo "Stage A complete! Checkpoint: $STAGE_A_OUTPUT/lora_moe_final.safetensors"

# ============================================================================
# STAGE B: Expert LoRA Training
# ============================================================================
echo ""
echo "=== Stage B: Expert LoRA Training ==="
echo "Goal: Instrument-specific specialization"
echo "Duration: ~15 epochs, 3-5 hours on 1xA100"
echo ""

python src/musubi_tuner/wan_train_lora_moe.py \
    --config configs/lora_moe_stage_b.toml \
    --task t2v-A14B \
    --training_stage stage_b \
    --dit "$DIT_PATH" \
    --dit_high_noise "$DIT_HIGH_NOISE_PATH" \
    --vae "$VAE_PATH" \
    --t5 "$T5_PATH" \
    --clip "$CLIP_PATH" \
    --base_lora_weights "$STAGE_A_OUTPUT/lora_moe_final.safetensors" \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --instrument_data_path "$INSTRUMENT_DATA_PATH" \
    --output_dir "$STAGE_B_OUTPUT" \
    --logging_dir "./logs/stage_b" \
    --num_train_epochs 15 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --lr_scheduler cosine \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --offload_inactive_dit \
    --use_roi_loss \
    --use_temporal_loss \
    --weight_roi_recon 3.0 \
    --weight_temporal 0.5 \
    --weight_routing_entropy 0.01 \
    --weight_routing_load_balance 0.05 \
    --save_steps 500 \
    --logging_steps 10 \
    --seed 42

echo "Stage B complete! Checkpoint: $STAGE_B_OUTPUT/lora_moe_final.safetensors"

# ============================================================================
# STAGE C: Learnable Router Training (Optional)
# ============================================================================
echo ""
echo "=== Stage C: Learnable Router Training (Optional) ==="
echo "Goal: Replace rule-based router with learned MLP"
echo "Duration: ~5 epochs, 1 hour on 1xA100"
echo ""
read -p "Run Stage C? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python src/musubi_tuner/wan_train_lora_moe.py \
        --config configs/lora_moe_stage_c.toml \
        --task t2v-A14B \
        --training_stage stage_c \
        --dit "$DIT_PATH" \
        --dit_high_noise "$DIT_HIGH_NOISE_PATH" \
        --vae "$VAE_PATH" \
        --t5 "$T5_PATH" \
        --clip "$CLIP_PATH" \
        --lora_moe_weights "$STAGE_B_OUTPUT/lora_moe_final.safetensors" \
        --train_data_dir "$TRAIN_DATA_DIR" \
        --instrument_data_path "$INSTRUMENT_DATA_PATH" \
        --output_dir "$STAGE_C_OUTPUT" \
        --logging_dir "./logs/stage_c" \
        --routing_mode learned \
        --num_train_epochs 5 \
        --batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-3 \
        --lr_scheduler cosine \
        --mixed_precision bf16 \
        --gradient_checkpointing \
        --save_steps 200 \
        --logging_steps 10 \
        --seed 42

    echo "Stage C complete! Checkpoint: $STAGE_C_OUTPUT/lora_moe_final.safetensors"
    FINAL_CHECKPOINT="$STAGE_C_OUTPUT/lora_moe_final.safetensors"
else
    echo "Skipping Stage C"
    FINAL_CHECKPOINT="$STAGE_B_OUTPUT/lora_moe_final.safetensors"
fi

# ============================================================================
# Inference Example
# ============================================================================
echo ""
echo "=== Inference Example ==="
echo "Generating surgical video with LoRA-MoE"
echo ""

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
    --output_path outputs/demo_hook.mp4 \
    --seed 42

echo ""
echo "=== Pipeline Complete! ==="
echo "Final weights: $FINAL_CHECKPOINT"
echo "Demo video: outputs/demo_hook.mp4"
echo ""
echo "Next steps:"
echo "1. Evaluate with metrics (tip PCK/IoU, identity accuracy)"
echo "2. Run ablation studies (see docs/lora_moe_guide.md)"
echo "3. Generate videos for all 4 instrument types"
echo "4. Fine-tune hyperparameters based on results"
