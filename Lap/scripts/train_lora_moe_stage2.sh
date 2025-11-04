#!/bin/bash
# LoRA-MoE Stage 2 Training Script
# Trains Expert LoRAs + Router simultaneously on cached latents

# This script assumes:
# 1. Stage 1 vanilla LoRA training is complete
# 2. Cached latents and text features exist
# 3. Instrument labels have been extracted from captions

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Adjust to your available GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
# Paths (ADJUST THESE TO YOUR SETUP)
PROJECT_ROOT="/mnt/cfs/jj/proj/musubi-tuner"
DATA_ROOT="/mnt/cfs/jj/proj/musubi-tuner/Lap"
CACHE_ROOT="/mnt/cfs/jj/proj/musubi-tuner/Lap/cache/trace50_allvideos_20s"
CKPT_ROOT="/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B"

# Accelerate multi-GPU config
ACCELERATE_CONFIG="/mnt/cfs/jj/proj/musubi-tuner/eight_gpu_config.yaml"

# Ensure src-layout package is importable
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Activate conda environment
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate musu

# Output directory
# Experiment tag (version). Priority: $1 > $EXP_TAG env > timestamp
EXP_TAG="${1:-${EXP_TAG}}"
if [ -z "${EXP_TAG}" ]; then
    EXP_TAG=$(date +%Y%m%d_%H%M%S)
fi

OUTPUT_DIR="${PROJECT_ROOT}/outputs/lora_moe_stage2/${EXP_TAG}"
mkdir -p "${OUTPUT_DIR}"
# Dataset config (from Stage 1)
DATASET_CONFIG="${DATA_ROOT}/config/trace50_all_videos_20s.toml"

# Instrument-augmented clips JSONL
# You should run augment_clips_with_instruments.py first to generate this
CLIPS_WITH_INSTRUMENTS="${PROJECT_ROOT}/Lap/preprocessing/filtered_clips_with_instruments.jsonl"

# Model checkpoints
DIT_PATH="/mnt/cfs/jj/proj/musubi-tuner/outputs/merged_low_noise_v3.safetensors"
DIT_HIGH_NOISE_PATH="/mnt/cfs/jj/proj/musubi-tuner/outputs/merged_high_noise_v3.safetensors"  # merged WAN2.2 split weights
VAE_PATH="${CKPT_ROOT}/Wan2.1_VAE.pth"
T5_PATH="${CKPT_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"
CLIP_PATH="${CKPT_ROOT}/clip-vit-large-patch14"  # Or appropriate CLIP path



# ============================================================================
# Pre-flight Checks
# ============================================================================

echo "=== LoRA-MoE Stage 2 Training ==="
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Check DiT base checkpoints (v3 merged)
if [ ! -f "${DIT_PATH}" ]; then
    echo "ERROR: Low-noise DiT checkpoint not found: ${DIT_PATH}"
    exit 1
fi
echo "✓ Found low-noise DiT: ${DIT_PATH}"

if [ ! -f "${DIT_HIGH_NOISE_PATH}" ]; then
    echo "ERROR: High-noise DiT checkpoint not found: ${DIT_HIGH_NOISE_PATH}"
    exit 1
fi
echo "✓ Found high-noise DiT: ${DIT_HIGH_NOISE_PATH}"

# Check Accelerate config
if [ ! -f "${ACCELERATE_CONFIG}" ]; then
    echo "ERROR: Accelerate config not found: ${ACCELERATE_CONFIG}"
    exit 1
fi
echo "✓ Using Accelerate config: ${ACCELERATE_CONFIG}"

# Check cached data
if [ ! -d "${CACHE_ROOT}" ]; then
    echo "ERROR: Cache directory not found: ${CACHE_ROOT}"
    echo "Please run wan_cache_latents.py and wan_cache_text_encoder_outputs.py first."
    exit 1
fi
echo "✓ Found cache directory: ${CACHE_ROOT}"

# Ensure dataset config already points to the correct cache_directory in Stage 1 config

# Check instrument-augmented clips
if [ ! -f "${CLIPS_WITH_INSTRUMENTS}" ]; then
    echo "WARNING: Instrument-augmented clips not found: ${CLIPS_WITH_INSTRUMENTS}"
    echo ""
    echo "Attempting to generate from filtered_clips_processed.jsonl..."

    FILTERED_CLIPS="${PROJECT_ROOT}/Lap/preprocessing/filtered_clips_processed.jsonl"
    if [ ! -f "${FILTERED_CLIPS}" ]; then
        echo "ERROR: filtered_clips_processed.jsonl not found: ${FILTERED_CLIPS}"
        exit 1
    fi

    # Run augmentation script
    python "${PROJECT_ROOT}/Lap/scripts/augment_clips_with_instruments.py" \
        --input "${FILTERED_CLIPS}" \
        --output "${CLIPS_WITH_INSTRUMENTS}" \
        --soft_labels

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to augment clips with instrument labels"
        exit 1
    fi
fi
echo "✓ Found instrument-augmented clips: ${CLIPS_WITH_INSTRUMENTS}"

# ============================================================================
# Training Configuration
# ============================================================================

# External LoRA-MoE config file (YAML/TOML/JSON)
LORA_MOE_CONFIG_FILE="${PROJECT_ROOT}/Lap/config/lora_moe.yaml"

# Training parameters
NUM_EPOCHS=24
BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
LEARNING_RATE=5e-5
MIXED_PRECISION="bf16"

# Loss weights
WEIGHT_DIFFUSION=1.0
WEIGHT_ROI=3.0
WEIGHT_IDENTITY=0.0  # Set to 0.5 if you have instrument classifier
WEIGHT_TEMPORAL=0.0  # Set to 0.5 if you have optical flow model
WEIGHT_ROUTING_ENTROPY=0.08
WEIGHT_ROUTING_LOAD_BALANCE=0.25



echo ""
echo "Training Configuration:"
echo "  Experts: ${NUM_EXPERTS}"
echo "  LoRA rank: Q=${RANK_Q}, K/V/O=${RANK_K}, FFN=${RANK_FFN}"
echo "  Router: ${ROUTING_MODE} (hidden_dim=${ROUTER_HIDDEN_DIM})"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Batch size: ${BATCH_SIZE} x ${GRAD_ACCUM_STEPS} grad accum"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  GPUs: ${NUM_GPUS}"
echo ""

# ============================================================================
# Launch Training
# ============================================================================

echo "=== Starting LoRA-MoE Training ==="
echo "Output directory: ${OUTPUT_DIR}"
echo "Experiment tag: ${EXP_TAG}"
echo ""

exec accelerate launch --config_file "${ACCELERATE_CONFIG}" -m musubi_tuner.wan_train_lora_moe \
    --sdpa \
    --task i2v-A14B \
    --training_stage stage_b \
    --output_dir "${OUTPUT_DIR}" \
    --output_name lora-moe-stage2-${EXP_TAG} \
    --logging_dir "${OUTPUT_DIR}/logs" \
    \
    --dit "${DIT_PATH}" \
    --dit_high_noise "${DIT_HIGH_NOISE_PATH}" \
    --vae "${VAE_PATH}" \
    --t5 "${T5_PATH}" \
    --clip "${CLIP_PATH}" \
    \
    --dataset_config "${DATASET_CONFIG}" \
    --log_config \
    --instrument_data_path "${CLIPS_WITH_INSTRUMENTS}" \
    --train_router \
    --use_teacher_guidance \
    --teacher_kl_weight 0.4 \
    --lora_moe_config_file "${LORA_MOE_CONFIG_FILE}" \
    \
    --max_train_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --optimizer_type AdamW \
    --lr_scheduler cosine \
    --lr_warmup_steps 100 \
    --mixed_precision bf16 \
    --save_every_n_epochs 1 \
    \
    --min_timestep 875 \
    --max_timestep 1000 \
    --preserve_distribution_shape \
    \
    --weight_base_diffusion ${WEIGHT_DIFFUSION} \
    --weight_roi_recon ${WEIGHT_ROI} \
    --weight_identity ${WEIGHT_IDENTITY} \
    --weight_temporal ${WEIGHT_TEMPORAL} \
    --weight_routing_entropy ${WEIGHT_ROUTING_ENTROPY} \
    --weight_routing_load_balance ${WEIGHT_ROUTING_LOAD_BALANCE} \
    \
    # --gradient_checkpointing \
    --offload_inactive_dit \
    --max_grad_norm 1.0 \
    --seed 42
    # --save_every_n_steps 10 \


# ============================================================================
# Training Complete
# ============================================================================

if [ $? -eq 0 ]; then
    echo ""
    echo "=== LoRA-MoE Training Complete! ==="
    echo "Output: ${OUTPUT_DIR}/lora_moe_final.safetensors"
    echo ""
    echo "This checkpoint contains:"
    echo "  - 6 trained expert LoRAs (bipolar, clipper, grasper, hook, irrigator, scissors)"
    echo "  - Learned router (MLP with teacher guidance)"
    echo "  - Base: DiT v3 merged high/low-noise checkpoints"
    echo ""
    echo "Next steps:"
    echo "  1. Run inference with different instrument hints"
    echo "  2. Evaluate tip accuracy and identity preservation"
    echo "  3. Compare against vanilla LoRA baseline"
else
    echo ""
    echo "=== Training Failed ==="
    echo "Check logs in: ${OUTPUT_DIR}/logs"
    exit 1
fi
