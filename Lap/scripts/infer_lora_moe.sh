#!/bin/bash
# LoRA-MoE Inference Script for WAN2.2
# Generates videos using trained LoRA-MoE weights with instrument-specific adaptation

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Paths (ADJUST THESE TO YOUR SETUP)
PROJECT_ROOT="/mnt/cfs/jj/proj/musubi-tuner"
CKPT_ROOT="/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B"

# Ensure src-layout package is importable
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Activate conda environment
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate musu

# ============================================================================
# Model Paths
# ============================================================================

# Base model checkpoints (Stage 1 LoRA already merged into DiT v3)
DIT_PATH="${PROJECT_ROOT}/outputs/merged_low_noise_v3.safetensors"
DIT_HIGH_NOISE_PATH="${PROJECT_ROOT}/outputs/merged_high_noise_v3.safetensors"
VAE_PATH="${CKPT_ROOT}/Wan2.1_VAE.pth"
T5_PATH="${CKPT_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"

# LoRA-MoE weights (trained from stage 2)
# Use latest from 8gpu_v3_nogc if available, else fallback to a dated run
# if [ -z "${LORA_MOE_WEIGHTS}" ]; then
#     MOE_DIR="/mnt/cfs/jj/proj/musubi-tuner/outputs/lora_moe_stage2/8gpu_v3_nogc"
#     if ls ${MOE_DIR}/lora-moe-stage2-*-0*.safetensors >/dev/null 2>&1; then
#         # Prefer numbered incremental checkpoints to avoid oversized final files
#         LORA_MOE_WEIGHTS=$(ls -t ${MOE_DIR}/lora-moe-stage2-*-0*.safetensors | head -n1)
#     elif ls ${MOE_DIR}/*.safetensors >/dev/null 2>&1; then
#         LORA_MOE_WEIGHTS=$(ls -t ${MOE_DIR}/*.safetensors | head -n1)
#     else
#         LORA_MOE_WEIGHTS="/mnt/cfs/jj/proj/musubi-tuner/outputs/lora_moe_stage2/20251026_021123/lora-moe-stage2-20251026_021123.safetensors"
#     fi
# fi
LORA_MOE_WEIGHTS="/mnt/cfs/jj/proj/musubi-tuner/outputs/lora_moe_stage2/20251028_231504/lora-moe-stage2-20251028_231504.safetensors"
# ============================================================================
# Generation Settings
# ============================================================================

# Task type: "i2v-A14B" for image-to-video, "t2v-A14B" for text-to-video
TASK="i2v-A14B"

# Prompt (in Chinese)
PROMPT="电凝钩分离胆囊周围的网膜组织"

# Instrument hint for expert routing (must be one of: bipolar, clipper, grasper, hook, irrigator, scissors)
# Leave empty for auto-detection from prompt
INSTRUMENT_HINT="hook"

# Input image (required for i2v task)
IMAGE_PATH="/mnt/cfs/jj/proj/musubi-tuner/Lap/clips_all_video/cholec02/first_frames/cholec02_0046_part1.jpg"

# Output settings
OUTPUT_DIR="${PROJECT_ROOT}/outputs/lora_moe_inference"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_PATH="${OUTPUT_DIR}/generated_video_$(date +%Y%m%d_%H%M%S).mp4"

# Video parameters
NUM_FRAMES=161
VIDEO_SIZE="320 480"  # height width
NUM_STEPS=50
SEED=42

# LoRA scaling factors (Stage 1 base LoRA merged into DiT v3)
ALPHA_EXPERT=1.0  # Expert LoRA strength

# GPU settings
export CUDA_VISIBLE_DEVICES=7

# ============================================================================
# Pre-flight Checks
# ============================================================================

echo "=== LoRA-MoE Inference ==="
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Check LoRA-MoE weights
if [ ! -f "${LORA_MOE_WEIGHTS}" ]; then
    echo "ERROR: LoRA-MoE weights not found: ${LORA_MOE_WEIGHTS}"
    echo ""
    echo "Please ensure Stage 2 training is complete."
    echo "Expected output: outputs/lora_moe_stage2/lora_moe_final.safetensors"
    exit 1
fi
echo "✓ Found LoRA-MoE weights: ${LORA_MOE_WEIGHTS}"

# Check base model files
for path in "${DIT_PATH}" "${DIT_HIGH_NOISE_PATH}" "${VAE_PATH}" "${T5_PATH}"; do
    if [ ! -f "${path}" ] && [ ! -d "${path}" ]; then
        echo "ERROR: Model file not found: ${path}"
        exit 1
    fi
done
echo "✓ Found base model checkpoints"

# Check input image for i2v
if [ "${TASK}" == "i2v-A14B" ] || [ "${TASK}" == "i2v-5B" ]; then
    if [ -z "${IMAGE_PATH}" ] || [ ! -f "${IMAGE_PATH}" ]; then
        echo "ERROR: Image path required for i2v task: ${IMAGE_PATH}"
        exit 1
    fi
    echo "✓ Found input image: ${IMAGE_PATH}"
fi

# ============================================================================
# Run Inference
# ============================================================================

echo ""
echo "Generation Settings:"
echo "  Task: ${TASK}"
echo "  Prompt: ${PROMPT}"
echo "  Instrument: ${INSTRUMENT_HINT:-auto-detected}"
echo "  Frames: ${NUM_FRAMES}"
echo "  Steps: ${NUM_STEPS}"
echo "  Seed: ${SEED}"
echo "  Alpha (expert): ${ALPHA_EXPERT}"
echo ""
echo "=== Starting Video Generation ==="
echo "Output will be saved to: ${OUTPUT_PATH}"
echo ""

# Build command
CMD="python -m musubi_tuner.wan_inference_lora_moe \
    --task ${TASK} \
    --dit \"${DIT_PATH}\" \
    --dit_high_noise \"${DIT_HIGH_NOISE_PATH}\" \
    --vae \"${VAE_PATH}\" \
    --t5 \"${T5_PATH}\" \
    --lora_moe_weights \"${LORA_MOE_WEIGHTS}\" \
    --lora_moe_config_file \"${PROJECT_ROOT}/Lap/config/lora_moe.yaml\" \
    --prompt \"${PROMPT}\" \
    --num_frames ${NUM_FRAMES} \
    --video_size ${VIDEO_SIZE} \
    --num_steps ${NUM_STEPS} \
    --alpha_expert ${ALPHA_EXPERT} \
    --seed ${SEED} \
    --output_path \"${OUTPUT_PATH}\""

# Add optional arguments
if [ -n "${INSTRUMENT_HINT}" ]; then
    CMD="${CMD} --instrument_hint \"${INSTRUMENT_HINT}\""
fi

if [ -n "${IMAGE_PATH}" ]; then
    CMD="${CMD} --image_path \"${IMAGE_PATH}\""
fi

# Execute
eval ${CMD}

# ============================================================================
# Complete
# ============================================================================

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Generation Complete! ==="
    echo "Video saved to: ${OUTPUT_PATH}"
    echo ""
    echo "Tips:"
    echo "  - Try different instrument hints to see expert routing effects"
    echo "  - Adjust alpha_expert to control LoRA strength"
    echo "  - Use --seed for reproducible generation"
else
    echo ""
    echo "=== Generation Failed ==="
    echo "Check error messages above"
    exit 1
fi
