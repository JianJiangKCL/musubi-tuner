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

# Base model checkpoints
DIT_PATH="${PROJECT_ROOT}/outputs/merged_low_noise.safetensors"
DIT_HIGH_NOISE_PATH="${PROJECT_ROOT}/outputs/merged_high_noise.safetensors"  # Optional
VAE_PATH="${CKPT_ROOT}/Wan2.1_VAE.pth"
T5_PATH="${CKPT_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"
CLIP_PATH="${CKPT_ROOT}/clip-vit-large-patch14"

# LoRA-MoE weights (trained from stage 2)
LORA_MOE_WEIGHTS="${PROJECT_ROOT}/outputs/lora_moe_stage2/lora_moe_final.safetensors"

# ============================================================================
# Generation Settings
# ============================================================================

# Task type: "i2v-A14B" for image-to-video, "t2v-A14B" for text-to-video
TASK="i2v-A14B"

# Prompt (in Chinese)
PROMPT="抓钳抓取组织。电凝钩切割血管。"

# Instrument hint for expert routing
# Options: "Scissors", "Hook/Electrocautery", "Suction", "Other"
# Leave empty for auto-detection from prompt
INSTRUMENT_HINT="Hook/Electrocautery"

# Input image (required for i2v task)
IMAGE_PATH="${PROJECT_ROOT}/Lap/example_frames/start_frame.png"

# Output settings
OUTPUT_DIR="${PROJECT_ROOT}/outputs/lora_moe_inference"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_PATH="${OUTPUT_DIR}/generated_video_$(date +%Y%m%d_%H%M%S).mp4"

# Video parameters
NUM_FRAMES=81
VIDEO_SIZE="256 256"  # height width
NUM_STEPS=40
GUIDANCE_SCALE=7.5
FLOW_SHIFT=7.0
SEED=42

# LoRA scaling factors
ALPHA_BASE=0.7    # Base LoRA strength
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
for path in "${DIT_PATH}" "${VAE_PATH}" "${T5_PATH}"; do
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
echo "  Guidance scale: ${GUIDANCE_SCALE}"
echo "  Seed: ${SEED}"
echo "  Alpha (base/expert): ${ALPHA_BASE}/${ALPHA_EXPERT}"
echo ""
echo "=== Starting Video Generation ==="
echo "Output will be saved to: ${OUTPUT_PATH}"
echo ""

# Build command
CMD="python -m musubi_tuner.wan_inference_lora_moe \
    --task ${TASK} \
    --dit \"${DIT_PATH}\" \
    --vae \"${VAE_PATH}\" \
    --t5 \"${T5_PATH}\" \
    --clip \"${CLIP_PATH}\" \
    --lora_moe_weights \"${LORA_MOE_WEIGHTS}\" \
    --prompt \"${PROMPT}\" \
    --num_frames ${NUM_FRAMES} \
    --video_size ${VIDEO_SIZE} \
    --num_steps ${NUM_STEPS} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --flow_shift ${FLOW_SHIFT} \
    --alpha_base ${ALPHA_BASE} \
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
    echo "  - Adjust alpha_base and alpha_expert to control LoRA strength"
    echo "  - Use --seed for reproducible generation"
else
    echo ""
    echo "=== Generation Failed ==="
    echo "Check error messages above"
    exit 1
fi
