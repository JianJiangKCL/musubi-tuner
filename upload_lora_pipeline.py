#!/usr/bin/env python3
"""
LORA Model Upload Pipeline

Automatically uploads LORA models to liblib.art using the existing pipeline.

Usage:
    python upload_lora_pipeline.py <lora_weight>

Arguments:
    lora_weight: The weight/step number for the LORA model (e.g., 40, 100, etc.)
                 Will look for files like: attack-of-lorac-{lora_weight:06d}.safetensors
                 If not found, will try: attack-of-lorac.safetensors (final version)

Example:
    python upload_lora_pipeline.py 40
    python upload_lora_pipeline.py 100
"""

import os
import sys
import csv
import time
from datetime import datetime
import importlib.util

def load_upload_g2():
    """Load the upload_g2 module"""
    spec = importlib.util.spec_from_file_location("upload_g2", "scripts/upload_g2.py")
    upload_g2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upload_g2)
    return upload_g2

def find_model_file(lora_weight):
    """Find the model file based on lora_weight"""
    base_dir = "/mnt/cfs/jj/musubi-tuner/lora_outputs/attack/high_noise_trace30"

    # Try formatted filename first (e.g., attack-of-lorac-000040.safetensors)
    formatted_name = f"attack-of-lorac-{int(lora_weight):06d}.safetensors"
    formatted_path = os.path.join(base_dir, formatted_name)

    if os.path.exists(formatted_path):
        print(f"Found model file: {formatted_name}")
        return formatted_path

    # Try the base filename (final version)
    base_name = "attack-of-lorac.safetensors"
    base_path = os.path.join(base_dir, base_name)

    if os.path.exists(base_path):
        print(f"Using final model file: {base_name}")
        return base_path

    # List available files for debugging
    if os.path.exists(base_dir):
        available_files = [f for f in os.listdir(base_dir) if f.endswith('.safetensors')]
        print(f"Available model files in {base_dir}:")
        for f in available_files:
            print(f"  - {f}")

    raise FileNotFoundError(f"No suitable model file found for lora_weight {lora_weight}")

def get_model_metadata(file_path, lora_weight):
    """Generate metadata for the model"""
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)

    # Extract model name from filename
    if file_name.startswith("attack-of-lorac-"):
        if file_name.endswith(".safetensors"):
            base_name = file_name[:-13]  # Remove .safetensors
            if base_name.endswith("-000000"):  # Remove -000000 suffix
                model_name = base_name[:-7]
            else:
                model_name = base_name
        else:
            model_name = file_name
    else:
        model_name = "Attack of Lorac"

    # Create version name based on weight
    if str(lora_weight).isdigit() and int(lora_weight) > 0:
        version_name = f"High Noise Trace30 Step{int(lora_weight)}"
    else:
        version_name = "High Noise Trace30 Final"

    return {
        'model_name': model_name,
        'version_name': version_name,
        'file_size': file_size,
        'file_path': file_path,
        'file_name': file_name,
        'trigger_word': 'attack of lorac',
        'request_id': int(lora_weight) if str(lora_weight).isdigit() else 1
    }

def upload_to_oss(model_file_path, metadata):
    """Upload model file to OSS"""
    print(f"\n{'='*60}")
    print("Step 1: Uploading to OSS")
    print(f"{'='*60}")

    upload_g2 = load_upload_g2()
    oss_url, model_dir_uuid = upload_g2.upload_model_file(model_file_path)

    if not oss_url:
        raise Exception("Failed to upload model to OSS")

    print(f"✅ OSS upload successful!")
    print(f"   URL: {oss_url}")
    print(f"   Model Dir UUID: {model_dir_uuid}")

    return oss_url, model_dir_uuid

def create_upload_csv(oss_url, metadata):
    """Create CSV file for liblib.art upload"""
    print(f"\n{'='*60}")
    print("Step 2: Creating upload CSV")
    print(f"{'='*60}")

    # Extract the path part from OSS URL for the online_path field
    # OSS URL: https://liblibai-online.liblib.cloud/web/model/{model_dir_uuid}/{model_file_uuid}.safetensors
    # We need: web/model/{model_dir_uuid}/{model_file_uuid}.safetensors
    if 'liblibai-online.liblib.cloud/' in oss_url:
        online_path = oss_url.split('liblibai-online.liblib.cloud/')[1]
    else:
        online_path = oss_url

    csv_data = [
        ['request_id', '模型名称', '模型版本名称', 'model_size', 'online_path', '触发词', 'file_path'],
        [
            metadata['request_id'],
            metadata['model_name'],
            metadata['version_name'],
            metadata['file_size'],
            online_path,
            metadata['trigger_word'],
            metadata['file_path']
        ]
    ]

    csv_filename = f"lora_upload_weight_{metadata['request_id']}.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"✅ CSV file created: {csv_filename}")
    print(f"   Model: {metadata['model_name']}")
    print(f"   Version: {metadata['version_name']}")
    print(f"   Size: {metadata['file_size'] / 1024 / 1024:.1f} MB")
    print(f"   Trigger: {metadata['trigger_word']}")

    return csv_filename

def upload_to_liblib(csv_filename):
    """Upload to liblib.art using existing script"""
    print(f"\n{'='*60}")
    print("Step 3: Uploading to liblib.art")
    print(f"{'='*60}")

    # Import and run the upload_to_liblib function
    sys.path.append('scripts')
    from upload_to_liblib import upload_to_liblib

    result = upload_to_liblib(csv_filename)

    print("✅ liblib.art upload process completed!")
    return result

def main():
    if len(sys.argv) != 2:
        print("Usage: python upload_lora_pipeline.py <lora_weight>")
        print("Example: python upload_lora_pipeline.py 40")
        sys.exit(1)

    try:
        lora_weight = sys.argv[1]
        print(f"🚀 Starting LORA upload pipeline for weight: {lora_weight}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Find model file
        print(f"\n{'='*60}")
        print("Step 0: Finding model file")
        print(f"{'='*60}")
        model_file_path = find_model_file(lora_weight)

        # Step 2: Get metadata
        metadata = get_model_metadata(model_file_path, lora_weight)
        print(f"Model metadata:")
        print(f"  File: {metadata['file_name']}")
        print(f"  Size: {metadata['file_size'] / 1024 / 1024:.1f} MB")
        print(f"  Model: {metadata['model_name']}")
        print(f"  Version: {metadata['version_name']}")

        # Step 3: Upload to OSS
        oss_url, model_dir_uuid = upload_to_oss(model_file_path, metadata)

        # Step 4: Create CSV
        csv_filename = create_upload_csv(oss_url, metadata)

        # Step 5: Upload to liblib.art
        upload_to_liblib(csv_filename)

        print(f"\n{'='*60}")
        print("🎉 UPLOAD COMPLETE!")
        print(f"{'='*60}")
        print(f"Model: {metadata['model_name']}")
        print(f"Version: {metadata['version_name']}")
        print(f"Weight: {lora_weight}")
        print(f"OSS URL: {oss_url}")
        print(f"CSV File: {csv_filename}")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()








