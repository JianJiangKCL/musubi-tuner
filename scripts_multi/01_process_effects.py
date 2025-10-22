#!/usr/bin/env python3
"""
Process video effects: unzip, create video_captions.jsonl, and generate TOML configs
"""

import os
import json
import zipfile
from pathlib import Path
import re

INPUT_DIR = "/mnt/cfs/jj/musubi-tuner/2025_10_21"
CAPTION_FILE = os.path.join(INPUT_DIR, "caption.txt")
CONFIG_OUTPUT_DIR = os.path.join(INPUT_DIR, "config")
CACHE_BASE_DIR = os.path.join(INPUT_DIR, "cache")

# VAE and T5 paths (from reference script)
VAE_PATH = "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
T5_PATH = "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth"


def read_captions():
    """Read caption.txt and parse effect names and captions"""
    captions = {}
    with open(CAPTION_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by lines and parse each caption
    lines = content.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue
        
        # Extract effect name (everything before the first Chinese character)
        effect_name = ""
        for i, char in enumerate(line):
            # Check if character is Chinese (CJK Unified Ideographs)
            if '\u4e00' <= char <= '\u9fff':
                # Found first Chinese character, extract effect name before it
                effect_name = line[:i].strip()
                break
        
        if effect_name:
            caption = line  # Keep full line as caption
            captions[effect_name] = caption
    
    return captions


def unzip_files():
    """Unzip all zip files in INPUT_DIR if not already unzipped"""
    print("=" * 60)
    print("Step 1: Unzipping files")
    print("=" * 60)
    
    zip_files = list(Path(INPUT_DIR).glob("*.zip"))
    
    for zip_path in zip_files:
        # Determine output directory (remove .zip and any trailing numbers)
        effect_name = zip_path.stem
        # Remove trailing numbers like " 2"
        effect_name = re.sub(r'\s+\d+$', '', effect_name)
        
        output_dir = os.path.join(INPUT_DIR, effect_name)
        # Also check for the renamed version (with underscores)
        output_dir_renamed = os.path.join(INPUT_DIR, effect_name.replace(' ', '_'))
        
        # Check if already unzipped (either with spaces or underscores)
        if (os.path.exists(output_dir) and os.listdir(output_dir)) or \
           (os.path.exists(output_dir_renamed) and os.listdir(output_dir_renamed)):
            print(f"✓ Already unzipped: {effect_name}")
            continue
        
        print(f"Unzipping: {zip_path.name} -> {effect_name}/")
        os.makedirs(output_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Clean up __MACOSX folders and ._ files
        import subprocess
        subprocess.run(f'find "{output_dir}" -name "__MACOSX" -type d -exec rm -rf {{}} + 2>/dev/null || true', 
                      shell=True, check=False)
        subprocess.run(f'find "{output_dir}" -name "._*" -type f -delete 2>/dev/null || true', 
                      shell=True, check=False)
        
        print(f"✓ Unzipped: {effect_name}")
    
    print()


def find_video_files(directory):
    """Find all video files in a directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip __MACOSX and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__MACOSX']
        
        for file in files:
            # Skip hidden files (starting with . or ._)
            if file.startswith('.') or file.startswith('._'):
                continue
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    return sorted(video_files)


def create_video_captions_jsonl():
    """Create video_captions.jsonl for each effect"""
    print("=" * 60)
    print("Step 2: Creating video_captions.jsonl files")
    print("=" * 60)
    
    captions = read_captions()
    effect_dirs = []
    
    # Find all effect directories (unzipped folders)
    for item in os.listdir(INPUT_DIR):
        item_path = os.path.join(INPUT_DIR, item)
        if os.path.isdir(item_path) and item not in ['cache', 'config']:
            effect_dirs.append(item)
    
    for effect_dir in sorted(effect_dirs):
        effect_path = os.path.join(INPUT_DIR, effect_dir)
        
        # Find matching caption
        caption = None
        caption_key_matched = None
        for caption_key in captions.keys():
            # Flexible matching: normalize both names for comparison
            # Remove spaces and underscores, convert to lowercase
            normalized_caption_key = caption_key.lower().replace(' ', '').replace('_', '')
            normalized_effect_dir = effect_dir.lower().replace(' ', '').replace('_', '')
            
            if normalized_caption_key == normalized_effect_dir:
                caption = captions[caption_key]
                caption_key_matched = caption_key
                break
        
        if not caption:
            print(f"⚠ Warning: No caption found for {effect_dir}, skipping")
            continue
        
        # Add "Effect" after the effect name in the caption
        # Replace "Effect_Name description" with "Effect_Name Effect description"
        if caption_key_matched:
            caption = caption.replace(caption_key_matched, f"{caption_key_matched} Effect", 1)
        
        # Find video files
        video_files = find_video_files(effect_path)
        
        if not video_files:
            print(f"⚠ Warning: No video files found in {effect_dir}, skipping")
            continue
        
        # Create dataset subdirectory
        dataset_dir = os.path.join(effect_path, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create video_captions.jsonl
        jsonl_path = os.path.join(dataset_dir, "video_captions.jsonl")
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for video_file in video_files:
                entry = {
                    "video_path": video_file,
                    "caption": caption
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"✓ Created: {effect_dir}/dataset/video_captions.jsonl ({len(video_files)} videos)")
    
    print()


def create_toml_configs():
    """Create TOML config files for each effect"""
    print("=" * 60)
    print("Step 3: Creating TOML config files")
    print("=" * 60)
    
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    # Find all effect directories with video_captions.jsonl
    effect_dirs = []
    for item in os.listdir(INPUT_DIR):
        item_path = os.path.join(INPUT_DIR, item)
        if os.path.isdir(item_path) and item not in ['cache', 'config']:
            jsonl_path = os.path.join(item_path, "dataset", "video_captions.jsonl")
            if os.path.exists(jsonl_path):
                effect_dirs.append(item)
    
    for effect_dir in sorted(effect_dirs):
        effect_path = os.path.join(INPUT_DIR, effect_dir)
        
        # Create safe name for config file
        safe_name = effect_dir.lower().replace(' ', '_').replace('-', '_')
        config_name = f"{safe_name}.toml"
        config_path = os.path.join(CONFIG_OUTPUT_DIR, config_name)
        
        # Create cache directory path
        cache_dir = os.path.join(CACHE_BASE_DIR, safe_name)
        jsonl_file = os.path.join(effect_path, "dataset", "video_captions.jsonl")
        
        # Create TOML content based on reference
        toml_content = f"""# Common parameters (resolution, caption_extension, batch_size, num_repeats, enable_bucket, bucket_no_upscale) 
# can be set in either general or datasets sections
# Video-specific parameters (target_frames, frame_extraction, frame_stride, frame_sample, max_frames, source_fps)
# must be set in each datasets section

# caption_extension is not required for metadata jsonl file
# cache_directory is required for each dataset with metadata jsonl file

# general configurations
[general]
resolution = [960 , 960]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_jsonl_file = "{jsonl_file}"
frame_extraction = "full"
max_frames = 230
resolution = [298,298]
cache_directory = "{cache_dir}"


"""
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(toml_content)
        
        print(f"✓ Created: config/{config_name}")
    
    print()


def main():
    print("\n" + "=" * 60)
    print("Video Effects Processing Pipeline")
    print("=" * 60)
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Config Output: {CONFIG_OUTPUT_DIR}")
    print(f"Cache Base: {CACHE_BASE_DIR}")
    print()
    
    # Step 1: Unzip files
    unzip_files()
    
    # Step 2: Create video_captions.jsonl
    create_video_captions_jsonl()
    
    # Step 3: Create TOML configs
    create_toml_configs()
    
    print("=" * 60)
    print("✓ Processing complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run caching script: 02_cache_all.sh")
    print("2. Run training script: 03_train_all.sh")
    print()


if __name__ == "__main__":
    main()

