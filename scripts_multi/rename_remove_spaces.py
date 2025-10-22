#!/usr/bin/env python3
"""
Remove spaces from folder names and update all configuration files
"""

import os
import json
import shutil
from pathlib import Path

INPUT_DIR = "/mnt/cfs/jj/musubi-tuner/2025_10_21"

def get_new_name(old_name):
    """Convert 'Cotton Cloud' to 'cotton_cloud'"""
    return old_name.replace(' ', '_').lower()

def rename_folders():
    """Rename all folders with spaces"""
    print("=" * 60)
    print("Step 1: Renaming folders (removing spaces)")
    print("=" * 60)
    
    renames = []
    
    # Find all directories with spaces (excluding cache and config)
    for item in os.listdir(INPUT_DIR):
        item_path = os.path.join(INPUT_DIR, item)
        
        if not os.path.isdir(item_path):
            continue
        
        if item in ['cache', 'config']:
            continue
            
        if ' ' in item:
            new_name = item.replace(' ', '_')
            new_path = os.path.join(INPUT_DIR, new_name)
            
            print(f"Renaming: '{item}' -> '{new_name}'")
            
            # Rename the directory
            if os.path.exists(new_path):
                print(f"  Warning: '{new_name}' already exists, removing it first")
                shutil.rmtree(new_path)
            
            shutil.move(item_path, new_path)
            renames.append((item, new_name))
            print(f"  ✓ Renamed")
    
    print()
    return renames

def update_jsonl_files(renames):
    """Update video paths in jsonl files"""
    print("=" * 60)
    print("Step 2: Updating video_captions.jsonl files")
    print("=" * 60)
    
    for old_name, new_name in renames:
        jsonl_path = os.path.join(INPUT_DIR, new_name, "dataset", "video_captions.jsonl")
        
        if not os.path.exists(jsonl_path):
            print(f"  Warning: {jsonl_path} not found")
            continue
        
        print(f"Updating: {new_name}/dataset/video_captions.jsonl")
        
        # Read and update jsonl file
        updated_lines = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                # Update the video path
                entry['video_path'] = entry['video_path'].replace(f"/{old_name}/", f"/{new_name}/")
                updated_lines.append(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Write back
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        
        print(f"  ✓ Updated {len(updated_lines)} entries")
    
    print()

def update_toml_files(renames):
    """Update paths in TOML config files"""
    print("=" * 60)
    print("Step 3: Updating TOML config files")
    print("=" * 60)
    
    config_dir = os.path.join(INPUT_DIR, "config")
    
    if not os.path.exists(config_dir):
        print("  Warning: config directory not found")
        return
    
    for old_name, new_name in renames:
        old_safe_name = old_name.lower().replace(' ', '_').replace('-', '_')
        new_safe_name = new_name.lower().replace(' ', '_').replace('-', '_')
        
        # TOML file should already have the safe name
        toml_file = os.path.join(config_dir, f"{old_safe_name}.toml")
        
        if not os.path.exists(toml_file):
            print(f"  Warning: {toml_file} not found")
            continue
        
        print(f"Updating: config/{old_safe_name}.toml")
        
        # Read and update TOML file
        with open(toml_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update paths
        content = content.replace(f'"{INPUT_DIR}/{old_name}/', f'"{INPUT_DIR}/{new_name}/')
        content = content.replace(f'"/mnt/cfs/jj/musubi-tuner/2025_10_21/cache/{old_safe_name}"', 
                                 f'"/mnt/cfs/jj/musubi-tuner/2025_10_21/cache/{new_safe_name}"')
        
        # Write back
        with open(toml_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✓ Updated")
    
    print()

def rename_cache_directories(renames):
    """Rename cache directories to match new names"""
    print("=" * 60)
    print("Step 4: Renaming cache directories")
    print("=" * 60)
    
    cache_dir = os.path.join(INPUT_DIR, "cache")
    
    if not os.path.exists(cache_dir):
        print("  No cache directory found (OK if not cached yet)")
        print()
        return
    
    for old_name, new_name in renames:
        old_safe_name = old_name.lower().replace(' ', '_').replace('-', '_')
        new_safe_name = new_name.lower().replace(' ', '_').replace('-', '_')
        
        old_cache_path = os.path.join(cache_dir, old_safe_name)
        new_cache_path = os.path.join(cache_dir, new_safe_name)
        
        if os.path.exists(old_cache_path):
            # Skip if old and new are the same
            if old_safe_name == new_safe_name:
                print(f"Cache {old_safe_name} already has correct name, skipping")
                continue
            
            print(f"Renaming cache: {old_safe_name} -> {new_safe_name}")
            if os.path.exists(new_cache_path):
                print(f"  Warning: {new_safe_name} already exists, removing it")
                shutil.rmtree(new_cache_path)
            shutil.move(old_cache_path, new_cache_path)
            print(f"  ✓ Renamed")
    
    print()

def main():
    print("\n" + "=" * 60)
    print("Remove Spaces from Folder Names")
    print("=" * 60)
    print(f"Working Directory: {INPUT_DIR}")
    print()
    
    # Step 1: Rename folders
    renames = rename_folders()
    
    if not renames:
        print("✓ No folders with spaces found!")
        return
    
    # Step 2: Update jsonl files
    update_jsonl_files(renames)
    
    # Step 3: Update TOML files
    update_toml_files(renames)
    
    # Step 4: Rename cache directories
    rename_cache_directories(renames)
    
    print("=" * 60)
    print("✓ All spaces removed successfully!")
    print("=" * 60)
    print()
    print("Renamed folders:")
    for old_name, new_name in renames:
        print(f"  '{old_name}' -> '{new_name}'")
    print()
    print("Next steps:")
    print("1. Run caching: bash scripts_multi/02_cache_all.sh")
    print("2. Run training: bash scripts_multi/03_train_all.sh")
    print()

if __name__ == "__main__":
    main()

