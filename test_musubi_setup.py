#!/usr/bin/env python3
"""Test script to verify Musubi setup and diagnose issues."""

import os
import sys
import subprocess
from pathlib import Path
import json

def test_conda_env():
    """Test if conda environment is accessible."""
    print("🔍 Testing conda environment 'musu'...")
    try:
        result = subprocess.run(['conda', 'run', '-n', 'musu', 'python', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Conda environment 'musu' is accessible: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ Failed to access conda environment: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error testing conda environment: {e}")
        return False

def test_model_files():
    """Test if all required model files exist."""
    print("\n🔍 Checking model files...")
    models = {
        "T5 Model": "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
        "VAE Model": "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/Wan2.1_VAE.pth",
        "DiT Model": "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
        "DiT High Noise": "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors"
    }
    
    all_exist = True
    for name, path in models.items():
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"✓ {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {name}: {path} NOT FOUND")
            all_exist = False
    
    return all_exist

def test_dataset_structure():
    """Test dataset structure and video files."""
    print("\n🔍 Checking dataset structure...")
    dataset_dir = Path("/mnt/cfs/jj/musubi-tuner/datasets/to_improve")
    
    if not dataset_dir.exists():
        print(f"✗ Dataset directory not found: {dataset_dir}")
        return False
    
    jsonl_files = list(dataset_dir.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files:")
    
    all_videos_exist = True
    for jsonl_file in jsonl_files:
        print(f"\n  📄 {jsonl_file.name}:")
        
        # Read first few lines to check video paths
        with open(jsonl_file, 'r') as f:
            lines = f.readlines()[:3]  # Check first 3 entries
            
        for i, line in enumerate(lines):
            data = json.loads(line.strip())
            video_path = Path(data.get('video_path', ''))
            caption = data.get('caption', '')[:50] + "..."
            
            if video_path.exists():
                size_mb = video_path.stat().st_size / (1024 * 1024)
                print(f"    ✓ Video {i+1}: {video_path.name} ({size_mb:.1f} MB)")
            else:
                print(f"    ✗ Video {i+1}: {video_path} NOT FOUND")
                all_videos_exist = False
    
    return all_videos_exist

def test_script_files():
    """Test if required scripts exist."""
    print("\n🔍 Checking required scripts...")
    scripts = {
        "Text Encoder Cache": "/mnt/cfs/jj/musubi-tuner/wan_cache_text_encoder_outputs.py",
        "Latents Cache": "/mnt/cfs/jj/musubi-tuner/wan_cache_latents.py",
        "Training Script": "/mnt/cfs/jj/musubi-tuner/src/musubi_tuner/wan_train_network.py",
        "Accelerate Config": "/mnt/cfs/jj/musubi-tuner/single_gpu_config.yaml"
    }
    
    all_exist = True
    for name, path in scripts.items():
        if Path(path).exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} NOT FOUND")
            all_exist = False
    
    return all_exist

def test_gpu_availability():
    """Test GPU availability."""
    print("\n🔍 Checking GPU availability...")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.free', 
                               '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Available GPUs:")
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                gpu_id = parts[0]
                gpu_name = parts[1]
                memory_free = parts[2]
                print(f"  GPU {gpu_id}: {gpu_name} ({memory_free} free)")
            return True
        else:
            print("✗ Failed to query GPUs")
            return False
    except Exception as e:
        print(f"✗ Error checking GPUs: {e}")
        return False

def test_sample_toml_creation():
    """Test creating a sample TOML file."""
    print("\n🔍 Testing TOML creation...")
    import toml
    
    test_config = {
        "general": {
            "resolution": [960, 960],
            "batch_size": 1,
            "enable_bucket": True,
            "bucket_no_upscale": False
        },
        "datasets": [{
            "video_jsonl_file": "/mnt/cfs/jj/musubi-tuner/datasets/to_improve/Eating_zoom.jsonl",
            "frame_extraction": "full",
            "max_frames": 230,
            "resolution": [298, 298],
            "cache_directory": "/mnt/cfs/jj/musubi-tuner/cache/test_cache"
        }]
    }
    
    try:
        toml_str = toml.dumps(test_config)
        print("✓ TOML creation successful")
        print("Sample TOML content:")
        print(toml_str[:200] + "...")
        return True
    except Exception as e:
        print(f"✗ TOML creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Musubi Tuner Setup Test ===\n")
    
    tests = [
        ("Conda Environment", test_conda_env),
        ("Model Files", test_model_files),
        ("Dataset Structure", test_dataset_structure),
        ("Script Files", test_script_files),
        ("GPU Availability", test_gpu_availability),
        ("TOML Creation", test_sample_toml_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        success = test_func()
        results.append((test_name, success))
    
    print(f"\n{'='*50}")
    print("=== Test Summary ===")
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n✅ All tests passed! The setup appears to be correct.")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
