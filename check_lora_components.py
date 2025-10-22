#!/usr/bin/env python3
"""
Script to inspect LoRA components in safetensors files.
Checks if files contain both high and low noise model parts.
"""

import os
import sys
import argparse
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Add the src directory to the path to import musubi_tuner utilities
sys.path.insert(0, '/mnt/cfs/jj/musubi-tuner/src')

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


def get_lora_components(safetensors_path: str) -> Dict[str, List[str]]:
    """
    Extract LoRA components from a safetensors file.

    Returns:
        Dict with keys: 'down_keys', 'up_keys', 'alpha_keys', 'other_keys'
    """
    components = {
        'down_keys': [],
        'up_keys': [],
        'alpha_keys': [],
        'other_keys': [],
        'all_keys': []
    }

    try:
        with MemoryEfficientSafeOpen(safetensors_path) as f:
            keys = list(f.keys())
            components['all_keys'] = keys

            for key in keys:
                if key.endswith('.lora_down.weight'):
                    components['down_keys'].append(key)
                elif key.endswith('.lora_up.weight'):
                    components['up_keys'].append(key)
                elif key.endswith('.alpha'):
                    components['alpha_keys'].append(key)
                else:
                    components['other_keys'].append(key)

    except Exception as e:
        print(f"Error reading {safetensors_path}: {e}")
        return components

    return components


def extract_module_names(keys: List[str]) -> Set[str]:
    """
    Extract unique module names from LoRA keys.
    """
    module_names = set()

    for key in keys:
        # Remove LoRA-specific suffixes to get the base module name
        if key.endswith('.lora_down.weight'):
            module_name = key.replace('.lora_down.weight', '')
        elif key.endswith('.lora_up.weight'):
            module_name = key.replace('.lora_up.weight', '')
        elif key.endswith('.alpha'):
            module_name = key.replace('.alpha', '')
        else:
            continue

        # Clean up the module name (remove lora_unet_ prefix if present)
        if module_name.startswith('lora_unet_'):
            module_name = module_name.replace('lora_unet_', '', 1)

        module_names.add(module_name)

    return module_names


def categorize_noise_levels(components: Dict[str, List[str]], file_path: str) -> Dict[str, Set[str]]:
    """
    Categorize modules by noise level based on file path (not key names).
    In this project, noise level is indicated by directory/file path, not key names.
    """
    noise_categories = {
        'high_noise': set(),
        'low_noise': set(),
        'uncategorized': set()
    }

    # Determine noise level from file path
    file_path_lower = file_path.lower()

    if 'high_noise' in file_path_lower or 'trace20' in file_path_lower:
        # All modules in this file are high noise
        module_names = extract_module_names(components['down_keys'] + components['up_keys'] + components['alpha_keys'])
        noise_categories['high_noise'] = module_names
    elif 'low_noise' in file_path_lower or 'trace30' in file_path_lower:
        # All modules in this file are low noise
        module_names = extract_module_names(components['down_keys'] + components['up_keys'] + components['alpha_keys'])
        noise_categories['low_noise'] = module_names
    else:
        # Cannot determine noise level from path
        module_names = extract_module_names(components['down_keys'] + components['up_keys'] + components['alpha_keys'])
        noise_categories['uncategorized'] = module_names

    return noise_categories


def analyze_lora_file(file_path: str) -> Dict:
    """
    Comprehensive analysis of a LoRA safetensors file.
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(file_path)}")
    print(f"Full path: {file_path}")
    print(f"{'='*60}")

    components = get_lora_components(file_path)

    if not components['all_keys']:
        return {"error": "No keys found in file"}

    analysis = {
        'file_path': file_path,
        'total_keys': len(components['all_keys']),
        'lora_components': {
            'down_weights': len(components['down_keys']),
            'up_weights': len(components['up_keys']),
            'alphas': len(components['alpha_keys'])
        },
        'other_components': len(components['other_keys']),
        'noise_analysis': categorize_noise_levels(components, file_path)
    }

    # Print summary
    print(f"Total keys: {analysis['total_keys']}")
    print(f"LoRA components:")
    print(f"  - Down weights: {analysis['lora_components']['down_weights']}")
    print(f"  - Up weights: {analysis['lora_components']['up_weights']}")
    print(f"  - Alpha values: {analysis['lora_components']['alphas']}")
    print(f"Other components: {analysis['other_components']}")

    # Noise level analysis
    noise = analysis['noise_analysis']
    print("\nNoise level analysis:")
    print(f"  - High noise modules: {len(noise['high_noise'])}")
    print(f"  - Low noise modules: {len(noise['low_noise'])}")
    print(f"  - Uncategorized modules: {len(noise['uncategorized'])}")

    if noise['high_noise']:
        print("\nHigh noise modules (sample):")
        sample_high = sorted(list(noise['high_noise']))[:5]
        for module in sample_high:
            print(f"    - {module}")
        if len(noise['high_noise']) > 5:
            print(f"    ... and {len(noise['high_noise']) - 5} more")

    if noise['low_noise']:
        print("\nLow noise modules (sample):")
        sample_low = sorted(list(noise['low_noise']))[:5]
        for module in sample_low:
            print(f"    - {module}")
        if len(noise['low_noise']) > 5:
            print(f"    ... and {len(noise['low_noise']) - 5} more")

    return analysis


def compare_files(file1_analysis: Dict, file2_analysis: Dict) -> Dict:
    """
    Compare two LoRA files and check for mixed components.
    """
    print(f"\n{'='*60}")
    print("COMPARISON ANALYSIS")
    print(f"{'='*60}")

    comparison = {
        'file1_has_high_noise': len(file1_analysis['noise_analysis']['high_noise']) > 0,
        'file1_has_low_noise': len(file1_analysis['noise_analysis']['low_noise']) > 0,
        'file2_has_high_noise': len(file2_analysis['noise_analysis']['high_noise']) > 0,
        'file2_has_low_noise': len(file2_analysis['noise_analysis']['low_noise']) > 0,
        'mixed_components': {
            'file1': len(file1_analysis['noise_analysis']['high_noise']) > 0 and len(file1_analysis['noise_analysis']['low_noise']) > 0,
            'file2': len(file2_analysis['noise_analysis']['high_noise']) > 0 and len(file2_analysis['noise_analysis']['low_noise']) > 0
        }
    }

    print("File 1 analysis:")
    print(f"  - Contains high noise components: {comparison['file1_has_high_noise']}")
    print(f"  - Contains low noise components: {comparison['file1_has_low_noise']}")
    print(f"  - Has mixed components: {comparison['mixed_components']['file1']}")

    print("\nFile 2 analysis:")
    print(f"  - Contains high noise components: {comparison['file2_has_high_noise']}")
    print(f"  - Contains low noise components: {comparison['file2_has_low_noise']}")
    print(f"  - Has mixed components: {comparison['mixed_components']['file2']}")

    # Check for overlap in modules
    file1_modules = file1_analysis['noise_analysis']['high_noise'] | file1_analysis['noise_analysis']['low_noise'] | file1_analysis['noise_analysis']['uncategorized']
    file2_modules = file2_analysis['noise_analysis']['high_noise'] | file2_analysis['noise_analysis']['low_noise'] | file2_analysis['noise_analysis']['uncategorized']

    overlap = file1_modules & file2_modules
    file1_only = file1_modules - file2_modules
    file2_only = file2_modules - file1_modules

    comparison['module_overlap'] = len(overlap)
    comparison['file1_unique_modules'] = len(file1_only)
    comparison['file2_unique_modules'] = len(file2_only)

    print("\nModule comparison:")
    print(f"  - Overlapping modules: {len(overlap)}")
    print(f"  - File 1 unique modules: {len(file1_only)}")
    print(f"  - File 2 unique modules: {len(file2_only)}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description='Inspect LoRA components in safetensors files')
    parser.add_argument('--files', nargs='+', help='Paths to safetensors files to analyze')
    parser.add_argument('--high-noise', default='/mnt/cfs/jj/musubi-tuner/peel_it_outputs_high_noise_trace20/peel_it-of-lorac.safetensors',
                        help='Path to high noise LoRA file')
    parser.add_argument('--low-noise', default='/mnt/cfs/jj/musubi-tuner/peel_it_outputs_low_noise_trace30/peel_it-of-lorac.safetensors',
                        help='Path to low noise LoRA file')

    args = parser.parse_args()

    files_to_analyze = []
    if args.files:
        files_to_analyze = args.files
    else:
        files_to_analyze = [args.high_noise, args.low_noise]

    analyses = []
    for file_path in files_to_analyze:
        analysis = analyze_lora_file(file_path)
        if 'error' not in analysis:
            analyses.append(analysis)

    if len(analyses) == 2:
        comparison = compare_files(analyses[0], analyses[1])

        # Final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")

        if comparison['mixed_components']['file1']:
            print("⚠️  WARNING: File 1 contains both high and low noise components!")
        else:
            print("✅ File 1 appears to be dedicated to a single noise level")

        if comparison['mixed_components']['file2']:
            print("⚠️  WARNING: File 2 contains both high and low noise components!")
        else:
            print("✅ File 2 appears to be dedicated to a single noise level")

        if comparison['module_overlap'] > 0:
            print(f"📊 Files share {comparison['module_overlap']} common modules")
        else:
            print("📊 Files have no overlapping modules")


if __name__ == '__main__':
    main()
