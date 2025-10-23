#!/usr/bin/env python3
"""
Script to find unique captions in filtered_clips.jsonl
"""

import json
from pathlib import Path
from collections import Counter


def find_unique_captions(jsonl_path):
    """
    Read JSONL file and find unique captions
    
    Args:
        jsonl_path: Path to the JSONL file
    
    Returns:
        dict: Dictionary with caption statistics
    """
    captions = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                data = json.loads(line)
                if 'caption' in data:
                    captions.append(data['caption'])
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
                continue
    
    # Get unique captions and their counts
    caption_counts = Counter(captions)
    
    return caption_counts, len(captions)


def main():
    # Path to the JSONL file
    jsonl_path = Path(__file__).parent / "filtered_clips.jsonl"
    
    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}")
        return
    
    print(f"Reading captions from: {jsonl_path}")
    print("-" * 80)
    
    caption_counts, total_captions = find_unique_captions(jsonl_path)
    
    # Print statistics
    print(f"\nTotal captions: {total_captions}")
    print(f"Unique captions: {len(caption_counts)}")
    print("-" * 80)
    
    # Print unique captions sorted by frequency (most common first)
    print("\nUnique captions (sorted by frequency):")
    print("-" * 80)
    for idx, (caption, count) in enumerate(caption_counts.most_common(), 1):
        print(f"{idx:3d}. [{count:3d}x] {caption}")
    
    # Save results to a file
    output_path = Path(__file__).parent / "unique_captions_report.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Unique Captions Report\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"Source file: {jsonl_path}\n")
        f.write(f"Total captions: {total_captions}\n")
        f.write(f"Unique captions: {len(caption_counts)}\n")
        f.write(f"=" * 80 + "\n\n")
        
        f.write("Captions sorted by frequency (most common first):\n")
        f.write("-" * 80 + "\n")
        for idx, (caption, count) in enumerate(caption_counts.most_common(), 1):
            f.write(f"{idx:3d}. [{count:3d}x] {caption}\n")
    
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()

