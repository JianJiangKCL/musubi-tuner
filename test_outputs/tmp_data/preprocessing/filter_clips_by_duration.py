#!/usr/bin/env python3
"""
Filter video clips by duration and create JSONL file.

Usage:
    python filter_clips_by_duration.py --clips_dir /path/to/clips --texts_dir /path/to/texts --max_duration 10.0 --output_jsonl output.jsonl

This script reads existing video clips and their text files, filters by duration, and creates a JSONL file.
"""

import argparse
import json
import subprocess
from pathlib import Path


def get_video_duration(video_file):
    """Get video duration using ffprobe."""
    ffprobe_path = '/mnt/cfs/jj/musubi-tuner/ffmpeg-4.4.1-amd64-static/ffprobe'
    
    cmd = [
        ffprobe_path,
        '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0',
        str(video_file)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def read_text_caption(text_file):
    """Read caption from text file (first line only)."""
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                return lines[0].strip()  # First line is the caption
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Filter video clips by duration and create JSONL')
    parser.add_argument('--clips_dir', required=True, help='Directory containing video clips')
    parser.add_argument('--texts_dir', required=True, help='Directory containing text files')
    parser.add_argument('--max_duration', type=float, required=True, help='Maximum duration in seconds')
    parser.add_argument('--output_jsonl', required=True, help='Output JSONL file path')
    parser.add_argument('--min_duration', type=float, default=0.0, help='Minimum duration in seconds (default: 0.0)')

    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    texts_dir = Path(args.texts_dir)
    output_jsonl = Path(args.output_jsonl)

    if not clips_dir.exists():
        print(f"Error: Clips directory {clips_dir} does not exist")
        return 1

    if not texts_dir.exists():
        print(f"Error: Texts directory {texts_dir} does not exist")
        return 1

    # Create output directory if needed
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Find all video clips
    video_files = list(clips_dir.glob("*.mp4"))
    video_files.sort()

    print(f"Found {len(video_files)} video clips")
    print(f"Filtering clips with duration: {args.min_duration}s ≤ duration ≤ {args.max_duration}s")

    filtered_clips = []
    skipped_count = 0

    with open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
        for video_file in video_files:
            # Get corresponding text file
            text_file = texts_dir / f"{video_file.stem}.txt"
            
            if not text_file.exists():
                print(f"Warning: Text file not found for {video_file.name}")
                continue

            # Get video duration
            duration = get_video_duration(video_file)
            if duration is None:
                print(f"Warning: Could not get duration for {video_file.name}")
                continue

            # Check duration filter
            if duration < args.min_duration or duration > args.max_duration:
                skipped_count += 1
                print(f"Skipping {video_file.name}: duration {duration:.1f}s not in range [{args.min_duration}, {args.max_duration}]")
                continue

            # Read caption
            caption = read_text_caption(text_file)
            if caption is None:
                print(f"Warning: Could not read caption for {video_file.name}")
                continue

            # Add to filtered list
            json_entry = {
                "video_path": str(video_file.absolute()),
                "caption": caption
            }
            
            jsonl_file.write(json.dumps(json_entry, ensure_ascii=False) + '\n')
            filtered_clips.append(video_file.name)
            
            print(f"✓ {video_file.name}: {duration:.1f}s - {caption[:50]}...")

    print(f"\nCompleted!")
    print(f"Total clips processed: {len(video_files)}")
    print(f"Clips matching duration filter: {len(filtered_clips)}")
    print(f"Clips skipped: {skipped_count}")
    print(f"JSONL file created: {output_jsonl}")

    return 0


if __name__ == '__main__':
    exit(main())
