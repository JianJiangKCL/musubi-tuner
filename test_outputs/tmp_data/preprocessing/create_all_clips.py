#!/usr/bin/env python3
"""
Create all video clips from annotation file and video file (no duration filtering).

Usage:
    python create_all_clips.py --annotation_file /path/to/annotation.txt --video_file /path/to/video.mp4 --output_dir /path/to/output

This script parses annotation files with timestamps and creates ALL video clips with corresponding text files.
"""

import argparse
import os
import re
import subprocess
from pathlib import Path


def parse_timestamp(timestamp_str):
    """Convert timestamp string (MM'SS) to total seconds."""
    if "'" not in timestamp_str:
        return None

    parts = timestamp_str.split("'")
    if len(parts) != 2:
        return None

    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    except ValueError:
        return None


def parse_line(line):
    """Parse a single line from the annotation file."""
    line = line.strip()
    if not line:
        return None

    # Find all timestamp patterns in the line
    timestamp_pattern = r"(\d+'?\d+)"
    timestamps = re.findall(timestamp_pattern, line)

    if len(timestamps) == 0:
        # No timestamps - ignore
        return None

    # Extract action description (everything before the first timestamp)
    # Look for the first occurrence of a timestamp pattern
    first_timestamp_match = re.search(timestamp_pattern, line)
    if not first_timestamp_match:
        return None

    action_desc = line[:first_timestamp_match.start()].strip()

    # Remove leading spaces and special prefixes
    action_desc = re.sub(r'^(state|reason|constant)\s+', '', action_desc)

    if not action_desc:
        return None

    # Parse timestamp ranges
    time_ranges = []
    
    if len(timestamps) == 1:
        # Single timestamp - create 2-second clip starting from this timestamp
        start_time = parse_timestamp(timestamps[0])
        if start_time is not None:
            end_time = start_time + 2  # Add 2 seconds
            time_ranges.append((start_time, end_time))
    else:
        # Multiple timestamps - parse as pairs
        i = 0
        while i < len(timestamps) - 1:
            start_time = parse_timestamp(timestamps[i])
            end_time = parse_timestamp(timestamps[i + 1])

            if start_time is not None and end_time is not None and start_time < end_time:
                time_ranges.append((start_time, end_time))
                i += 2  # Move to next pair
            else:
                i += 1  # Skip invalid pair

    if not time_ranges:
        return None

    return {
        'action': action_desc,
        'time_ranges': time_ranges
    }


def create_video_clip(video_file, start_time, end_time, output_file):
    """Create a video clip using ffmpeg."""
    duration = end_time - start_time

    # Use full path to static ffmpeg binary
    ffmpeg_path = '/mnt/cfs/jj/musubi-tuner/ffmpeg-4.4.1-amd64-static/ffmpeg'

    cmd = [
        ffmpeg_path,
        '-i', str(video_file),
        '-ss', str(start_time),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-avoid_negative_ts', 'make_zero',
        '-y',  # Overwrite output files
        str(output_file)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"FFmpeg error: {e.stderr}"


def create_text_file(action_desc, start_time, end_time, output_file):
    """Create a text file with the action description and duration info."""
    duration = end_time - start_time
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{action_desc}\n")
            f.write(f"Duration: {duration:.1f}s\n")
            f.write(f"Time range: {start_time}s - {end_time}s\n")
        return True, None
    except Exception as e:
        return False, str(e)


def sanitize_filename(name):
    """Sanitize string to be used as filename."""
    # Replace invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def main():
    parser = argparse.ArgumentParser(description='Create ALL video clips from annotations (no filtering)')
    parser.add_argument('--annotation_file', required=True, help='Path to annotation text file')
    parser.add_argument('--video_file', required=True, help='Path to video file')
    parser.add_argument('--output_dir', default='/mnt/cfs/jj/musubi-tuner/tmp_data/clips_all', help='Output directory for clips and text files')
    parser.add_argument('--clip_prefix', default='clip', help='Prefix for clip filenames')

    args = parser.parse_args()

    # Validate input files
    annotation_file = Path(args.annotation_file)
    video_file = Path(args.video_file)
    output_dir = Path(args.output_dir)

    if not annotation_file.exists():
        print(f"Error: Annotation file {annotation_file} does not exist")
        return 1

    if not video_file.exists():
        print(f"Error: Video file {video_file} does not exist")
        return 1

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / 'clips'
    texts_dir = output_dir / 'texts'
    clips_dir.mkdir(exist_ok=True)
    texts_dir.mkdir(exist_ok=True)

    # Parse annotation file
    print(f"Parsing annotation file: {annotation_file}")
    actions = []

    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            parsed = parse_line(line)
            if parsed:
                parsed['line_number'] = line_num
                actions.append(parsed)

    print(f"Found {len(actions)} valid actions with time ranges")

    # Create clips and text files
    clip_count = 0
    total_ranges = sum(len(action['time_ranges']) for action in actions)

    print(f"Processing {total_ranges} time ranges...")
    print(f"Output will be saved to: {output_dir}")

    for action in actions:
        action_desc = action['action']
        sanitized_desc = sanitize_filename(action_desc)[:50]  # Limit length

        for range_idx, (start_time, end_time) in enumerate(action['time_ranges']):
            clip_count += 1
            duration = end_time - start_time

            # Create filename
            if len(action['time_ranges']) > 1:
                clip_name = f"{args.clip_prefix}_{clip_count:04d}_part{range_idx + 1}"
            else:
                clip_name = f"{args.clip_prefix}_{clip_count:04d}"

            clip_file = clips_dir / f"{clip_name}.mp4"
            text_file = texts_dir / f"{clip_name}.txt"

            print(f"Creating clip {clip_count}/{total_ranges}: {clip_name} (duration: {duration:.1f}s)")

            # Create video clip
            success, error = create_video_clip(video_file, start_time, end_time, clip_file)
            if not success:
                print(f"Warning: Failed to create video clip {clip_name}: {error}")
                continue

            # Create text file
            success, error = create_text_file(action_desc, start_time, end_time, text_file)
            if not success:
                print(f"Warning: Failed to create text file {clip_name}: {error}")

    print(f"\nCompleted! Created {clip_count} video clips and text files")
    print(f"Clips directory: {clips_dir}")
    print(f"Texts directory: {texts_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
