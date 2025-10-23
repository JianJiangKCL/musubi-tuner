#!/usr/bin/env python3
"""
Process JSONL file to trim videos based on [Xs] markers in the data.
For entries with [Xs] marker, create a new video clip of X seconds from the start,
and update the video_path and duration accordingly.
"""

import json
import re
import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_duration_marker(line):
    """
    Extract duration marker from the end of a JSON line.
    
    Args:
        line: JSON string that may end with [Xs] marker
        
    Returns:
        tuple: (json_data_dict, duration_in_seconds or None)
    """
    # Check if line ends with [Xs] pattern
    match = re.search(r'\[(\d+)s\]\s*$', line)
    
    if match:
        duration = int(match.group(1))
        # Remove the marker from the line to parse JSON
        json_str = line[:match.start()].strip()
        data = json.loads(json_str)
        return data, duration
    else:
        # No marker, parse as-is
        data = json.loads(line.strip())
        return data, None


def trim_video(input_path, output_path, duration, ffmpeg_path='ffmpeg'):
    """
    Trim video from start to specified duration using ffmpeg.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        duration: Duration in seconds
        ffmpeg_path: Path to ffmpeg executable
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(input_path):
        print(f"Warning: Input video not found: {input_path}")
        return False
    
    # FFmpeg command to trim video from start
    cmd = [
        ffmpeg_path,
        '-i', input_path,
        '-t', str(duration),  # Duration from start
        '-c', 'copy',  # Copy codec (fast, no re-encoding)
        '-y',  # Overwrite output file if exists
        output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error trimming video {input_path}: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        return False


def process_jsonl(input_file, output_file=None, dry_run=False, ffmpeg_path='ffmpeg'):
    """
    Process JSONL file, trimming videos where needed.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file (default: same dir as input with _processed suffix)
        dry_run: If True, don't actually trim videos, just show what would be done
        ffmpeg_path: Path to ffmpeg executable
    """
    input_path = Path(input_file)
    
    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
    else:
        output_file = Path(output_file)
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    # Read all lines first to count them
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    
    processed_entries = []
    videos_to_trim = 0
    videos_trimmed = 0
    videos_failed = 0
    
    # First pass: count videos to trim
    for line in lines:
        _, duration = parse_duration_marker(line)
        if duration is not None:
            videos_to_trim += 1
    
    print(f"\nFound {len(lines)} entries, {videos_to_trim} need video trimming")
    
    if dry_run:
        print("\n=== DRY RUN MODE ===")
    
    # Process each line
    for line in tqdm(lines, desc="Processing entries"):
        data, target_duration = parse_duration_marker(line)
        
        if target_duration is None:
            # No marker, keep as-is
            processed_entries.append(data)
        else:
            # Need to trim video
            original_video_path = data['video_path']
            video_path = Path(original_video_path)
            
            # Create new filename: original_name_Xs.mp4
            new_filename = f"{video_path.stem}_{target_duration}s{video_path.suffix}"
            new_video_path = video_path.parent / new_filename
            
            if dry_run:
                print(f"\nWould trim: {video_path.name}")
                print(f"  From: {data['duration']:.2f}s -> To: {target_duration}s")
                print(f"  Output: {new_video_path.name}")
                data['video_path'] = str(new_video_path)
                data['duration'] = target_duration
                processed_entries.append(data)
                videos_trimmed += 1
            else:
                # Trim the video
                success = trim_video(
                    str(video_path),
                    str(new_video_path),
                    target_duration,
                    ffmpeg_path
                )
                
                if success:
                    # Update the entry
                    data['video_path'] = str(new_video_path)
                    data['duration'] = target_duration
                    processed_entries.append(data)
                    videos_trimmed += 1
                else:
                    # Keep original entry if trimming failed
                    print(f"\nWarning: Failed to trim {video_path.name}, keeping original entry")
                    processed_entries.append(data)
                    videos_failed += 1
    
    # Write output JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n=== Summary ===")
    print(f"Total entries: {len(lines)}")
    print(f"Entries needing trimming: {videos_to_trim}")
    if not dry_run:
        print(f"Successfully trimmed: {videos_trimmed}")
        if videos_failed > 0:
            print(f"Failed to trim: {videos_failed}")
    print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Process JSONL file to trim videos based on [Xs] markers'
    )
    parser.add_argument(
        'input_file',
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Path to output JSONL file (default: input_file_processed.jsonl)',
        default=None
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually trimming videos'
    )
    parser.add_argument(
        '--ffmpeg',
        help='Path to ffmpeg executable',
        default='/mnt/cfs/jj/musubi-tuner/ffmpeg-4.4.1-amd64-static/ffmpeg'
    )
    
    args = parser.parse_args()
    
    process_jsonl(args.input_file, args.output, args.dry_run, args.ffmpeg)


if __name__ == '__main__':
    main()

