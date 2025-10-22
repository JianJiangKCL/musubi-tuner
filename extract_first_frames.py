#!/usr/bin/env python3
"""
Script to extract first frames from videos listed in a JSONL file.

Usage:
    python extract_first_frames.py <path_to_jsonl_file>

Example:
    python extract_first_frames.py /path/to/filtered_clips_processed.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm


def extract_first_frame(video_path, output_path):
    """
    Extract the first frame from a video using ffmpeg.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the extracted frame
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use ffmpeg to extract the first frame
        # -i: input file
        # -vframes 1: extract only 1 frame
        # -q:v 2: high quality (scale 2-5, lower is better)
        # -y: overwrite output file if it exists
        ffmpeg_path = '/mnt/cfs/jj/musubi-tuner/ffmpeg-4.4.1-amd64-static/ffmpeg'
        cmd = [
            ffmpeg_path,
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '2',
            '-y',
            output_path
        ]
        
        # Run ffmpeg with suppressed output
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frame from {video_path}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error processing {video_path}: {e}", file=sys.stderr)
        return False


def process_jsonl(jsonl_path):
    """
    Process the JSONL file and extract first frames from all videos.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        tuple: (successful_count, failed_count)
    """
    jsonl_path = Path(jsonl_path)
    
    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}", file=sys.stderr)
        return 0, 0
    
    # Read all entries from the JSONL file
    entries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)
                continue
    
    print(f"Found {len(entries)} video entries in the JSONL file")
    
    # Process each video
    successful = 0
    failed = 0
    
    for entry in tqdm(entries, desc="Extracting first frames", unit="video"):
        video_path = entry.get('video_path')
        
        if not video_path:
            print(f"Warning: Entry missing 'video_path': {entry}", file=sys.stderr)
            failed += 1
            continue
        
        video_path = Path(video_path)
        
        # Check if video file exists
        if not video_path.exists():
            print(f"Warning: Video file not found: {video_path}", file=sys.stderr)
            failed += 1
            continue
        
        # Determine output path
        # Replace 'clips' folder with 'first_frames' and change extension to .jpg
        # Example: .../cholec02/clips/video.mp4 -> .../cholec02/first_frames/video.jpg
        parts = video_path.parts
        
        # Find the index of 'clips' in the path
        try:
            clips_index = parts.index('clips')
            # Reconstruct the path with 'first_frames' instead of 'clips'
            output_parts = list(parts[:clips_index]) + ['first_frames'] + list(parts[clips_index + 1:])
            output_path = Path(*output_parts)
            # Change extension to .jpg
            output_path = output_path.with_suffix('.jpg')
        except ValueError:
            # If 'clips' not in path, just save in a 'first_frames' folder next to the video
            output_path = video_path.parent.parent / 'first_frames' / video_path.with_suffix('.jpg').name
        
        # Extract the first frame
        if extract_first_frame(str(video_path), str(output_path)):
            successful += 1
        else:
            failed += 1
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description='Extract first frames from videos listed in a JSONL file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s /path/to/filtered_clips_processed.jsonl
        """
    )
    parser.add_argument(
        'jsonl_file',
        type=str,
        help='Path to the JSONL file containing video paths'
    )
    
    args = parser.parse_args()
    
    # Process the JSONL file
    successful, failed = process_jsonl(args.jsonl_file)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"Successfully extracted: {successful} frames")
    print(f"Failed: {failed} frames")
    print(f"Total: {successful + failed} videos")
    print("="*50)
    
    # Exit with error code if any failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()

