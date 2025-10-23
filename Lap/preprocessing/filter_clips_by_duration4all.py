#!/usr/bin/env python3
"""
Filter video clips by duration for all videos and create JSONL files.

Usage:
    python filter_clips_by_duration4all.py [base_dir] --min_duration 2.0 --max_duration 10.0 --output_jsonl filtered_clips.jsonl

This script reads all video clips from subdirectories, filters by duration, and creates a JSONL file.
"""

import argparse
import json
import re
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


def should_skip_caption(caption):
    """Check if caption should be skipped based on content filters.
    
    Returns True if the caption should be skipped, False otherwise.
    
    Filters:
    - CVS related content (cvs, CVS)
    - Surgery end markers (手术结束, 腔镜移出体外)
    - Temporary actions (暂时)
    """
    if not caption:
        return False
    
    caption_lower = caption.lower()
    
    # Filter CVS related content
    if 'cvs' in caption_lower:
        return True
    
    # Filter surgery end markers
    if '手术结束' in caption:
        return True
    
    # Filter scope removal indicating end (but allow cleaning)
    if '腔镜移出体外' in caption and '手术结束' in caption:
        return True
    
    # Filter temporary actions
    if '暂时' in caption:
        return True
    
    return False


def clean_caption(caption):
    """Clean caption text by removing tool labels and bracket content.
    
    - Removes [Tools in: ...; Tools observed: ...] content
    - Replaces 抓钳A/B/C/0/1/2 with 抓钳
    - Replaces 戳卡a/b/c/d with 戳卡
    - Removes any other content in square brackets
    """
    if not caption:
        return caption
    
    # Remove content in square brackets (e.g., [Tools in: ...; Tools observed: ...])
    caption = re.sub(r'\s*\[.*?\]\s*', '', caption)
    
    # Replace 抓钳A/B/C/D/0/1/2/etc with 抓钳
    # Handle patterns like "抓钳A、抓钳B、抓钳C" -> "抓钳"
    caption = re.sub(r'抓钳[A-Za-z0-9](?:、抓钳[A-Za-z0-9])+', '抓钳', caption)
    # Handle abbreviated patterns like "抓钳A、B、C" -> "抓钳"
    caption = re.sub(r'抓钳[A-Za-z0-9](?:、[A-Za-z0-9])+', '抓钳', caption)
    # Handle remaining single instances
    caption = re.sub(r'抓钳[A-Za-z0-9]', '抓钳', caption)
    
    # Replace 戳卡a/b/c/d/A/B/C/D/0/1/2/etc with 戳卡
    # Handle patterns like "戳卡a、戳卡b、戳卡c" -> "戳卡"
    caption = re.sub(r'戳卡[A-Za-z0-9](?:、戳卡[A-Za-z0-9])+', '戳卡', caption)
    # Handle abbreviated patterns like "戳卡a、b、c" -> "戳卡"
    caption = re.sub(r'戳卡[A-Za-z0-9](?:、[A-Za-z0-9])+', '戳卡', caption)
    # Handle remaining single instances
    caption = re.sub(r'戳卡[A-Za-z0-9]', '戳卡', caption)
    
    # Also handle other tool patterns (电凝钩, 双极电凝, etc. with labels)
    caption = re.sub(r'电凝钩[A-Za-z0-9]', '电凝钩', caption)
    caption = re.sub(r'双极电凝[A-Za-z0-9]', '双极电凝', caption)
    
    # Remove extra spaces
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    return caption


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


def process_video_directory(video_dir, min_duration, max_duration):
    """Process a single video directory and return filtered clips."""
    clips_dir = video_dir / 'clips'
    texts_dir = video_dir / 'texts'
    
    if not clips_dir.exists() or not texts_dir.exists():
        print(f"Warning: Skipping {video_dir.name} - clips or texts directory not found")
        return [], 0
    
    # Find all video clips
    video_files = list(clips_dir.glob("*.mp4"))
    video_files.sort()
    
    print(f"\n{'='*80}")
    print(f"Processing: {video_dir.name}")
    print(f"{'='*80}")
    print(f"Found {len(video_files)} video clips")
    
    filtered_clips = []
    skipped_count = 0
    
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
        if duration < min_duration or duration > max_duration:
            skipped_count += 1
            continue
        
        # Read caption
        caption = read_text_caption(text_file)
        if caption is None:
            print(f"Warning: Could not read caption for {video_file.name}")
            continue
        
        # Check if caption should be skipped (CVS, surgery end, etc.)
        if should_skip_caption(caption):
            skipped_count += 1
            continue
        
        # Clean caption (remove tool labels and bracket content)
        caption = clean_caption(caption)
        
        if not caption:
            print(f"Warning: Empty caption after cleaning for {video_file.name}")
            continue
        
        # Add to filtered list
        json_entry = {
            "video_path": str(video_file.absolute()),
            "caption": caption,
            "duration": round(duration, 2),
            "source": video_dir.name
        }
        
        filtered_clips.append(json_entry)
        
        if len(filtered_clips) % 50 == 0:
            print(f"  Processed {len(filtered_clips)} clips so far...")
    
    print(f"Total clips: {len(video_files)}")
    print(f"Clips matching duration filter [{min_duration}s - {max_duration}s]: {len(filtered_clips)}")
    print(f"Clips skipped: {skipped_count}")
    
    return filtered_clips, len(video_files)


def main():
    parser = argparse.ArgumentParser(
        description='Filter video clips by duration for all videos and create JSONL'
    )
    parser.add_argument(
        'base_dir',
        nargs='?',
        default='/mnt/cfs/jj/musubi-tuner/test_outputs/tmp_data/clips_all_video',
        help='Base directory containing video subdirectories (default: /mnt/cfs/jj/musubi-tuner/tmp_data/clips_all_video)'
    )
    parser.add_argument(
        '--min_duration',
        type=float,
        default=2.0,
        help='Minimum duration in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        default=20.0,
        help='Maximum duration in seconds (default: 10.0)'
    )
    parser.add_argument(
        '--output_jsonl',
        default='filtered_clips.jsonl',
        help='Output JSONL file path (default: filtered_clips.jsonl)'
    )
    parser.add_argument(
        '--separate_files',
        action='store_true',
        help='Create separate JSONL files for each video (e.g., filtered_cholec02.jsonl)'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist")
        return 1
    
    # Find all subdirectories with clips
    video_dirs = [d for d in base_dir.iterdir() if d.is_dir() and (d / 'clips').exists()]
    video_dirs.sort()
    
    if not video_dirs:
        print(f"Error: No video directories found in {base_dir}")
        return 1
    
    print(f"{'='*80}")
    print(f"VIDEO CLIP FILTERING")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    print(f"Video directories found: {len(video_dirs)}")
    print(f"Duration filter: {args.min_duration}s ≤ duration ≤ {args.max_duration}s")
    print(f"Output mode: {'Separate files' if args.separate_files else 'Combined file'}")
    
    all_filtered_clips = []
    total_clips_count = 0
    
    if args.separate_files:
        # Create separate JSONL file for each video
        for video_dir in video_dirs:
            filtered_clips, clips_count = process_video_directory(
                video_dir,
                args.min_duration,
                args.max_duration
            )
            
            if filtered_clips:
                # Create output file for this video
                output_file = Path(args.output_jsonl).parent / f"filtered_{video_dir.name}.jsonl"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for entry in filtered_clips:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                print(f"Created: {output_file}")
            
            all_filtered_clips.extend(filtered_clips)
            total_clips_count += clips_count
    else:
        # Create combined JSONL file
        output_jsonl = Path(args.output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
            for video_dir in video_dirs:
                filtered_clips, clips_count = process_video_directory(
                    video_dir,
                    args.min_duration,
                    args.max_duration
                )
                
                # Write filtered clips to JSONL
                for entry in filtered_clips:
                    jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                all_filtered_clips.extend(filtered_clips)
                total_clips_count += clips_count
        
        print(f"\n{'='*80}")
        print(f"Created combined JSONL file: {output_jsonl}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total video clips processed: {total_clips_count}")
    print(f"Total clips matching duration filter: {len(all_filtered_clips)}")
    print(f"Total clips skipped: {total_clips_count - len(all_filtered_clips)}")
    
    # Breakdown by source
    print(f"\nBreakdown by source:")
    source_counts = {}
    for entry in all_filtered_clips:
        source = entry['source']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    for source in sorted(source_counts.keys()):
        print(f"  {source}: {source_counts[source]} clips")
    
    # Duration statistics
    if all_filtered_clips:
        durations = [entry['duration'] for entry in all_filtered_clips]
        print(f"\nDuration statistics:")
        print(f"  Min: {min(durations):.2f}s")
        print(f"  Max: {max(durations):.2f}s")
        print(f"  Avg: {sum(durations)/len(durations):.2f}s")
    
    print(f"\n{'='*80}")
    
    return 0


if __name__ == '__main__':
    exit(main())

