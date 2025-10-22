#!/usr/bin/env python3
"""
Recreate cholec02 clips and texts only.

Usage:
    python recreate_cholec02_clips.py
"""

import argparse
import re
import subprocess
from pathlib import Path
import shutil


def parse_timestamp(timestamp_str, is_plain_integer_seconds=False, is_mmss_with_dot=False):
    """Convert timestamp string to total seconds."""
    # Handle MM:SS format
    if ':' in timestamp_str:
        parts = timestamp_str.split(':')
        if len(parts) == 2:
            try:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            except ValueError:
                return None
    
    # Handle MM'SS format
    if "'" in timestamp_str:
        parts = timestamp_str.split("'")
        if len(parts) == 2:
            try:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            except ValueError:
                return None
    
    # Handle decimal format (could be MM.SS or decimal minutes)
    if '.' in timestamp_str and is_mmss_with_dot:
        # MM.SS format
        parts = timestamp_str.split('.')
        if len(parts) == 2:
            try:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            except ValueError:
                return None
    
    # Handle decimal format or plain integers
    try:
        value = float(timestamp_str)
        
        # If it's a plain integer and the flag is set, treat as seconds
        if is_plain_integer_seconds and timestamp_str.isdigit():
            return value
        # Otherwise, treat as decimal minutes
        else:
            return value * 60
    except ValueError:
        return None


def extract_timestamps_from_text(text):
    """Extract timestamp strings from text."""
    timestamps = []
    
    # Pattern 1: MM'SS format
    pattern1 = r"\d+'\d+"
    timestamps.extend(re.findall(pattern1, text))
    
    # Pattern 2: Decimal format with dash (e.g., "0.32-0.34")
    pattern2 = r"\d+\.\d+-\d+\.\d+"
    range_matches = re.findall(pattern2, text)
    for match in range_matches:
        timestamps.extend(match.split('-'))
    
    # Pattern 3: Standalone decimal numbers
    if not range_matches:
        pattern3 = r"\d+\.\d+"
        timestamps.extend(re.findall(pattern3, text))
    
    # Pattern 4: Plain integers
    if not timestamps:
        pattern4 = r"\b\d+\b"
        timestamps.extend(re.findall(pattern4, text))
    
    return timestamps


def detect_timestamp_format(timestamps):
    """Detect whether timestamps are in seconds or other formats."""
    for ts in timestamps:
        if "'" in ts or '.' in ts or ':' in ts:
            return False
    return True


def parse_line_txt_format(line, is_mmss_dot=False):
    """Parse a single line from the TXT annotation file."""
    line = line.strip()
    if not line:
        return None

    # Tab-separated format
    if '\t' in line:
        parts = line.split('\t')
        action_desc = parts[0].strip()
        timestamp_fields = [p.strip() for p in parts[1:] if p.strip()]
    else:
        # Space-separated format
        all_timestamps = extract_timestamps_from_text(line)
        
        if not all_timestamps:
            return None
        
        # Extract action description
        first_ts_pos = len(line)
        for ts in all_timestamps:
            pos = line.find(ts)
            if pos != -1 and pos < first_ts_pos:
                first_ts_pos = pos
        
        action_desc = line[:first_ts_pos].strip()
        action_desc = re.sub(r'^(state|reason|constant)\s+', '', action_desc)
        
        timestamp_text = line[first_ts_pos:].strip()
        timestamp_fields = [timestamp_text]
    
    if not action_desc:
        return None
    
    # Parse timestamp ranges
    time_ranges = []
    
    for ts_field in timestamp_fields:
        timestamps = extract_timestamps_from_text(ts_field)
        
        if not timestamps:
            continue
        
        is_plain_seconds = detect_timestamp_format(timestamps)
        
        # Check if it's a range format
        if '-' in ts_field and len(timestamps) == 2:
            start_time = parse_timestamp(timestamps[0], is_plain_seconds, is_mmss_dot)
            end_time = parse_timestamp(timestamps[1], is_plain_seconds, is_mmss_dot)
            
            if start_time is not None and end_time is not None and start_time < end_time:
                time_ranges.append((start_time, end_time))
        elif len(timestamps) == 1:
            # Single timestamp: create 2-second clip
            start_time = parse_timestamp(timestamps[0], is_plain_seconds, is_mmss_dot)
            if start_time is not None:
                end_time = start_time + 2
                time_ranges.append((start_time, end_time))
        else:
            # Multiple timestamps: parse as consecutive pairs
            i = 0
            while i < len(timestamps) - 1:
                start_time = parse_timestamp(timestamps[i], is_plain_seconds, is_mmss_dot)
                end_time = parse_timestamp(timestamps[i + 1], is_plain_seconds, is_mmss_dot)
                
                if start_time is not None and end_time is not None and start_time < end_time:
                    time_ranges.append((start_time, end_time))
                    i += 2
                else:
                    i += 1
    
    if not time_ranges:
        return None
    
    return {
        'action': action_desc,
        'time_ranges': time_ranges
    }


def get_indent_level(line):
    """Get indentation level of a line."""
    if not line or not line[0].isspace():
        return 0
    
    # Count tabs
    if line.startswith('\t'):
        return len(line) - len(line.lstrip('\t'))
    
    # Count spaces (treat 4 spaces as one level)
    spaces = len(line) - len(line.lstrip(' '))
    return spaces // 4 if spaces > 0 else 0


def parse_txt_format(txt_file):
    """Parse TXT annotation file with hierarchical structure support."""
    actions = []
    is_mmss_dot = False  # cholec02 uses decimal minutes, not MM.SS
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_num = i + 1
        
        if not line.strip():
            i += 1
            continue
        
        # Get indentation level of current line
        current_indent = get_indent_level(line)
        
        # Check if next non-empty line has more indentation
        has_sub_actions = False
        for j in range(i + 1, len(lines)):
            next_line = lines[j]
            if not next_line.strip():
                continue
            next_indent = get_indent_level(next_line)
            if next_indent > current_indent:
                has_sub_actions = True
            break
        
        # If current line has sub-actions, skip it (it's just a time period marker)
        # Otherwise, parse it as an actual action
        if not has_sub_actions or current_indent > 0:
            parsed = parse_line_txt_format(line, is_mmss_dot)
            if parsed:
                parsed['line_number'] = line_num
                actions.append(parsed)
        
        i += 1
    
    return actions


def create_video_clip(video_file, start_time, end_time, output_file):
    """Create a video clip using ffmpeg."""
    duration = end_time - start_time

    ffmpeg_path = '/mnt/cfs/jj/musubi-tuner/ffmpeg-4.4.1-amd64-static/ffmpeg'

    cmd = [
        ffmpeg_path,
        '-i', str(video_file),
        '-ss', str(start_time),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-avoid_negative_ts', 'make_zero',
        '-y',
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


def main():
    parser = argparse.ArgumentParser(description='Recreate cholec02 clips only')
    parser.add_argument('--video', default='/mnt/cfs/jj/musubi-tuner/tmp_data/video02.mp4',
                        help='Video file path')
    parser.add_argument('--annotation', default='/mnt/cfs/jj/musubi-tuner/tmp_data/myanno-cholec2.txt',
                        help='Annotation file path')
    parser.add_argument('--output_dir', default='/mnt/cfs/jj/musubi-tuner/tmp_data/clips_all_video/cholec02',
                        help='Output directory')
    parser.add_argument('--backup', action='store_true', default=True,
                        help='Backup existing clips and texts')
    
    args = parser.parse_args()
    
    video_file = Path(args.video)
    annotation_file = Path(args.annotation)
    output_dir = Path(args.output_dir)
    
    if not video_file.exists():
        print(f"Error: Video file {video_file} does not exist")
        return 1
    
    if not annotation_file.exists():
        print(f"Error: Annotation file {annotation_file} does not exist")
        return 1
    
    # Backup existing clips and texts
    if args.backup and output_dir.exists():
        backup_dir = output_dir.parent / f"{output_dir.name}_backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        print(f"Backing up existing files to {backup_dir}")
        shutil.copytree(output_dir, backup_dir)
    
    # Create output directories
    clips_dir = output_dir / 'clips'
    texts_dir = output_dir / 'texts'
    
    # Remove existing clips and texts
    if clips_dir.exists():
        shutil.rmtree(clips_dir)
    if texts_dir.exists():
        shutil.rmtree(texts_dir)
    
    clips_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("RECREATING CHOLEC02 CLIPS")
    print("=" * 80)
    print(f"Video: {video_file}")
    print(f"Annotation: {annotation_file}")
    print(f"Output: {output_dir}")
    print()
    
    # Parse annotation file
    print("Parsing annotation file...")
    actions = parse_txt_format(annotation_file)
    print(f"Found {len(actions)} valid actions with time ranges")
    
    # Create clips and text files
    clip_count = 0
    total_ranges = sum(len(action['time_ranges']) for action in actions)
    
    print(f"Processing {total_ranges} time ranges...")
    print()
    
    for action in actions:
        action_desc = action['action']
        
        for range_idx, (start_time, end_time) in enumerate(action['time_ranges']):
            clip_count += 1
            duration = end_time - start_time
            
            # Create filename
            if len(action['time_ranges']) > 1:
                clip_name = f"cholec02_{clip_count:04d}_part{range_idx + 1}"
            else:
                clip_name = f"cholec02_{clip_count:04d}"
            
            clip_file = clips_dir / f"{clip_name}.mp4"
            text_file = texts_dir / f"{clip_name}.txt"
            
            if clip_count % 10 == 0:
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
    
    print()
    print("=" * 80)
    print(f"COMPLETED!")
    print(f"Created {clip_count} video clips and text files")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

