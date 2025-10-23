#!/usr/bin/env python3
"""
Create all video clips from multiple videos and their annotation files (no duration filtering).

Usage:
    python create_all_clips4allvideos.py

This script processes all videos with their corresponding annotation files and creates
video clips with corresponding text files.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def parse_timestamp(timestamp_str, is_plain_integer_seconds=False, is_mmss_with_dot=False):
    """Convert timestamp string to total seconds.
    
    Args:
        timestamp_str: The timestamp string to parse
        is_plain_integer_seconds: If True, plain integers are treated as seconds.
                                 If False, they're treated as decimal minutes.
        is_mmss_with_dot: If True, decimal format is treated as MM.SS (e.g., 0.3 = 3s, not 18s)
    
    Formats supported:
        - MM:SS format (e.g., "9:43" = 9*60+43 = 583 seconds)
        - MM'SS format (e.g., "1'30" = 1*60+30 = 90 seconds)
        - MM.SS format (e.g., "75.54" = 75*60+54 = 4554 seconds) - cholec7 special format
        - Decimal minutes (e.g., "0.9" = 0.9*60 = 54 seconds)
        - Plain integers (e.g., "14" = 14 seconds OR 14 minutes depending on flag)
    """
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
        # MM.SS format (cholec7): 0.3 = 3s, 75.54 = 4554s
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
    """Extract timestamp strings from text, handling multiple formats."""
    timestamps = []
    
    # Pattern 1: MM'SS format (e.g., "1'30", "10'45")
    pattern1 = r"\d+'\d+"
    timestamps.extend(re.findall(pattern1, text))
    
    # Pattern 2: Decimal format with dash (e.g., "0.32-0.34")
    # First, find all standalone decimals that are part of ranges
    pattern2 = r"\d+\.\d+-\d+\.\d+"
    range_matches = re.findall(pattern2, text)
    for match in range_matches:
        timestamps.extend(match.split('-'))
    
    # Pattern 3: Standalone decimal numbers (e.g., "0.9", "1.33")
    # Only if not already captured as part of a range
    if not range_matches:
        pattern3 = r"\d+\.\d+"
        timestamps.extend(re.findall(pattern3, text))
    
    # Pattern 4: Plain integers (seconds) - only if no other format found
    if not timestamps:
        pattern4 = r"\b\d+\b"
        timestamps.extend(re.findall(pattern4, text))
    
    return timestamps


def detect_timestamp_format(timestamps):
    """Detect whether timestamps are in seconds (plain integers) or other formats.
    
    Returns True if timestamps should be treated as plain seconds (cholec14 style),
    False otherwise (cholec2/cholec36 styles with minutes).
    """
    # If any timestamp has ' or . or :, then it's not plain seconds
    for ts in timestamps:
        if "'" in ts or '.' in ts or ':' in ts:
            return False
    # All timestamps are plain integers
    return True


def parse_line_txt_format(line, is_mmss_dot=False):
    """Parse a single line from the TXT annotation file.
    
    Args:
        line: The line to parse
        is_mmss_dot: If True, decimal format is MM.SS (cholec7), else decimal minutes (cholec2)
    
    Handles multiple formats:
    1. Tab-separated with decimal minutes: "action\t0.32-0.34\t1.02-1.04" (cholec2)
    2. Tab-separated with MM.SS: "action\t0.3-0.10\t1.19-1.48" (cholec7)
    3. Space-separated with seconds: "action 15 20" (cholec14)
    4. Space-separated with MM'SS: "action 1'02 1'48" (cholec36)
    """
    line = line.strip()
    if not line:
        return None

    # Try tab-separated format first (cholec2/cholec7 style)
    if '\t' in line:
        parts = line.split('\t')
        action_desc = parts[0].strip()
        timestamp_fields = [p.strip() for p in parts[1:] if p.strip()]
    else:
        # Space-separated format (cholec14, cholec36 style)
        # Find all timestamps in the line first
        all_timestamps = extract_timestamps_from_text(line)
        
        if not all_timestamps:
            return None
        
        # Extract action description (everything before first timestamp)
        first_ts_pos = len(line)
        for ts in all_timestamps:
            pos = line.find(ts)
            if pos != -1 and pos < first_ts_pos:
                first_ts_pos = pos
        
        action_desc = line[:first_ts_pos].strip()
        
        # Remove special prefixes
        action_desc = re.sub(r'^(state|reason|constant)\s+', '', action_desc)
        
        # Get the timestamp part
        timestamp_text = line[first_ts_pos:].strip()
        timestamp_fields = [timestamp_text]
    
    if not action_desc:
        return None
    
    # Parse timestamp ranges
    time_ranges = []
    
    for ts_field in timestamp_fields:
        # Extract all timestamps from this field
        timestamps = extract_timestamps_from_text(ts_field)
        
        if not timestamps:
            continue
        
        # Detect whether timestamps are plain seconds or other formats
        is_plain_seconds = detect_timestamp_format(timestamps)
        
        # Check if it's a range format "X-Y" in the original text
        if '-' in ts_field and len(timestamps) == 2:
            # It's a range
            start_time = parse_timestamp(timestamps[0], is_plain_seconds, is_mmss_dot)
            end_time = parse_timestamp(timestamps[1], is_plain_seconds, is_mmss_dot)
            
            if start_time is not None and end_time is not None and start_time < end_time:
                time_ranges.append((start_time, end_time))
        elif len(timestamps) == 1:
            # Single timestamp: create 2-second clip
            start_time = parse_timestamp(timestamps[0], is_plain_seconds, is_mmss_dot)
            if start_time is not None:
                end_time = start_time + 2  # Add 2 seconds
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


def parse_json_format(json_file):
    """Parse JSON annotation file (cholec7 format)."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    actions = []
    
    for entry in data:
        entry_type = entry.get('type')
        text = entry.get('text', '')
        
        if not text:
            continue
        
        time_ranges = []
        
        if entry_type == 'instant':
            # For instant events, create a 2-second clip
            start_s = entry.get('time_s')
            if start_s is not None:
                end_s = start_s + 2
                time_ranges.append((start_s, end_s))
        
        elif entry_type == 'segment':
            # For segments, use the start and end times
            start_s = entry.get('start_s')
            end_s = entry.get('end_s')
            
            if start_s is not None and end_s is not None and start_s < end_s:
                time_ranges.append((start_s, end_s))
        
        if time_ranges:
            # Add tool information if available
            tools_info = []
            if 'tools_in' in entry:
                tools_info.append(f"Tools in: {', '.join(entry['tools_in'])}")
            if 'tools_observed' in entry:
                tools_info.append(f"Tools observed: {', '.join(entry['tools_observed'])}")
            
            action_desc = text
            if tools_info:
                action_desc += f" [{'; '.join(tools_info)}]"
            
            actions.append({
                'action': action_desc,
                'time_ranges': time_ranges
            })
    
    return actions


def get_indent_level(line):
    """Get indentation level of a line.
    
    Returns:
        int: Number of leading tabs/spaces. Treats 4 spaces = 1 level, 8 spaces = 2 levels, etc.
    """
    if not line or not line[0].isspace():
        return 0
    
    # Count tabs
    if line.startswith('\t'):
        return len(line) - len(line.lstrip('\t'))
    
    # Count spaces (treat 4 spaces as one level)
    spaces = len(line) - len(line.lstrip(' '))
    return spaces // 4 if spaces > 0 else 0


def detect_mmss_with_dot_format(txt_file):
    """Detect if the TXT file uses MM.SS format (cholec7 style).
    
    Returns True if file uses MM.SS format (period instead of apostrophe),
    False otherwise.
    """
    # Check filename first
    if 'cholec7' in str(txt_file).lower():
        return True
    
    # Check content - look for patterns like "75.54" which would be >75 minutes as decimal
    # but makes sense as 75 min 54 sec
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for large decimal numbers (> 60) which would indicate MM.SS format
    # (since decimal minutes > 60 would be unusual)
    large_decimals = re.findall(r'\b([6-9]\d\.\d+|\d{3,}\.\d+)\b', content)
    if large_decimals:
        return True
    
    return False


def parse_txt_format(txt_file):
    """Parse TXT annotation file with hierarchical structure support.
    
    Handles files where:
    - Lines with less/no indent are parent actions (may just be time period markers)
    - Lines with more indent are sub-actions (actual clips to extract)
    
    Works with both tab and space indentation (cholec2 uses tabs, cholec14/36 use spaces).
    
    Strategy: Skip parent actions that have sub-actions following them.
    """
    actions = []
    
    # Detect if this file uses MM.SS format (cholec7)
    is_mmss_dot = detect_mmss_with_dot_format(txt_file)
    
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
        
        # Check if next non-empty line has more indentation (is a sub-action)
        has_sub_actions = False
        for j in range(i + 1, len(lines)):
            next_line = lines[j]
            if not next_line.strip():
                continue  # Skip empty lines
            next_indent = get_indent_level(next_line)
            if next_indent > current_indent:
                has_sub_actions = True
            break  # Check only the first non-empty line
        
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


def process_video(video_file, annotation_file, output_dir, clip_prefix):
    """Process a single video with its annotation file."""
    # Validate input files
    video_file = Path(video_file)
    annotation_file = Path(annotation_file)
    output_dir = Path(output_dir)

    if not video_file.exists():
        print(f"Error: Video file {video_file} does not exist")
        return 0

    if not annotation_file.exists():
        print(f"Error: Annotation file {annotation_file} does not exist")
        return 0

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / 'clips'
    texts_dir = output_dir / 'texts'
    clips_dir.mkdir(exist_ok=True)
    texts_dir.mkdir(exist_ok=True)

    # Parse annotation file based on format
    print(f"\nProcessing video: {video_file.name}")
    print(f"Annotation file: {annotation_file.name}")
    
    if annotation_file.suffix == '.json':
        print("Using JSON format parser")
        actions = parse_json_format(annotation_file)
    else:
        print("Using TXT format parser")
        actions = parse_txt_format(annotation_file)

    print(f"Found {len(actions)} valid actions with time ranges")

    # Create clips and text files
    clip_count = 0
    total_ranges = sum(len(action['time_ranges']) for action in actions)

    print(f"Processing {total_ranges} time ranges...")

    for action in actions:
        action_desc = action['action']
        sanitized_desc = sanitize_filename(action_desc)[:50]  # Limit length

        for range_idx, (start_time, end_time) in enumerate(action['time_ranges']):
            clip_count += 1
            duration = end_time - start_time

            # Create filename
            if len(action['time_ranges']) > 1:
                clip_name = f"{clip_prefix}_{clip_count:04d}_part{range_idx + 1}"
            else:
                clip_name = f"{clip_prefix}_{clip_count:04d}"

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

    print(f"Completed {video_file.name}! Created {clip_count} video clips and text files")
    return clip_count


def main():
    parser = argparse.ArgumentParser(description='Create ALL video clips from multiple videos and their annotations')
    parser.add_argument('--output_base_dir', default='/mnt/cfs/jj/musubi-tuner/tmp_data/clips_all_video', 
                        help='Base output directory for all clips')

    args = parser.parse_args()

    # Define video and annotation file pairs
    video_annotation_pairs = [
        {
            'video': '/mnt/cfs/jj/musubi-tuner/tmp_data/video02.mp4',
            'annotation': '/mnt/cfs/jj/musubi-tuner/tmp_data/myanno-cholec2.txt',
            'prefix': 'cholec02'
        },
        {
            'video': '/mnt/cfs/jj/musubi-tuner/tmp_data/video07.mp4',
            'annotation': '/mnt/cfs/jj/musubi-tuner/tmp_data/myanno-cholec7.txt',
            'prefix': 'cholec07'
        },
        {
            'video': '/mnt/cfs/jj/musubi-tuner/tmp_data/video14.mp4',
            'annotation': '/mnt/cfs/jj/musubi-tuner/tmp_data/myanno-cholec14.txt',
            'prefix': 'cholec14'
        },
        {
            'video': '/mnt/cfs/jj/musubi-tuner/tmp_data/video36.mp4',
            'annotation': '/mnt/cfs/jj/musubi-tuner/tmp_data/myanno-cholec36.txt',
            'prefix': 'cholec36'
        }
    ]

    print("=" * 80)
    print("Starting batch processing of all videos")
    print("=" * 80)

    total_clips = 0

    for pair in video_annotation_pairs:
        video_file = pair['video']
        annotation_file = pair['annotation']
        prefix = pair['prefix']
        
        # Create output directory for this video
        output_dir = Path(args.output_base_dir) / prefix
        
        clips_created = process_video(video_file, annotation_file, output_dir, prefix)
        total_clips += clips_created

    print("\n" + "=" * 80)
    print(f"ALL PROCESSING COMPLETE!")
    print(f"Total clips created: {total_clips}")
    print(f"Output base directory: {args.output_base_dir}")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    exit(main())
