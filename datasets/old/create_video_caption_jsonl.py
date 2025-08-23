#!/usr/bin/env python3
"""
Script to create a JSONL file mapping video paths to their captions.
Matches videos from a directory with captions from a CSV file.
"""

import os
import csv
import json
import argparse
from pathlib import Path


def load_captions_from_csv(csv_path):
    """
    Load video captions from CSV file into a dictionary.
    
    Args:
        csv_path (str): Path to the CSV file containing video captions
        
    Returns:
        dict: Mapping from video filename to caption
    """
    captions_dict = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            # Read the first line to check if it's a header
            first_line = csvfile.readline().strip()
            csvfile.seek(0)  # Reset to beginning
            
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                video_name = row['video'].strip()
                caption = row['caption'].strip()
                captions_dict[video_name] = caption
                
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}
    
    print(f"Loaded {len(captions_dict)} captions from CSV file")
    return captions_dict


def find_video_files(video_dir):
    """
    Find all video files in the specified directory.
    
    Args:
        video_dir (str): Path to directory containing video files
        
    Returns:
        list: List of full paths to video files
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    video_files = []
    
    try:
        video_path = Path(video_dir)
        if not video_path.exists():
            print(f"Video directory does not exist: {video_dir}")
            return []
        
        for file_path in video_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_files.append(str(file_path.absolute()))
                
    except Exception as e:
        print(f"Error scanning video directory: {e}")
        return []
    
    print(f"Found {len(video_files)} video files in directory")
    return video_files


def create_jsonl_file(video_files, captions_dict, output_path):
    """
    Create JSONL file with video paths and captions.
    
    Args:
        video_files (list): List of video file paths
        captions_dict (dict): Mapping from video filename to caption
        output_path (str): Path for output JSONL file
    """
    matched_count = 0
    unmatched_videos = []
    
    try:
        with open(output_path, 'w', encoding='utf-8') as jsonl_file:
            for video_path in video_files:
                # Extract filename from full path
                video_filename = os.path.basename(video_path)
                
                # Try to find caption for this video
                if video_filename in captions_dict:
                    caption = captions_dict[video_filename]
                    
                    # Create JSON object
                    json_obj = {
                        "video_path": video_path,
                        "caption": caption
                    }
                    
                    # Write to JSONL file
                    jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                    matched_count += 1
                else:
                    unmatched_videos.append(video_filename)
    
    except Exception as e:
        print(f"Error writing JSONL file: {e}")
        return
    
    print(f"Successfully created JSONL file: {output_path}")
    print(f"Matched {matched_count} videos with captions")
    
    if unmatched_videos:
        print(f"Warning: {len(unmatched_videos)} videos had no matching captions:")
        for video in unmatched_videos[:10]:  # Show first 10 unmatched
            print(f"  - {video}")
        if len(unmatched_videos) > 10:
            print(f"  ... and {len(unmatched_videos) - 10} more")


def main():
    # Set default paths
    video_dir = "/mnt/cfs/jj/datasets/OpenVid-1M/OpenVidHD/OpenVidHD_part_14"
    csv_path = "/mnt/cfs/jj/musubi-tuner/OpenVidHD.csv"
    output_path = "/mnt/cfs/jj/musubi-tuner/video_captions.jsonl"
    
    parser = argparse.ArgumentParser(description="Create JSONL file mapping videos to captions")
    parser.add_argument("--video-dir", default=video_dir, 
                       help=f"Directory containing video files (default: {video_dir})")
    parser.add_argument("--csv-path", default=csv_path,
                       help=f"Path to CSV file with captions (default: {csv_path})")
    parser.add_argument("--output", default=output_path,
                       help=f"Output JSONL file path (default: {output_path})")
    
    args = parser.parse_args()
    
    print("Starting video-caption matching process...")
    print(f"Video directory: {args.video_dir}")
    print(f"CSV file: {args.csv_path}")
    print(f"Output file: {args.output}")
    print("-" * 50)
    
    # Load captions from CSV
    captions_dict = load_captions_from_csv(args.csv_path)
    if not captions_dict:
        print("Failed to load captions from CSV file. Exiting.")
        return
    
    # Find video files
    video_files = find_video_files(args.video_dir)
    if not video_files:
        print("No video files found in directory. Exiting.")
        return
    
    # Create JSONL file
    create_jsonl_file(video_files, captions_dict, args.output)
    
    print("-" * 50)
    print("Process completed!")


if __name__ == "__main__":
    main()