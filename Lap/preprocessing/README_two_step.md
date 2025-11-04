# Two-Step Video Processing Pipeline

This pipeline separates video clip creation from duration filtering for better efficiency and flexibility.

## Overview

1. **Step 1**: `create_all_clips.py` - Creates ALL video clips with text files (supports both timestamp pairs and single timestamps)
2. **Step 2**: `filter_clips_by_duration.py` - Filters clips by duration and creates JSONL files

## Files

- `create_all_clips.py` - Creates all video clips and text files (no filtering)
- `filter_clips_by_duration.py` - Filters existing clips by duration and creates JSONL
- `run_two_step_process.sh` - Automated script to run both steps

## Usage

### Option 1: Automated Two-Step Process
```bash
cd /mnt/cfs/jj/musubi-tuner/tmp_data/preprocessing
./run_two_step_process.sh
```

### Option 2: Manual Step-by-Step

#### Step 1: Create All Clips
```bash
python create_all_clips.py \
    --annotation_file /mnt/cfs/jj/musubi-tuner/tmp_data/myanno-cholec36.txt \
    --video_file /mnt/cfs/jj/musubi-tuner/tmp_data/video36.mp4 \
    --output_dir /mnt/cfs/jj/musubi-tuner/tmp_data/clips_all \
    --clip_prefix cholec36
```

#### Step 2: Filter by Duration
```bash
# For clips ≤ 10 seconds
python filter_clips_by_duration.py \
    --clips_dir /mnt/cfs/jj/musubi-tuner/tmp_data/clips_all/clips \
    --texts_dir /mnt/cfs/jj/musubi-tuner/tmp_data/clips_all/texts \
    --max_duration 10.0 \
    --output_jsonl /mnt/cfs/jj/musubi-tuner/tmp_data/clips_all/video36_captions_10s.jsonl

# For clips ≤ 20 seconds
python filter_clips_by_duration.py \
    --clips_dir /mnt/cfs/jj/musubi-tuner/tmp_data/clips_all/clips \
    --texts_dir /mnt/cfs/jj/musubi-tuner/tmp_data/clips_all/texts \
    --max_duration 20.0 \
    --output_jsonl /mnt/cfs/jj/musubi-tuner/tmp_data/clips_all/video36_captions_20s.jsonl
```

## Parameters

### create_all_clips.py
- `--annotation_file`: Path to annotation text file
- `--video_file`: Path to video file
- `--output_dir`: Output directory (default: `/mnt/cfs/jj/musubi-tuner/tmp_data/clips_all`)
- `--clip_prefix`: Prefix for clip filenames (default: 'clip')

### filter_clips_by_duration.py
- `--clips_dir`: Directory containing video clips
- `--texts_dir`: Directory containing text files
- `--max_duration`: Maximum duration in seconds
- `--min_duration`: Minimum duration in seconds (default: 0.0)
- `--output_jsonl`: Output JSONL file path

## Output Structure

```
clips_all/
├── clips/                          # All video clips
│   ├── cholec36_0001.mp4
│   ├── cholec36_0002.mp4
│   └── ...
├── texts/                          # Text files with captions and duration info
│   ├── cholec36_0001.txt
│   ├── cholec36_0002.txt
│   └── ...
├── video36_captions_5s.jsonl       # Clips ≤ 5 seconds
├── video36_captions_10s.jsonl      # Clips ≤ 10 seconds
├── video36_captions_20s.jsonl      # Clips ≤ 20 seconds
├── video36_captions_30s.jsonl      # Clips ≤ 30 seconds
└── video36_captions_all.jsonl      # All clips
```

## Supported Annotation Formats

The pipeline supports two types of timestamp annotations:

### Timestamp Pairs (Time Ranges)
```
抓钳A向上阻挡胆囊和肝脏 1'54 5'32
电凝钩开始沿胆囊床逆行剥离胆囊 20'20 21'07 - 21'14 21'51
```
Creates clips for the specified time ranges.

### Single Timestamps (2-Second Clips)
```
戳卡a进入腹腔 0'47
钛夹出现 18'08
```
Creates 2-second clips starting from the specified timestamp (e.g., 0'47 → clip from 47s to 49s).

## Text File Format

Each text file contains:
```
抓钳A向上阻挡胆囊和肝脏
Duration: 98.0s
Time range: 114s - 212s
```

## JSONL Format

Each line contains:
```json
{"video_path": "/full/path/to/clip.mp4", "caption": "action description"}
```

## Advantages

1. **Efficiency**: Create clips once, filter multiple times
2. **Flexibility**: Easy to create different duration filters
3. **Debugging**: Can inspect individual text files for duration info
4. **Reusability**: Can create new JSONL files without re-processing video
5. **Storage**: Keep all clips, just filter the training data as needed

## Example Workflow

1. Run `./run_two_step_process.sh` once to create all clips
2. Use different JSONL files for different training experiments:
   - `video36_captions_5s.jsonl` for short clip training
   - `video36_captions_20s.jsonl` for medium clip training
   - `video36_captions_all.jsonl` for full dataset training
