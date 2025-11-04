# Video Clip Filtering with Caption Cleaning

## Overview

The `filter_clips_by_duration4all.py` script processes all video clips from multiple sources, filters them by duration, and creates a JSONL file with cleaned captions.

## Features

### 1. Duration Filtering
- Filter clips by minimum and maximum duration
- Default range: 2.0s - 10.0s (customizable)

### 2. Caption Cleaning

The script automatically cleans captions by:

#### Removing Tool Labels
- `抓钳A`, `抓钳B`, `抓钳C`, `抓钳0`, `抓钳1`, `抓钳2` → `抓钳`
- `戳卡a`, `戳卡b`, `戳卡c`, `戳卡d` → `戳卡`
- `电凝钩A`, `电凝钩B` → `电凝钩`
- `双极电凝0`, `双极电凝1` → `双极电凝`

#### Handling Multiple Tool Mentions
- `抓钳A、抓钳B、抓钳C协作` → `抓钳协作`
- `抓钳A和抓钳B调整位置` → `抓钳和抓钳调整位置`
- `抓钳A、B、C协作` → `抓钳协作`
- `戳卡a和戳卡b同时进入` → `戳卡和戳卡同时进入`

#### Removing Bracket Content
- Removes tool information like `[Tools in: 抓钳A, 抓钳B; Tools observed: 抓钳B]`
- Removes any content in square brackets

#### Content Filtering
The script also filters out clips with:
- **CVS related content**: Any caption containing "CVS" or "cvs"
- **Surgery end markers**: Captions containing "手术结束" (surgery end)
- **Temporary actions**: Captions containing "暂时" (temporarily)

### 3. Output Formats

#### Combined Mode (default)
Creates a single JSONL file with all filtered clips:
```bash
python filter_clips_by_duration4all.py --output_jsonl filtered.jsonl
```

#### Separate Mode
Creates individual JSONL files for each video:
```bash
python filter_clips_by_duration4all.py --separate_files --output_jsonl filtered.jsonl
```
This creates:
- `filtered_cholec02.jsonl`
- `filtered_cholec07.jsonl`
- `filtered_cholec14.jsonl`
- `filtered_cholec36.jsonl`

## Usage

### Basic Usage

```bash
# Use defaults (2-10 seconds, default directory)
python filter_clips_by_duration4all.py

# Custom duration range
python filter_clips_by_duration4all.py --min_duration 3.0 --max_duration 15.0

# Custom output file
python filter_clips_by_duration4all.py --output_jsonl my_clips.jsonl

# Custom directory
python filter_clips_by_duration4all.py /path/to/clips_directory
```

### Complete Example

```bash
python filter_clips_by_duration4all.py \
  /mnt/cfs/jj/musubi-tuner/tmp_data/clips_all_video \
  --min_duration 2.0 \
  --max_duration 10.0 \
  --output_jsonl filtered.jsonl
```

## Output Format

Each entry in the JSONL file contains:

```json
{
  "video_path": "/full/path/to/clip.mp4",
  "caption": "抓钳向上牵拉胆囊",
  "duration": 3.67,
  "source": "cholec02"
}
```

## Current Results

From the latest run with 2-10 second filter:

- **Total clips processed**: 1,105
- **Clips matching filter**: 507 (45.9%)
- **Average duration**: 3.96s

### Breakdown by Source

| Source | Clips | Percentage |
|--------|-------|------------|
| cholec02 | 64 | 12.6% |
| cholec07 | 244 | 48.1% |
| cholec14 | 134 | 26.4% |
| cholec36 | 65 | 12.8% |

### Caption Cleaning Examples

| Original | Cleaned |
|----------|---------|
| 抓钳A向上牵拉胆囊 | 抓钳向上牵拉胆囊 |
| 戳卡a进入腹腔 | 戳卡进入腹腔 |
| 抓钳B向左下方牵拉胆囊周围组织 | 抓钳向左下方牵拉胆囊周围组织 |
| 抓钳A、抓钳B、抓钳C协作向上牵拉胆囊 | 抓钳协作向上牵拉胆囊 |
| 戳卡a和戳卡b同时进入 | 戳卡和戳卡同时进入 |
| 抓钳A和抓钳B调整位置 [Tools in: 抓钳A, 抓钳B] | 抓钳和抓钳调整位置 |
| 电凝钩A分离组织 | 电凝钩分离组织 |

### Content Filtering Examples

The following types of clips are **excluded** from the output:

| Original Caption | Reason |
|-----------------|--------|
| cvs第1项和第3项标准完成 | Contains "cvs" |
| CVS第二项标准完成 | Contains "CVS" |
| 可见胆囊管与胆囊动脉CVS13两项标准完成 | Contains "CVS" |
| 腔镜移出体外，手术结束 | Contains "手术结束" |
| 电凝钩暂时离开 | Contains "暂时" |
| 抓钳暂时离开胆囊 | Contains "暂时" |
| 双极电凝暂时离开 | Contains "暂时" |

## Verification

All captions have been verified to ensure:
- ✓ All 抓钳 labels (A/B/C/0/1/2) have been removed
- ✓ All 戳卡 labels (a/b/c/d) have been removed
- ✓ All bracket content has been removed
- ✓ All CVS related content has been filtered out (8 items)
- ✓ All surgery end markers have been filtered out (1 item)
- ✓ All temporary action markers "暂时" have been filtered out (7 items)
- ✓ Text remains meaningful and grammatically correct

## Files

- `filter_clips_by_duration4all.py` - Main filtering script with caption cleaning
- `filtered.jsonl` - Output file with filtered and cleaned clips
- `filter_clips_by_duration.py` - Original single-directory version (deprecated)

## Notes

- The script automatically detects and processes all subdirectories in the base directory
- Each subdirectory should have `clips/` and `texts/` subdirectories
- Progress is displayed during processing
- Statistics are provided at the end of processing

