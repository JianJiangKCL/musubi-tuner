#!/usr/bin/env python3
"""
Extract the first frame from each video in a directory and save as images.

Defaults:
- Input directory: /mnt/cfs/jj/musubi-tuner/datasets/baby_bee/raw_data
- Output directory: /mnt/cfs/jj/musubi-tuner/datasets/baby_bee/first_frames

Usage examples:
  python extract_1st_frame.py
  python extract_1st_frame.py --input-dir /path/to/videos --output-dir /path/to/frames
  python extract_1st_frame.py --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


DEFAULT_INPUT_DIR = \
    Path("/mnt/cfs/jj/musubi-tuner/datasets/baby_bee/raw_data")
DEFAULT_OUTPUT_DIR = \
    Path("/mnt/cfs/jj/musubi-tuner/datasets/baby_bee/first_frames")


def is_video_file(path: Path) -> bool:
    video_exts = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
        ".mpg",
        ".mpeg",
        ".wmv",
        ".flv",
        ".ts",
    }
    return path.is_file() and path.suffix.lower() in video_exts


def which(program: str) -> Optional[str]:
    return shutil.which(program)


def extract_frame_with_ffmpeg(
    input_video: Path, output_image: Path, overwrite: bool
) -> Tuple[bool, str]:
    ffmpeg_path = which("ffmpeg")
    if not ffmpeg_path:
        return False, "ffmpeg not found"

    output_image.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_image),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return True, ""
        else:
            return False, proc.stderr.strip() or proc.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def extract_frame_with_opencv(
    input_video: Path, output_image: Path, overwrite: bool
) -> Tuple[bool, str]:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return False, f"opencv import error: {exc}"

    if output_image.exists() and not overwrite:
        return True, "exists"

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        return False, "failed to open video"
    try:
        success, frame = cap.read()
        if not success or frame is None:
            return False, "failed to read frame"

        output_image.parent.mkdir(parents=True, exist_ok=True)
        write_ok = cv2.imwrite(str(output_image), frame)
        if not write_ok:
            return False, "failed to write image"
        return True, ""
    finally:
        cap.release()


def iter_video_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if is_video_file(path):
            yield path


def build_output_path(output_dir: Path, video_path: Path, image_ext: str) -> Path:
    return output_dir / f"{video_path.stem}{image_ext}"


def process_directory(
    input_dir: Path, output_dir: Path, overwrite: bool, image_ext: str
) -> int:
    total = 0
    succeeded = 0
    failed: List[Tuple[Path, str]] = []

    for video_path in iter_video_files(input_dir):
        total += 1
        output_path = build_output_path(output_dir, video_path, image_ext)

        # Try ffmpeg first for speed and robustness, then fallback to OpenCV
        ok, msg = extract_frame_with_ffmpeg(video_path, output_path, overwrite)
        if not ok:
            ok, msg = extract_frame_with_opencv(video_path, output_path, overwrite)

        if ok:
            succeeded += 1
            print(f"[OK] {video_path.name} -> {output_path.name}")
        else:
            failed.append((video_path, msg))
            print(f"[FAIL] {video_path.name}: {msg}")

    print(
        f"\nDone. Processed {total} video(s). Success: {succeeded}. Failures: {len(failed)}."
    )
    if failed:
        print("Failures:")
        for vp, reason in failed:
            print(f"  - {vp.name}: {reason}")
    return 0 if succeeded == total else 1 if succeeded > 0 else 2


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the first frame from each video in a directory and save as images."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing input videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write output images",
    )
    parser.add_argument(
        "--ext",
        dest="image_ext",
        choices=[".jpg", ".png"],
        default=".jpg",
        help="Image extension/format to write",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images if present",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    overwrite: bool = args.overwrite
    image_ext: str = args.image_ext

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist or is not a directory: {input_dir}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    return process_directory(input_dir, output_dir, overwrite, image_ext)


if __name__ == "__main__":
    sys.exit(main())

