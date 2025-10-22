#!/usr/bin/env python3
"""
Extract the first frame from a video file or all videos in a directory.

The output image(s) will be saved in 'first_frame' subfolders within the same directory as each input video.

Usage examples:
  python extract_1st_frame.py video.mp4
  python extract_1st_frame.py /path/to/video.mp4 --overwrite
  python extract_1st_frame.py /path/to/video/folder
  python extract_1st_frame.py video.mov --ext .png
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


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
    """Iterate over video files in a directory."""
    for path in sorted(input_dir.iterdir()):
        if is_video_file(path):
            yield path


def process_single_video(video_path: Path, overwrite: bool, image_ext: str) -> bool:
    """Process a single video file and extract its first frame."""
    # Create output path in 'first_frame' subfolder of the video's directory
    output_dir = video_path.parent / "first_frame"
    output_path = output_dir / f"{video_path.stem}{image_ext}"

    # Try ffmpeg first for speed and robustness, then fallback to OpenCV
    ok, msg = extract_frame_with_ffmpeg(video_path, output_path, overwrite)
    if not ok:
        ok, msg = extract_frame_with_opencv(video_path, output_path, overwrite)

    if ok:
        print(f"[OK] {video_path.name} -> {output_path}")
        return True
    else:
        print(f"[FAIL] {video_path.name}: {msg}")
        return False


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the first frame from a video file or all videos in a directory."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Input video file or directory containing videos",
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
    input_path: Path = args.input_path
    overwrite: bool = args.overwrite
    image_ext: str = args.image_ext

    if not input_path.exists():
        print(f"Input path does not exist: {input_path}")
        return 2

    # Handle single video file
    if input_path.is_file():
        if not is_video_file(input_path):
            print(f"Input file is not a recognized video format: {input_path}")
            return 2

        success = process_single_video(input_path, overwrite, image_ext)
        return 0 if success else 1

    # Handle directory with video files
    elif input_path.is_dir():
        total = 0
        succeeded = 0

        for video_path in iter_video_files(input_path):
            total += 1
            if process_single_video(video_path, overwrite, image_ext):
                succeeded += 1

        if total == 0:
            print(f"No video files found in directory: {input_path}")
            return 2

        print(f"\nDone. Processed {total} video(s). Success: {succeeded}. Failures: {total - succeeded}.")
        return 0 if succeeded == total else 1 if succeeded > 0 else 2

    else:
        print(f"Input path is neither a file nor a directory: {input_path}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

