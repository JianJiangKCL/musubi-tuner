#!/usr/bin/env python3
import argparse
import json
import os
import sys
import zipfile
from typing import Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zip all videos referenced in a JSONL into a single archive."
    )
    parser.add_argument(
        "--jsonl",
        default="/mnt/cfs/jj/proj/musubi-tuner/Lap/preprocessing/filtered_clips_with_instruments.jsonl",
        help="Path to the input JSONL file (one JSON object per line, with 'video_path').",
    )
    parser.add_argument(
        "--output",
        default="/mnt/cfs/jj/proj/musubi-tuner/train_data.zip",
        help="Path to output zip file to create.",
    )
    parser.add_argument(
        "--relbase",
        default="/mnt/cfs/jj",
        help=(
            "Base directory to compute relative paths inside the zip. "
            "If a file is under this directory, its arcname will be relative to it."
        ),
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Fail if any referenced files are missing. By default, missing files are skipped.",
    )
    return parser.parse_args()


def load_unique_paths(jsonl_path: str) -> Tuple[List[str], int, int]:
    unique_paths: List[str] = []
    seen: Set[str] = set()
    bad_lines = 0
    no_path_lines = 0

    with open(jsonl_path, "r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            path = obj.get("video_path") or obj.get("path") or obj.get("file")
            if not path:
                no_path_lines += 1
                continue

            if path in seen:
                continue
            seen.add(path)
            unique_paths.append(path)

    return unique_paths, bad_lines, no_path_lines


def compute_arcname(file_path: str, relbase: str) -> str:
    try:
        file_path = os.path.abspath(file_path)
        relbase = os.path.abspath(relbase)
        common = os.path.commonpath([file_path, relbase])
        if common == relbase:
            return os.path.relpath(file_path, start=relbase)
    except Exception:
        pass

    for anchor in ("/clips_all_video/", "/clips/"):
        idx = file_path.find(anchor)
        if idx != -1:
            return file_path[idx + 1 :]

    return os.path.basename(file_path)


def filter_existing(paths: Iterable[str]) -> Tuple[List[str], List[str]]:
    existing: List[str] = []
    missing: List[str] = []
    for p in paths:
        if os.path.isfile(p):
            existing.append(p)
        else:
            missing.append(p)
    return existing, missing


def make_zip(files: Iterable[str], output_zip: str, relbase: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_zip))
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(
        output_zip, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for idx, path in enumerate(files, start=1):
            arcname = compute_arcname(path, relbase)
            zf.write(path, arcname)
            if idx % 100 == 0:
                print(f"Added {idx} files...", flush=True)


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.jsonl):
        print(f"ERROR: JSONL not found: {args.jsonl}", file=sys.stderr)
        return 2

    print(f"Reading: {args.jsonl}")
    all_paths, bad_lines, no_path_lines = load_unique_paths(args.jsonl)
    print(
        f"Found {len(all_paths)} unique referenced paths "
        f"({bad_lines} bad lines, {no_path_lines} lines without path)."
    )

    existing, missing = filter_existing(all_paths)
    if missing:
        print(f"Missing files: {len(missing)}")
        if args.fail_on_missing:
            for m in missing[:20]:
                print(f"  - {m}")
            if len(missing) > 20:
                print("  ...")
            print("Failing due to --fail-on-missing.", file=sys.stderr)
            return 3
    else:
        print("No missing files detected.")

    print(f"Creating zip: {args.output}")
    make_zip(existing, args.output, args.relbase)
    print(
        f"Done. Zipped {len(existing)} files into {args.output}. "
        f"Skipped {len(missing)} missing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


