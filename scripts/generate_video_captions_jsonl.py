import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Set, Dict


DEFAULT_EXTENSIONS: Set[str] = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".m4v",
}


def find_video_files(input_dir: Path, extensions: Set[str], recursive: bool) -> List[Path]:
    """Return a sorted list of video file Paths under input_dir.

    Args:
        input_dir: Directory to scan.
        extensions: Lowercased extensions to include, like {".mp4", ".mov"}.
        recursive: Whether to scan subdirectories.
    """
    if recursive:
        candidates: Iterable[Path] = input_dir.rglob("*")
    else:
        candidates = input_dir.glob("*")

    result: List[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix.lower() in extensions:
            result.append(path)

    # Sort deterministically: by directory then filename
    result.sort(key=lambda p: (str(p.parent), p.name))
    return result


def read_sidecar_caption(video_path: Path) -> str:
    """Read caption from a sidecar .txt file next to the video if present.

    The sidecar file is expected to have the same stem as the video, e.g.
    001.mp4 -> 001.txt. Newlines are collapsed to single spaces.
    Returns empty string if the file is missing or empty.
    """
    sidecar = video_path.with_suffix(".txt")
    if not sidecar.exists() or not sidecar.is_file():
        return ""
    try:
        text = sidecar.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    # Collapse multiple lines into a single line
    caption = " ".join([line.strip() for line in text.splitlines() if line.strip()])
    return caption.strip()


def write_jsonl_records(records: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a JSONL file mapping video paths to captions."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing videos (e.g., /mnt/cfs/jj/musubi-tuner/datasets/peel_it)",
    )
    parser.add_argument(
        "output_jsonl",
        type=Path,
        help="Output JSONL path (e.g., /mnt/cfs/jj/musubi-tuner/datasets/baby_bee/video_captions.jsonl)",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="",
        help="Caption text to assign to every video (default: empty string)",
    )
    parser.add_argument(
        "--ext",
        dest="extensions",
        action="append",
        default=None,
        help=(
            "Video file extension to include, e.g., --ext .mp4. "
            "Can be passed multiple times. Defaults to common video extensions."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning for videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir
    output_jsonl: Path = args.output_jsonl
    caption: str = args.caption

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found or not a directory: {input_dir}")

    extensions: Set[str] = (
        {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.extensions}
        if args.extensions
        else DEFAULT_EXTENSIONS
    )

    videos = find_video_files(input_dir, extensions, recursive=args.recursive)
    if not videos:
        print(
            f"No videos found in {input_dir} with extensions: {sorted(extensions)}",
            flush=True,
        )

    records: List[Dict[str, str]] = []
    missing_captions = 0

    if caption:
        for vp in videos:
            records.append({"video_path": str(vp.resolve()), "caption": caption})
    else:
        for vp in videos:
            cap = read_sidecar_caption(vp)
            if not cap:
                missing_captions += 1
            records.append({"video_path": str(vp.resolve()), "caption": cap})

    write_jsonl_records(records, output_jsonl)
    if caption:
        print(
            f"Wrote {len(records)} entries to {output_jsonl} (fixed caption)",
            flush=True,
        )
    else:
        print(
            f"Wrote {len(records)} entries to {output_jsonl} (sidecar captions, missing={missing_captions})",
            flush=True,
        )


if __name__ == "__main__":
    main()


