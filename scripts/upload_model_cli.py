#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure we can import sibling script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from upload_g2 import upload_model_file  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a model file to OSS and print the resulting URL.",
    )
    parser.add_argument(
        "file",
        help="Path to the model file (e.g., .safetensors, .ckpt, .pt, .bin, .pth, .gguf, .sft)",
    )
    parser.add_argument(
        "--dir-uuid",
        dest="dir_uuid",
        default=None,
        help=(
            "Optional model directory UUID to group multiple files together. If omitted, "
            "a new UUID will be generated."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the resulting URL on success.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    file_path = os.path.abspath(args.file)

    if not os.path.exists(file_path):
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"Uploading model file: {file_path}")
        if args.dir_uuid:
            print(f"Using model directory UUID: {args.dir_uuid}")

    url, dir_uuid = upload_model_file(file_path, args.dir_uuid)

    if not url:
        if not args.quiet:
            print("Upload failed.", file=sys.stderr)
        return 1

    # Print URL as the only output if quiet, otherwise provide both
    print(url)
    if not args.quiet:
        print(f"Model directory UUID: {dir_uuid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


