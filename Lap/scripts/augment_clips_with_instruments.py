#!/usr/bin/env python3
"""
Augment filtered_clips_processed.jsonl with instrument labels extracted from captions.

Usage:
    python augment_clips_with_instruments.py \\
        --input Lap/preprocessing/filtered_clips_processed.jsonl \\
        --output Lap/preprocessing/filtered_clips_with_instruments.jsonl
"""

import argparse
import json
import sys
import os

# Ensure project src is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
src_dir = os.path.join(repo_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from musubi_tuner.utils.instrument_utils import (
    extract_instrument_label,
    extract_instrument_logits,
    INSTRUMENT_FAMILIES
)


def main():
    parser = argparse.ArgumentParser(description="Augment clips with instrument labels")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to filtered_clips_processed.jsonl"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: same dir with _with_instruments suffix)"
    )
    parser.add_argument(
        "--soft_labels",
        action="store_true",
        help="Add soft logits in addition to hard labels"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        base = args.input.replace('.jsonl', '')
        args.output = f"{base}_with_instruments.jsonl"

    # Process clips
    print(f"Reading clips from {args.input}")
    items = []
    label_counts = {i: 0 for i in range(4)}

    with open(args.input, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON at line {line_num}")
                continue

            caption = item.get("caption", "")

            # Extract instrument label
            instrument_label = extract_instrument_label(caption)
            item["instrument_label"] = instrument_label
            label_counts[instrument_label] += 1

            # Optionally add soft logits
            if args.soft_labels:
                instrument_logits = extract_instrument_logits(caption)
                item["instrument_logits"] = instrument_logits

            items.append(item)

    # Save augmented clips
    print(f"\nWriting {len(items)} augmented clips to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Print statistics
    print("\nInstrument Label Distribution:")
    print("=" * 60)
    for label in sorted(label_counts.keys()):
        name = INSTRUMENT_FAMILIES[label]
        count = label_counts[label]
        pct = 100.0 * count / len(items) if len(items) > 0 else 0
        print(f"  {label} - {name:25s}: {count:4d} ({pct:5.1f}%)")

    print(f"\nTotal clips: {len(items)}")
    print(f"Output saved to: {args.output}")

    # Check for imbalance
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    if max_count > 0 and min_count > 0:
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 3.0:
            print(f"\n⚠️  Warning: Class imbalance detected (ratio: {imbalance_ratio:.1f})")
            print("   Consider using balanced sampling during training")
            print("   or adjusting routing regularization weights")

    return args.output


if __name__ == "__main__":
    output_path = main()
