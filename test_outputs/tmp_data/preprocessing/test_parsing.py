#!/usr/bin/env python3
"""
Test script to verify timestamp parsing logic.
"""

import re


def parse_timestamp(timestamp_str):
    """Convert timestamp string (MM'SS) to total seconds."""
    if "'" not in timestamp_str:
        return None

    parts = timestamp_str.split("'")
    if len(parts) != 2:
        return None

    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    except ValueError:
        return None


def parse_line(line):
    """Parse a single line from the annotation file."""
    line = line.strip()
    if not line:
        return None

    # Find all timestamp patterns in the line
    timestamp_pattern = r"(\d+'?\d+)"
    timestamps = re.findall(timestamp_pattern, line)

    if len(timestamps) < 2:
        # Single timestamp or no timestamps - ignore
        return None

    # Extract action description (everything before the first timestamp)
    # Look for the first occurrence of a timestamp pattern
    first_timestamp_match = re.search(timestamp_pattern, line)
    if not first_timestamp_match:
        return None

    action_desc = line[:first_timestamp_match.start()].strip()

    # Remove leading spaces and special prefixes
    action_desc = re.sub(r'^(state|reason|constant)\s+', '', action_desc)

    if not action_desc:
        return None

    # Parse all timestamp pairs
    time_ranges = []
    i = 0
    while i < len(timestamps) - 1:
        start_time = parse_timestamp(timestamps[i])
        end_time = parse_timestamp(timestamps[i + 1])

        if start_time is not None and end_time is not None and start_time < end_time:
            time_ranges.append((start_time, end_time))
            i += 2  # Move to next pair
        else:
            i += 1  # Skip invalid pair

    if not time_ranges:
        return None

    return {
        'action': action_desc,
        'time_ranges': time_ranges
    }


def test_parsing():
    """Test the parsing logic with sample lines."""
    test_cases = [
        # Valid cases
        "抓钳B向左下方牵拉胆囊 11'23 12'30",
        "电凝钩开始沿胆囊床逆行剥离胆囊 20'20 21'07 - 21'14 21'51",
        "state 电凝钩分离组织过程中偶尔产生了大量的烟雾 10'29 11'01",

        # Invalid cases (should be ignored)
        "钛夹出现 18'08",
        "戳卡a进入腹腔 0'47",
        "",
        "no timestamps here",
    ]

    print("Testing parsing logic:\n")

    for i, line in enumerate(test_cases, 1):
        print(f"Test case {i}: {line}")
        result = parse_line(line)

        if result:
            print(f"  ✓ Parsed: {result['action']}")
            print(f"  ✓ Time ranges: {result['time_ranges']}")
        else:
            print("  ✗ Ignored (no valid time ranges)")
        print()


def test_with_real_file():
    """Test with the actual annotation file."""
    annotation_file = "/mnt/cfs/jj/musubi-tuner/tmp_data/myanno-cholec36.txt"

    print(f"Testing with real annotation file: {annotation_file}\n")

    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        valid_actions = 0
        total_ranges = 0

        for line_num, line in enumerate(lines[:20], 1):  # Test first 20 lines
            result = parse_line(line)
            if result:
                valid_actions += 1
                total_ranges += len(result['time_ranges'])
                print(f"Line {line_num}: ✓ {result['action']} ({len(result['time_ranges'])} ranges)")

        print("\nSummary:")
        print(f"  Valid actions found: {valid_actions}")
        print(f"  Total time ranges: {total_ranges}")

    except FileNotFoundError:
        print("Annotation file not found for testing")


if __name__ == '__main__':
    test_parsing()
    test_with_real_file()
