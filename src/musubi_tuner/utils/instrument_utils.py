"""
Instrument label extractor from Chinese surgical captions.
Extracts instrument type from surgical video captions for LoRA-MoE routing.

Usage:
    from musubi_tuner.utils.instrument_utils import extract_instrument_label, INSTRUMENT_KEYWORDS

    caption = "抓钳抓取组织。电凝钩切割。"
    label = extract_instrument_label(caption)  # Returns 1 (Hook/Electrocautery)
"""

import re
from typing import Dict, List, Tuple, Optional


# Instrument family mapping (must match LoRA-MoE expert order)
INSTRUMENT_FAMILIES = {
    0: "Scissors",            # 剪刀
    1: "Hook/Electrocautery", # 电凝钩/双极电凝
    2: "Suction",             # 吸引
    3: "Other",               # 抓钳/戳卡/其他
}

# Keywords for each instrument family (Chinese surgical terms)
INSTRUMENT_KEYWORDS = {
    0: [  # Scissors
        "剪刀", "剪切", "剪断", "剪开",
    ],
    1: [  # Hook/Electrocautery
        "电凝钩", "双极电凝", "电凝", "钩", "烧灼", "止血钳",
    ],
    2: [  # Suction
        "吸引", "吸", "吸出", "清理",
    ],
    3: [  # Other (Graspers, Pushrods, etc.)
        "抓钳", "戳卡", "镊子", "夹持", "固定钳", "分离钳",
        "牵开器", "撑开", "标本袋",
    ],
}


def extract_instrument_label(caption: str, default_label: int = 3) -> int:
    """
    Extract instrument label from Chinese surgical caption.

    Args:
        caption: Chinese surgical caption (e.g., "抓钳抓取组织。电凝钩切割。")
        default_label: Default label if no instrument found (default: 3 = Other)

    Returns:
        label: Instrument family ID (0-3)
            0 = Scissors
            1 = Hook/Electrocautery
            2 = Suction
            3 = Other

    Examples:
        >>> extract_instrument_label("抓钳抓取组织")
        3  # Other

        >>> extract_instrument_label("电凝钩切割血管")
        1  # Hook/Electrocautery

        >>> extract_instrument_label("吸引清理术野")
        2  # Suction

        >>> extract_instrument_label("剪刀剪断组织")
        0  # Scissors
    """
    # Count matches for each instrument family
    scores = {i: 0 for i in range(4)}

    for family_id, keywords in INSTRUMENT_KEYWORDS.items():
        for keyword in keywords:
            # Count occurrences of each keyword
            count = caption.count(keyword)
            scores[family_id] += count

    # Get family with highest score
    max_score = max(scores.values())

    # If no matches found, return default (Other)
    if max_score == 0:
        return default_label

    # Return family with highest score
    # If tie, prefer in order: Hook > Scissors > Suction > Other
    # (Hook is most distinctive, Other is catch-all)
    priority_order = [1, 0, 2, 3]
    for family_id in priority_order:
        if scores[family_id] == max_score:
            return family_id

    return default_label


def extract_instrument_logits(caption: str, temperature: float = 0.7) -> List[float]:
    """
    Extract soft instrument logits from caption for routing.

    Args:
        caption: Chinese surgical caption
        temperature: Softmax temperature (lower = more peaked)

    Returns:
        logits: [4] list of probabilities summing to 1.0

    Example:
        >>> extract_instrument_logits("抓钳抓取。电凝钩切割。")
        [0.0, 0.65, 0.0, 0.35]  # Mix of Hook (65%) and Other (35%)
    """
    import math

    # Count matches for each instrument family
    scores = [0.0] * 4

    for family_id, keywords in INSTRUMENT_KEYWORDS.items():
        for keyword in keywords:
            count = caption.count(keyword)
            scores[family_id] += count

    # If no matches, return uniform distribution
    if sum(scores) == 0:
        return [0.25] * 4

    # Apply softmax with temperature
    logits = []
    for score in scores:
        logits.append(math.exp(score / temperature))

    # Normalize
    total = sum(logits)
    logits = [l / total for l in logits]

    return logits


def extract_multi_instrument_labels(caption: str, threshold: float = 0.15) -> List[int]:
    """
    Extract multiple instrument labels from caption (for co-occurrence).

    Args:
        caption: Chinese surgical caption
        threshold: Minimum probability to include an instrument

    Returns:
        labels: List of instrument family IDs present in caption

    Example:
        >>> extract_multi_instrument_labels("抓钳固定。电凝钩切割。")
        [1, 3]  # Hook and Other both present
    """
    logits = extract_instrument_logits(caption)
    labels = [i for i, prob in enumerate(logits) if prob >= threshold]

    # Always return at least one label
    if not labels:
        labels = [extract_instrument_label(caption)]

    return labels


def parse_filtered_clips_jsonl(
    jsonl_path: str,
    add_instrument_labels: bool = True
) -> List[Dict]:
    """
    Parse filtered_clips_processed.jsonl and add instrument labels.

    Args:
        jsonl_path: Path to filtered_clips_processed.jsonl
        add_instrument_labels: Whether to add instrument_label field

    Returns:
        items: List of dicts with added "instrument_label" field

    Example output:
        [
            {
                "video_path": "cholec36_0137_3s.mp4",
                "caption": "抓钳抓取标本袋",
                "duration": 3,
                "source": "cholec36",
                "instrument_label": 3  # Other (Grasper)
            },
            ...
        ]
    """
    import json

    items = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())

            if add_instrument_labels:
                # Extract instrument label from caption
                caption = item.get("caption", "")
                instrument_label = extract_instrument_label(caption)
                item["instrument_label"] = instrument_label

                # Also add logits for soft routing
                item["instrument_logits"] = extract_instrument_logits(caption)

            items.append(item)

    return items


def augment_dataset_config_with_instruments(
    config_toml_path: str,
    filtered_clips_jsonl_path: str,
    output_jsonl_path: Optional[str] = None
) -> str:
    """
    Augment dataset config with instrument labels.

    Args:
        config_toml_path: Path to dataset config TOML
        filtered_clips_jsonl_path: Path to filtered_clips_processed.jsonl
        output_jsonl_path: Where to save augmented JSONL (optional)

    Returns:
        output_path: Path to augmented JSONL file

    This function:
    1. Reads filtered_clips_processed.jsonl
    2. Extracts instrument labels from captions
    3. Saves augmented JSONL with instrument_label field
    4. Returns path for use in training
    """
    import os

    # Parse and add instrument labels
    items = parse_filtered_clips_jsonl(
        filtered_clips_jsonl_path,
        add_instrument_labels=True
    )

    # Determine output path
    if output_jsonl_path is None:
        base_dir = os.path.dirname(filtered_clips_jsonl_path)
        output_jsonl_path = os.path.join(
            base_dir,
            "filtered_clips_with_instruments.jsonl"
        )

    # Save augmented JSONL
    import json
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved {len(items)} items with instrument labels to {output_jsonl_path}")

    # Print statistics
    from collections import Counter
    label_counts = Counter(item["instrument_label"] for item in items)
    print("\nInstrument distribution:")
    for label in sorted(label_counts.keys()):
        name = INSTRUMENT_FAMILIES[label]
        count = label_counts[label]
        pct = 100.0 * count / len(items)
        print(f"  {label} ({name}): {count} ({pct:.1f}%)")

    return output_jsonl_path


# For testing
if __name__ == "__main__":
    # Test extraction
    test_captions = [
        "抓钳抓取组织",
        "电凝钩切割血管",
        "吸引清理术野",
        "剪刀剪断组织",
        "抓钳固定。电凝钩切割。",
        "超声刀分离组织",  # Should map to Other
    ]

    print("Testing instrument extraction:")
    print("-" * 60)
    for caption in test_captions:
        label = extract_instrument_label(caption)
        logits = extract_instrument_logits(caption)
        multi = extract_multi_instrument_labels(caption)

        print(f"Caption: {caption}")
        print(f"  Label: {label} ({INSTRUMENT_FAMILIES[label]})")
        print(f"  Logits: {[f'{l:.2f}' for l in logits]}")
        print(f"  Multi: {multi} ({[INSTRUMENT_FAMILIES[i] for i in multi]})")
        print()
