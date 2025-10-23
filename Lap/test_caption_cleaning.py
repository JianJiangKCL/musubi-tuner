#!/usr/bin/env python3
"""Test script to demonstrate caption cleaning."""

import re


def clean_caption(caption):
    """Clean caption text by removing tool labels and bracket content.
    
    - Removes [Tools in: ...; Tools observed: ...] content
    - Replaces 抓钳A/B/C/0/1/2 with 抓钳
    - Removes any other content in square brackets
    """
    if not caption:
        return caption
    
    # Remove content in square brackets (e.g., [Tools in: ...; Tools observed: ...])
    caption = re.sub(r'\s*\[.*?\]\s*', '', caption)
    
    # Replace 抓钳A/B/C/D/0/1/2/etc with 抓钳
    caption = re.sub(r'抓钳[A-Za-z0-9]', '抓钳', caption)
    
    # Remove extra spaces
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    return caption


# Test cases
test_cases = [
    "抓钳A向上牵拉胆囊",
    "抓钳B向左下方牵拉胆囊周围组织",
    "抓钳C向上牵拉胆囊，并阻挡肝脏",
    "抓钳0出现",
    "抓钳1向左上方牵拉胆囊",
    "抓钳A和抓钳B调整位置",
    "抓钳A和抓钳B来回牵拉大网膜和肠管 [Tools in: 抓钳A, 抓钳B, 电凝钩; Tools observed: 抓钳B, 电凝钩]",
    "抓钳B向下牵拉网膜组织 [Tools in: 抓钳A]",
    "电凝钩分离胆囊及周边组织",
    "腔镜进入体内",
]

print("Caption Cleaning Test Results:")
print("=" * 80)

for i, original in enumerate(test_cases, 1):
    cleaned = clean_caption(original)
    print(f"\n{i}. Original: {original}")
    print(f"   Cleaned:  {cleaned}")

print("\n" + "=" * 80)

