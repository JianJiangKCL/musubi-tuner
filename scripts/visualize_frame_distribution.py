#!/usr/bin/env python3
"""
可视化target_frames分布和采样策略
基于peel_it_i2v_meta_trace21.toml配置
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List
import os

# 从配置文件读取的target_frames
TARGET_FRAMES = [17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81]

def visualize_frame_distribution():
    """可视化帧数分布"""
    print("🎬 帧数分布可视化")
    print("=" * 50)

    # 基本统计
    print(f"总帧数种类: {len(TARGET_FRAMES)}")
    print(f"最小帧数: {min(TARGET_FRAMES)}")
    print(f"最大帧数: {max(TARGET_FRAMES)}")
    print(f"平均帧数: {np.mean(TARGET_FRAMES):.1f}")
    print(f"中位帧数: {np.median(TARGET_FRAMES)}")

    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 帧数分布直方图
    ax1.bar(range(len(TARGET_FRAMES)), TARGET_FRAMES, color='skyblue', alpha=0.7)
    ax1.set_title('帧数分布直方图')
    ax1.set_xlabel('索引')
    ax1.set_ylabel('帧数')
    ax1.grid(True, alpha=0.3)

    # 添加数值标签
    for i, v in enumerate(TARGET_FRAMES):
        ax1.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=8)

    # 2. 帧数与时长的关系 (16fps)
    durations = [f/16 for f in TARGET_FRAMES]
    ax2.plot(TARGET_FRAMES, durations, 'ro-', linewidth=2, markersize=6)
    ax2.set_title('帧数与视频时长关系 (16fps)')
    ax2.set_xlabel('帧数')
    ax2.set_ylabel('时长(秒)')
    ax2.grid(True, alpha=0.3)

    # 添加时长标签
    for x, y in zip(TARGET_FRAMES, durations):
        ax2.text(x, y + 0.05, f'{y:.2f}', ha='center', va='bottom', fontsize=8)

    # 3. 帧数间隔分布
    intervals = [TARGET_FRAMES[i+1] - TARGET_FRAMES[i] for i in range(len(TARGET_FRAMES)-1)]
    ax3.bar(range(len(intervals)), intervals, color='lightgreen', alpha=0.7)
    ax3.set_title('相邻帧数间隔')
    ax3.set_xlabel('间隔索引')
    ax3.set_ylabel('帧数差')
    ax3.grid(True, alpha=0.3)

    # 添加间隔标签
    for i, v in enumerate(intervals):
        ax3.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=8)

    # 4. 累积分布
    ax4.plot(TARGET_FRAMES, np.arange(1, len(TARGET_FRAMES)+1), 'bo-', linewidth=2, markersize=6)
    ax4.set_title('累积帧数分布')
    ax4.set_xlabel('帧数')
    ax4.set_ylabel('累积数量')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/mnt/cfs/jj/musubi-tuner/sampled_bucket/frame_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_sampling_strategy():
    """可视化采样策略"""
    print("\n🔍 采样策略分析")
    print("=" * 50)

    # 分析帧数模式
    print("帧数模式分析:")
    print(f"- 起始帧数: {TARGET_FRAMES[0]}")
    print(f"- 结束帧数: {TARGET_FRAMES[-1]}")
    print(f"- 总跨度: {TARGET_FRAMES[-1] - TARGET_FRAMES[0]} 帧")

    # 计算间隔统计
    intervals = [TARGET_FRAMES[i+1] - TARGET_FRAMES[i] for i in range(len(TARGET_FRAMES)-1)]
    print(f"- 平均间隔: {np.mean(intervals):.1f} 帧")
    print(f"- 最长间隔: {max(intervals)} 帧")
    print(f"- 最短间隔: {min(intervals)} 帧")

    # 分析分布特征
    print("\n分布特征:")
    print("- 短视频段(≤30帧):", [f for f in TARGET_FRAMES if f <= 30])
    print("- 中视频段(31-60帧):", [f for f in TARGET_FRAMES if 31 <= f <= 60])
    print("- 长视频段(≥61帧):", [f for f in TARGET_FRAMES if f >= 61])

    # 时长分布
    durations = [f/16 for f in TARGET_FRAMES]
    print("\n时长分布(16fps):")
    for frames, duration in zip(TARGET_FRAMES, durations):
        print(f"{frames:2d}帧 -> {duration:.2f}秒")

def create_training_config_summary():
    """创建训练配置摘要"""
    print("\n📋 训练配置摘要")
    print("=" * 50)

    # 需要先计算durations
    durations_for_summary = [f/16 for f in TARGET_FRAMES]
    config_summary = {
        "数据集": "peel_it",
        "target_frames": TARGET_FRAMES,
        "总帧数种类": len(TARGET_FRAMES),
        "frame_extraction": "uniform",
        "frame_sample": 2,
        "预期fps": 16,
        "时长范围": f"{min(durations_for_summary):.2f}-{max(durations_for_summary):.2f}秒"
    }

    for key, value in config_summary.items():
        print(f"{key}: {value}")

def visualize_uniform_sampling():
    """可视化均匀采样策略"""
    print("\n🎯 均匀采样策略可视化")
    print("=" * 50)

    # 模拟一个81帧的视频如何被均匀采样
    total_frames = 81
    sample_stride = 2  # 从配置中读取

    print(f"以{sample_stride}帧为步长对{total_frames}帧视频进行均匀采样:")

    # 计算采样点
    sampled_frames = []
    for i in range(0, total_frames, sample_stride):
        sampled_frames.append(i + 1)  # +1因为帧数从1开始

    print(f"采样帧: {sampled_frames}")
    print(f"采样数量: {len(sampled_frames)}")
    print(f"覆盖率: {len(sampled_frames)/total_frames*100:.1f}%")
    # 可视化采样模式
    fig, ax = plt.subplots(figsize=(12, 4))

    # 绘制所有帧
    all_frames = np.arange(1, total_frames + 1)
    ax.scatter(all_frames, np.ones_like(all_frames), c='lightgray', s=10, alpha=0.5, label='未采样帧')

    # 绘制采样帧
    ax.scatter(sampled_frames, np.ones(len(sampled_frames)), c='red', s=50, marker='x', label='采样帧')

    ax.set_xlim(0, total_frames + 1)
    ax.set_ylim(0.8, 1.2)
    ax.set_xlabel('帧号')
    ax.set_title(f'均匀采样可视化 (步长={sample_stride})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yticks([])  # 隐藏y轴刻度

    plt.tight_layout()
    plt.savefig('/mnt/cfs/jj/musubi-tuner/sampled_bucket/uniform_sampling.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    # 确保输出目录存在
    os.makedirs('/mnt/cfs/jj/musubi-tuner/sampled_bucket', exist_ok=True)

    print("🎬 Target Frames 采样策略可视化工具")
    print("=" * 60)
    print(f"配置: {TARGET_FRAMES}")

    # 运行所有可视化
    visualize_frame_distribution()
    visualize_sampling_strategy()
    create_training_config_summary()
    visualize_uniform_sampling()

    print("\n" + "=" * 60)
    print("✅ 可视化完成！")
    print("输出文件:")
    print("  - frame_distribution.png: 帧数分布图表")
    print("  - uniform_sampling.png: 均匀采样策略图")
    print(f"位置: /mnt/cfs/jj/musubi-tuner/sampled_bucket/")

if __name__ == "__main__":
    main()
