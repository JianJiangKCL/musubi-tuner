#!/usr/bin/env python3
"""
可视化实际的帧采样过程
基于peel_it_i2v_meta_trace21.toml配置
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple

# 从配置文件读取的参数
TARGET_FRAMES = [17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81]
FRAME_EXTRACTION = "uniform"
FRAME_SAMPLE = 2  # 从配置中读取

def simulate_video_frames(total_frames: int = 81) -> List[int]:
    """模拟一个视频的所有帧"""
    return list(range(1, total_frames + 1))

def uniform_sampling_simulation(video_frames: List[int], target_frame: int, frame_sample: int) -> List[List[int]]:
    """
    模拟均匀采样过程
    返回采样出来的帧列表
    """
    frame_count = len(video_frames)
    sampled_sequences = []

    # 计算采样位置 (从训练代码复制的逻辑)
    frame_indices = np.linspace(0, frame_count - target_frame, frame_sample, dtype=int)

    print(f"\n🎯 Target Frame: {target_frame}")
    print(f"   采样位置索引: {frame_indices}")
    print(f"   实际帧范围: {[f'[{i+1}:{i+target_frame}]' for i in frame_indices]}")

    for i in frame_indices:
        # 裁剪出 target_frame 帧
        start_idx = i
        end_idx = i + target_frame
        sampled_frames = video_frames[start_idx:end_idx]
        sampled_sequences.append(sampled_frames)

        print(f"   采样序列 {len(sampled_sequences)}: 帧 {sampled_frames[0]}-{sampled_frames[-1]} ({len(sampled_frames)}帧)")

    return sampled_sequences

def visualize_single_target_frame(video_frames: List[int], target_frame: int, frame_sample: int):
    """可视化单个target_frame的采样结果"""
    fig, ax = plt.subplots(figsize=(15, 6))

    # 绘制所有帧
    all_frames = np.arange(1, len(video_frames) + 1)
    ax.scatter(all_frames, np.ones_like(all_frames) * 2, c='lightgray', s=30, alpha=0.6, label='未采样帧')

    # 计算并绘制采样序列
    frame_indices = np.linspace(0, len(video_frames) - target_frame, frame_sample, dtype=int)

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for idx, i in enumerate(frame_indices):
        start_frame = i + 1
        end_frame = i + target_frame

        # 绘制采样帧
        sampled_frames = np.arange(start_frame, end_frame + 1)
        ax.scatter(sampled_frames, np.ones_like(sampled_frames) * 1, c=colors[idx % len(colors)],
                  s=60, marker='s', alpha=0.8, label=f'采样序列 {idx+1}')

        # 添加文本标签
        ax.text(start_frame + target_frame/2, 1.1, f'{start_frame}-{end_frame}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlim(0, len(video_frames) + 2)
    ax.set_ylim(0.5, 2.5)
    ax.set_xlabel('帧号', fontsize=12)
    ax.set_title(f'均匀采样可视化 - Target Frame: {target_frame}', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['采样帧', '全部帧'])

    plt.tight_layout()
    return fig

def create_comprehensive_visualization():
    """创建综合可视化"""
    print("🎬 实际帧采样过程可视化")
    print("=" * 60)
    print(f"配置参数:")
    print(f"  - frame_extraction: {FRAME_EXTRACTION}")
    print(f"  - frame_sample: {FRAME_SAMPLE}")
    print(f"  - target_frames: {TARGET_FRAMES}")
    print()

    # 模拟一个81帧的视频
    video_frames = simulate_video_frames(81)
    print(f"模拟视频总帧数: {len(video_frames)}")

    # 为每个target_frame创建可视化
    os.makedirs('/mnt/cfs/jj/musubi-tuner/sampled_bucket', exist_ok=True)

    for target_frame in TARGET_FRAMES:
        print(f"\n{'='*50}")
        print(f"处理 Target Frame: {target_frame}")

        # 模拟采样过程
        sampled_sequences = uniform_sampling_simulation(video_frames, target_frame, FRAME_SAMPLE)

        # 创建可视化
        fig = visualize_single_target_frame(video_frames, target_frame, FRAME_SAMPLE)
        output_path = f'/mnt/cfs/jj/musubi-tuner/sampled_bucket/sampling_{target_frame}frames.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"   📊 可视化已保存: sampling_{target_frame}frames.png")

def create_summary_table():
    """创建采样摘要表格"""
    print("\n📋 采样策略汇总表")
    print("=" * 80)
    print("2d")
    print("-" * 80)

    video_frames = simulate_video_frames(81)

    for target_frame in TARGET_FRAMES:
        frame_indices = np.linspace(0, len(video_frames) - target_frame, FRAME_SAMPLE, dtype=int)
        sample_ranges = []
        for i in frame_indices:
            start = i + 1
            end = i + target_frame
            sample_ranges.append(f"{start}-{end}")

        ranges_str = ", ".join(sample_ranges)
        print("2d")

def main():
    """主函数"""
    create_comprehensive_visualization()
    create_summary_table()

    print("\n" + "=" * 80)
    print("✅ 实际采样可视化完成！")
    print("输出文件:")
    print("  - sampling_{target_frame}frames.png: 各target_frame的采样可视化")
    print("位置: /mnt/cfs/jj/musubi-tuner/sampled_bucket/")
    print("\n说明:")
    print("- 每个PNG文件展示了对应target_frame的实际采样结果")
    print("- 红色/蓝色方块表示被采样出来的帧序列")
    print("- 灰色圆点表示视频中的全部帧")
    print("- 每个target_frame会生成2个采样序列（frame_sample=2）")

if __name__ == "__main__":
    main()









