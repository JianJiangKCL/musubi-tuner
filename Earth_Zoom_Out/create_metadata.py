#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建video_captions.jsonl元数据文件
将视频文件与对应的txt描述文件配对

使用方法：
python3 create_metadata.py [数据集目录] [输出文件路径]

示例：
python3 create_metadata.py /mnt/cfs/jj/musubi-tuner/Earth_Zoom_Out/dataset /mnt/cfs/jj/musubi-tuner/Earth_Zoom_Out/video_captions.jsonl
"""

import os
import json
import sys

def create_video_captions(dataset_dir, output_path):
    # 确保数据集目录存在
    if not os.path.exists(dataset_dir):
        print(f"❌ 错误: 数据集目录不存在: {dataset_dir}")
        return False
    
    # 获取数据集目录的绝对路径
    dataset_dir = os.path.abspath(dataset_dir)
    print(f"📁 数据集目录: {dataset_dir}")
    
    # 获取所有mp4文件
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith('.mp4')]
    video_files.sort()
    
    print(f"🎬 找到 {len(video_files)} 个视频文件")
    
    if len(video_files) == 0:
        print("❌ 错误: 未找到任何mp4文件")
        return False
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")
    
    # 创建jsonl文件
    with open(output_path, 'w', encoding='utf-8') as jsonl_file:
        for video_file in video_files:
            # 构造对应的txt文件名（在数据集目录中）
            txt_file = os.path.join(dataset_dir, video_file.replace('.mp4', '.txt'))
            # 视频文件的绝对路径
            video_absolute_path = os.path.join(dataset_dir, video_file)
            
            if os.path.exists(txt_file):
                # 读取描述文本
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    # 将换行符替换为空格，清理多余空格
                    text = ' '.join(text.split())
                
                # 创建JSON记录（使用绝对路径，字段名为video_path）
                record = {'video_path': video_absolute_path, 'caption': text}
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(f'✓ 已处理: {video_file} + {os.path.basename(txt_file)}')
            else:
                print(f'⚠ 警告: 找不到对应的描述文件 {txt_file}')
    
    print(f'\n✅ 元数据文件创建完成！输出路径: {output_path}')
    
    # 验证结果
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"📊 生成了 {len(lines)} 条记录")
        
        # 显示前3条记录作为示例
        print("\n📋 前3条记录示例:")
        for i, line in enumerate(lines[:3]):
            record = json.loads(line.strip())
            print(f"{i+1}. 视频: {record['video_path']}")
            print(f"   描述: {record['caption'][:100]}...")
            print()
        
        return True
    else:
        print("❌ 错误: 输出文件创建失败")
        return False

def main():
    # 解析命令行参数
    if len(sys.argv) == 3:
        dataset_dir = sys.argv[1]
        output_path = sys.argv[2]
    elif len(sys.argv) == 1:
        # 默认参数（兼容旧版本）
        dataset_dir = "/mnt/cfs/jj/musubi-tuner/Earth_Zoom_Out/dataset"
        output_path = "/mnt/cfs/jj/musubi-tuner/Earth_Zoom_Out/video_captions.jsonl"
        print("📝 使用默认参数:")
        print(f"   数据集目录: {dataset_dir}")
        print(f"   输出文件: {output_path}")
    else:
        print("❌ 用法错误!")
        print("使用方法:")
        print("  python3 create_metadata.py [数据集目录] [输出文件路径]")
        print()
        print("示例:")
        print("  python3 create_metadata.py /mnt/cfs/jj/musubi-tuner/Earth_Zoom_Out/dataset /mnt/cfs/jj/musubi-tuner/Earth_Zoom_Out/video_captions.jsonl")
        print()
        print("或直接运行（使用默认参数）:")
        print("  python3 create_metadata.py")
        sys.exit(1)
    
    # 执行主函数
    success = create_video_captions(dataset_dir, output_path)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
