#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建CSV格式的元数据文件
将视频文件与对应的txt描述文件配对，并生成CSV格式的元数据

支持两种使用方式：

1. 批量处理模式（推荐）:
   python3 create_csv_metadata.py [数据集父目录]
   为父目录下的每个子文件夹生成：
   - metadata.csv文件（放在各自的数据集目录下）

2. 单数据集处理模式:
   python3 create_csv_metadata.py [数据集目录] [输出CSV文件路径]
   只生成指定的metadata.csv文件

示例：
# 批量处理（推荐）
python3 create_csv_metadata.py /path/to/datasets/
# 会为 datasets/ 下的每个子文件夹生成对应的metadata.csv文件

# 单数据集处理
python3 create_csv_metadata.py /path/to/dataset /path/to/metadata.csv
"""

import os
import csv
import sys


def create_csv_metadata(dataset_dir, output_path):
    """为单个数据集目录创建metadata.csv文件"""
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
    
    # 创建CSV文件
    processed_count = 0
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        
        # 写入表头
        writer.writerow(['video', 'prompt'])
        
        for video_file in video_files:
            # 构造对应的txt文件名（在数据集目录中）
            txt_file = os.path.join(dataset_dir, video_file.replace('.mp4', '.txt'))
            
            if os.path.exists(txt_file):
                # 读取描述文本
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    # 将换行符替换为空格，清理多余空格
                    text = ' '.join(text.split())
                
                # 写入CSV记录（video列仅使用文件名，不包含路径）
                writer.writerow([video_file, text])
                processed_count += 1
                print(f'✓ 已处理: {video_file} + {os.path.basename(txt_file)}')
            else:
                print(f'⚠ 警告: 找不到对应的描述文件 {txt_file}')
    
    print(f'\n✅ CSV元数据文件创建完成！输出路径: {output_path}')
    
    # 验证结果
    if os.path.exists(output_path):
        print(f"📊 成功处理了 {processed_count} 条记录")
        
        # 显示前3条记录作为示例
        print("\n📋 前3条记录示例:")
        with open(output_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 跳过表头
            for i, row in enumerate(reader):
                if i >= 3:
                    break
                if len(row) >= 2:
                    print(f"{i+1}. 视频: {row[0]}")
                    print(f"   描述: {row[1][:100]}...")
                    print()
        
        return True
    else:
        print("❌ 错误: 输出文件创建失败")
        return False


def batch_create_csv_metadata(datasets_path):
    """批量处理多个数据集文件夹，为每个子文件夹创建metadata.csv文件"""
    # 确保父文件夹存在
    if not os.path.exists(datasets_path):
        print(f"❌ 错误: 数据集父目录不存在: {datasets_path}")
        return False
    
    # 获取绝对路径
    datasets_path = os.path.abspath(datasets_path)
    print(f"📁 数据集父目录: {datasets_path}")
    
    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(datasets_path) 
               if os.path.isdir(os.path.join(datasets_path, d))]
    subdirs.sort()
    
    print(f"📂 找到 {len(subdirs)} 个子文件夹:")
    for i, subdir in enumerate(subdirs, 1):
        print(f"  {i}. {subdir}")
    
    if len(subdirs) == 0:
        print("❌ 错误: 未找到任何子文件夹")
        return False
    
    print(f"\n🚀 开始批量处理...")
    
    success_count = 0
    failed_count = 0
    
    for i, subdir in enumerate(subdirs, 1):
        print(f"\n{'='*60}")
        print(f"处理第 {i}/{len(subdirs)} 个数据集: {subdir}")
        print(f"{'='*60}")
        
        # 构造子文件夹路径
        dataset_dir = os.path.join(datasets_path, subdir)
        # 输出文件路径：子文件夹路径/metadata.csv
        output_path = os.path.join(dataset_dir, "metadata.csv")
        
        try:
            # 调用单数据集处理逻辑创建CSV
            success = create_csv_metadata(dataset_dir, output_path)
            if success:
                success_count += 1
                print(f"✅ {subdir} CSV处理成功")
            else:
                failed_count += 1
                print(f"❌ {subdir} CSV处理失败")
        except Exception as e:
            failed_count += 1
            print(f"❌ {subdir} 处理异常: {str(e)}")
    
    # 输出最终统计
    print(f"\n{'='*60}")
    print(f"📊 批量处理完成!")
    print(f"{'='*60}")
    print(f"总数据集: {len(subdirs)} 个")
    print(f"CSV成功处理: {success_count} 个")
    print(f"处理失败: {failed_count} 个")
    
    if success_count > 0:
        print(f"\n✅ 成功处理的数据集文件:")
        for subdir in subdirs:
            csv_path = os.path.join(datasets_path, subdir, "metadata.csv")
            if os.path.exists(csv_path):
                print(f"  📄 CSV: {csv_path}")
    
    return success_count > 0


def print_help():
    """打印帮助信息"""
    print("📖 Create CSV Metadata - 视频数据集CSV元数据文件生成器")
    print("=" * 60)
    print()
    print("功能：")
    print("- 为视频数据集生成CSV格式的元数据文件")
    print("- 支持批量处理多个数据集")
    print("- CSV格式包含两列：video（文件名）和prompt（描述）")
    print()
    print("使用方法：")
    print()
    print("1. 批量处理模式（推荐）:")
    print("   python3 create_csv_metadata.py [数据集父目录]")
    print("   为父目录下的每个子文件夹生成：")
    print("   - metadata.csv文件（放在各自的数据集目录下）")
    print()
    print("2. 单数据集处理模式:")
    print("   python3 create_csv_metadata.py [数据集目录] [输出CSV文件路径]")
    print("   只生成指定的metadata.csv文件")
    print()
    print("示例：")
    print("   # 批量处理")
    print("   python3 create_csv_metadata.py /path/to/datasets/")
    print()
    print("   # 单数据集处理")
    print("   python3 create_csv_metadata.py /path/to/dataset /path/to/metadata.csv")
    print()
    print("   # 显示帮助")
    print("   python3 create_csv_metadata.py --help")
    print()
    print("输出格式示例：")
    print("   video,prompt")
    print("   01.mp4,\"描述内容...\"")
    print("   02.mp4,\"描述内容...\"")


def main():
    # 解析命令行参数
    if len(sys.argv) == 2:
        # 检查是否为帮助参数
        if sys.argv[1] in ['-h', '--help', 'help']:
            print_help()
            return
        
        # 批量处理模式
        datasets_path = sys.argv[1]
        print("🔄 批量处理模式:")
        print(f"   数据集父目录: {datasets_path}")
        print(f"   输出文件格式: [子文件夹]/metadata.csv")
        success = batch_create_csv_metadata(datasets_path)
    elif len(sys.argv) == 3:
        # 单数据集处理模式
        dataset_dir = sys.argv[1]
        output_path = sys.argv[2]
        print("📝 单数据集处理模式:")
        print(f"   数据集目录: {dataset_dir}")
        print(f"   输出文件: {output_path}")
        success = create_csv_metadata(dataset_dir, output_path)
    else:
        print("❌ 用法错误!")
        print("使用方法:")
        print()
        print("1. 批量处理模式（推荐）:")
        print("   python3 create_csv_metadata.py [数据集父目录]")
        print("   为父目录下的每个子文件夹生成metadata.csv文件")
        print()
        print("2. 单数据集处理模式:")
        print("   python3 create_csv_metadata.py [数据集目录] [输出CSV文件路径]")
        print("   只生成指定的metadata.csv文件")
        print()
        print("示例:")
        print("   # 批量处理")
        print("   python3 create_csv_metadata.py /path/to/datasets/")
        print()
        print("   # 单数据集处理")
        print("   python3 create_csv_metadata.py /path/to/dataset /path/to/metadata.csv")
        print()
        print("或显示帮助:")
        print("   python3 create_csv_metadata.py --help")
        sys.exit(1)
    
    # 执行结果检查
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
