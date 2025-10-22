#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建video_captions.jsonl元数据文件和TOML配置文件
将视频文件与对应的txt描述文件配对，并生成训练配置

支持三种使用方式：

1. 批量处理模式（使用默认配置目录）:
   python3 create_metadata.py [数据集父目录]
   为父目录下的每个子文件夹生成：
   - video_captions.jsonl文件（放在各自的数据集目录下）
   - 对应的.toml配置文件（放在默认配置目录下）

2. 批量处理模式（自定义配置目录）:
   python3 create_metadata.py [数据集父目录] [配置输出目录]
   为父目录下的每个子文件夹生成：
   - video_captions.jsonl文件（放在各自的数据集目录下）
   - 对应的.toml配置文件（放在指定的配置目录下）

3. 单数据集处理模式:
   python3 create_metadata.py [数据集目录] [输出文件路径.jsonl]
   生成指定的video_captions.jsonl文件和对应的.toml配置文件（保存到默认config目录）

示例：
# 批量处理（使用默认配置目录）
python3 create_metadata.py /mnt/cfs/jj/musubi-tuner/Chaowei/datasets/

# 批量处理（自定义配置目录）
python3 create_metadata.py /mnt/cfs/jj/musubi-tuner/Chaowei/datasets/ /custom/config/path/

# 单数据集处理
python3 create_metadata.py /mnt/cfs/jj/musubi-tuner/Earth_Zoom_Out/dataset /mnt/cfs/jj/musubi-tuner/Earth_Zoom_Out/video_captions.jsonl
"""

import os
import json
import sys

def create_video_captions(dataset_dir, output_path):
    """为单个数据集目录创建video_captions.jsonl文件"""
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


def batch_create_video_captions(datasets_path, config_output_dir=None):
    """批量处理多个数据集文件夹，为每个子文件夹创建video_captions.jsonl和toml配置文件"""
    # 确保父文件夹存在
    if not os.path.exists(datasets_path):
        print(f"❌ 错误: 数据集父目录不存在: {datasets_path}")
        return False
    
    # 获取绝对路径
    datasets_path = os.path.abspath(datasets_path)
    print(f"📁 数据集父目录: {datasets_path}")
    
    # 配置toml输出目录 - 支持参数配置
    if config_output_dir is None:
        config_output_dir = "/mnt/cfs/jj/musubi-tuner/Chaowei/config"
    config_output_dir = os.path.abspath(config_output_dir)
    print(f"📁 TOML配置输出目录: {config_output_dir}")
    
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
    toml_success_count = 0
    
    for i, subdir in enumerate(subdirs, 1):
        print(f"\n{'='*60}")
        print(f"处理第 {i}/{len(subdirs)} 个数据集: {subdir}")
        print(f"{'='*60}")
        
        # 构造子文件夹路径
        dataset_dir = os.path.join(datasets_path, subdir)
        # 输出文件路径：子文件夹路径/video_captions.jsonl
        output_path = os.path.join(dataset_dir, "video_captions.jsonl")
        
        try:
            # 步骤1: 调用原有的处理逻辑创建jsonl
            print(f"📝 步骤1: 创建video_captions.jsonl...")
            success = create_video_captions(dataset_dir, output_path)
            if success:
                success_count += 1
                print(f"✅ {subdir} JSONL处理成功")
                
                # 步骤2: 创建对应的toml配置文件
                print(f"⚙️ 步骤2: 创建TOML配置文件...")
                toml_success = create_toml_config(subdir, output_path, config_output_dir)
                if toml_success:
                    toml_success_count += 1
                    print(f"✅ {subdir} TOML配置创建成功")
                else:
                    print(f"⚠️ {subdir} TOML配置创建失败")
            else:
                failed_count += 1
                print(f"❌ {subdir} JSONL处理失败")
        except Exception as e:
            failed_count += 1
            print(f"❌ {subdir} 处理异常: {str(e)}")
    
    # 输出最终统计
    print(f"\n{'='*60}")
    print(f"📊 批量处理完成!")
    print(f"{'='*60}")
    print(f"总数据集: {len(subdirs)} 个")
    print(f"JSONL成功处理: {success_count} 个")
    print(f"TOML成功创建: {toml_success_count} 个")
    print(f"处理失败: {failed_count} 个")
    
    if success_count > 0:
        print(f"\n✅ 成功处理的数据集文件:")
        for subdir in subdirs:
            jsonl_path = os.path.join(datasets_path, subdir, "video_captions.jsonl")
            toml_path = os.path.join(config_output_dir, f"{subdir.replace(' ', '_')}.toml")
            if os.path.exists(jsonl_path):
                print(f"  📄 JSONL: {jsonl_path}")
            if os.path.exists(toml_path):
                print(f"  ⚙️ TOML: {toml_path}")
    
    return success_count > 0


def create_toml_config(dataset_name, jsonl_file_path, config_output_dir):
    """为数据集创建toml配置文件"""
    # 确保config输出目录存在
    os.makedirs(config_output_dir, exist_ok=True)
    
    # 处理数据集名称：空格替换为下划线，用于cache目录名称
    dataset_name_clean = dataset_name.replace(" ", "_")
    
    # toml文件名：数据集名称.toml
    toml_filename = f"{dataset_name_clean}.toml"
    toml_path = os.path.join(config_output_dir, toml_filename)
    
    # 生成toml内容
    toml_content = f"""# Common parameters (resolution, caption_extension, batch_size, num_repeats, enable_bucket, bucket_no_upscale) 
# can be set in either general or datasets sections
# Video-specific parameters (target_frames, frame_extraction, frame_stride, frame_sample, max_frames, source_fps)
# must be set in each datasets section

# caption_extension is not required for metadata jsonl file
# cache_directory is required for each dataset with metadata jsonl file

# general configurations
[general]
resolution = [960 , 960]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_jsonl_file = "{jsonl_file_path}"
frame_extraction = "full"
max_frames = 230
resolution = [298,298]
cache_directory = "/mnt/cfs/jj/musubi-tuner/Chaowei/cache/{dataset_name_clean}"


"""
    
    try:
        with open(toml_path, 'w', encoding='utf-8') as f:
            f.write(toml_content)
        print(f"✅ TOML配置文件已创建: {toml_path}")
        return True
    except Exception as e:
        print(f"❌ 创建TOML文件失败: {e}")
        return False


def print_help():
    """打印帮助信息"""
    print("📖 Create Metadata - 视频数据集元数据和配置文件生成器")
    print("=" * 60)
    print()
    print("功能：")
    print("- 为视频数据集生成video_captions.jsonl元数据文件")
    print("- 生成对应的TOML训练配置文件")
    print("- 支持批量处理多个数据集")
    print()
    print("使用方法：")
    print()
    print("1. 批量处理模式（推荐）:")
    print("   python3 create_metadata.py [数据集父目录]")
    print("   为父目录下的每个子文件夹生成：")
    print("   - video_captions.jsonl文件（放在各自的数据集目录下）")
    print("   - 对应的.toml配置文件（放在/mnt/cfs/jj/musubi-tuner/Chaowei/config/目录下）")
    print()
    print("2. 单数据集处理模式:")
    print("   python3 create_metadata.py [数据集目录] [输出文件路径]")
    print("   生成指定的video_captions.jsonl文件和对应的.toml配置文件")
    print()
    print("3. 默认模式:")
    print("   python3 create_metadata.py")
    print("   使用默认路径处理单个数据集")
    print()
    print("示例：")
    print("   # 批量处理")
    print("   python3 create_metadata.py /mnt/cfs/jj/musubi-tuner/Chaowei/datasets/")
    print()
    print("   # 单数据集处理")
    print("   python3 create_metadata.py /path/to/dataset /path/to/output.jsonl")
    print()
    print("   # 显示帮助")
    print("   python3 create_metadata.py --help")


def main():
    # 解析命令行参数
    if len(sys.argv) == 2:
        # 检查是否为帮助参数
        if sys.argv[1] in ['-h', '--help', 'help']:
            print_help()
            return
        
        # 新模式：批量处理多个数据集文件夹（使用默认配置目录）
        datasets_path = sys.argv[1]
        print("🔄 批量处理模式:")
        print(f"   数据集父目录: {datasets_path}")
        print(f"   输出文件格式: [子文件夹]/video_captions.jsonl")
        print(f"   配置目录: 默认(/mnt/cfs/jj/musubi-tuner/Chaowei/config)")
        success = batch_create_video_captions(datasets_path)
    elif len(sys.argv) == 3:
        # 检查第二个参数是否为配置目录（批量模式）还是输出文件（单数据集模式）
        datasets_path = sys.argv[1]
        second_arg = sys.argv[2]
        
        # 如果第二个参数是目录，则为批量处理模式 + 自定义配置目录
        if os.path.isdir(second_arg) or second_arg.endswith('/') or not second_arg.endswith('.jsonl'):
            config_output_dir = second_arg
            print("🔄 批量处理模式（自定义配置目录）:")
            print(f"   数据集父目录: {datasets_path}")
            print(f"   配置输出目录: {config_output_dir}")
            print(f"   输出文件格式: [子文件夹]/video_captions.jsonl")
            success = batch_create_video_captions(datasets_path, config_output_dir)
        else:
            # 旧模式：处理单个数据集文件夹
            dataset_dir = datasets_path
            output_path = second_arg
            print("📝 单数据集处理模式:")
            print(f"   数据集目录: {dataset_dir}")
            print(f"   输出文件: {output_path}")
            
            # 步骤1: 创建jsonl文件
            print(f"📝 步骤1: 创建video_captions.jsonl...")
            success = create_video_captions(dataset_dir, output_path)
            
            if success:
                # 步骤2: 创建对应的toml配置文件
                print(f"⚙️ 步骤2: 创建TOML配置文件...")
                # 从数据集目录路径提取数据集名称
                dataset_name = os.path.basename(dataset_dir.rstrip('/'))
                # 使用默认配置目录
                config_output_dir = "/mnt/cfs/jj/musubi-tuner/Chaowei/config"
                toml_success = create_toml_config(dataset_name, output_path, config_output_dir)
                if toml_success:
                    print(f"✅ TOML配置创建成功")
                else:
                    print(f"⚠️ TOML配置创建失败")
    elif len(sys.argv) == 1:
        # 默认参数（兼容旧版本）
        dataset_dir = "/mnt/cfs/jj/musubi-tuner/Earth_Zoom_Out/dataset"
        output_path = "/mnt/cfs/jj/musubi-tuner/Earth_Zoom_Out/video_captions.jsonl"
        print("📝 使用默认参数（单数据集模式）:")
        print(f"   数据集目录: {dataset_dir}")
        print(f"   输出文件: {output_path}")
        
        # 步骤1: 创建jsonl文件
        print(f"📝 步骤1: 创建video_captions.jsonl...")
        success = create_video_captions(dataset_dir, output_path)
        
        if success:
            # 步骤2: 创建对应的toml配置文件
            print(f"⚙️ 步骤2: 创建TOML配置文件...")
            # 从数据集目录路径提取数据集名称
            dataset_name = os.path.basename(dataset_dir.rstrip('/'))
            # 使用默认配置目录
            config_output_dir = "/mnt/cfs/jj/musubi-tuner/Chaowei/config"
            toml_success = create_toml_config(dataset_name, output_path, config_output_dir)
            if toml_success:
                print(f"✅ TOML配置创建成功")
            else:
                print(f"⚠️ TOML配置创建失败")
    else:
        print("❌ 用法错误!")
        print("使用方法:")
        print()
        print("1. 批量处理模式（推荐）:")
        print("   python3 create_metadata.py [数据集父目录]")
        print("   python3 create_metadata.py [数据集父目录] [配置输出目录]")
        print("   为父目录下的每个子文件夹生成：")
        print("   - video_captions.jsonl文件（放在各自的数据集目录下）")
        print("   - 对应的.toml配置文件（放在指定的配置目录下）")
        print()
        print("2. 单数据集处理模式:")
        print("   python3 create_metadata.py [数据集目录] [输出文件路径.jsonl]")
        print("   生成指定的video_captions.jsonl文件和对应的.toml配置文件")
        print()
        print("示例:")
        print("   # 批量处理（使用默认配置目录）")
        print("   python3 create_metadata.py /mnt/cfs/jj/musubi-tuner/Chaowei/datasets/")
        print()
        print("   # 批量处理（自定义配置目录）")
        print("   python3 create_metadata.py /mnt/cfs/jj/musubi-tuner/Chaowei/datasets/ /custom/config/path/")
        print()
        print("   # 单数据集处理")
        print("   python3 create_metadata.py /mnt/cfs/jj/musubi-tuner/Earth_Zoom_Out/dataset /mnt/cfs/jj/musubi-tuner/Earth_Zoom_Out/video_captions.jsonl")
        print()
        print("或直接运行（使用默认参数）:")
        print("   python3 create_metadata.py")
        sys.exit(1)
    
    # 执行结果检查
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
