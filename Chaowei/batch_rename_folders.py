#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量重命名文件夹脚本 - 移除[processed]前缀
作者: AI Assistant
功能: 扫描指定目录，找到所有带有[processed]前缀的文件夹并重命名移除前缀
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import re


def setup_logging(log_level: str = "INFO") -> None:
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('batch_rename.log', encoding='utf-8')
        ]
    )


def find_processed_folders(directory: str, recursive: bool = True) -> List[Path]:
    """
    查找所有带有[processed]前缀的文件夹
    
    Args:
        directory: 要搜索的目录路径
        recursive: 是否递归搜索子目录
    
    Returns:
        包含所有找到的[processed]文件夹路径的列表
    """
    processed_folders = []
    search_path = Path(directory)
    
    if not search_path.exists():
        logging.error(f"目录不存在: {directory}")
        return processed_folders
    
    # 使用正则表达式匹配[processed]前缀
    pattern = re.compile(r'^\[processed\](.+)$')
    
    try:
        if recursive:
            # 递归搜索所有子目录
            for item in search_path.rglob('*'):
                if item.is_dir() and pattern.match(item.name):
                    processed_folders.append(item)
                    logging.debug(f"找到处理过的文件夹: {item}")
        else:
            # 只搜索当前目录的直接子文件夹
            for item in search_path.iterdir():
                if item.is_dir() and pattern.match(item.name):
                    processed_folders.append(item)
                    logging.debug(f"找到处理过的文件夹: {item}")
    
    except PermissionError as e:
        logging.error(f"权限错误，无法访问目录 {directory}: {e}")
    except Exception as e:
        logging.error(f"搜索文件夹时发生错误: {e}")
    
    return processed_folders


def get_new_name(folder_path: Path) -> str:
    """
    根据原文件夹名称生成新的名称（移除[processed]前缀）
    
    Args:
        folder_path: 原文件夹路径
    
    Returns:
        移除[processed]前缀后的新名称
    """
    pattern = re.compile(r'^\[processed\](.+)$')
    match = pattern.match(folder_path.name)
    if match:
        return match.group(1)
    return folder_path.name


def check_rename_conflicts(folder_path: Path, new_name: str) -> Tuple[bool, str]:
    """
    检查重命名是否会产生冲突
    
    Args:
        folder_path: 原文件夹路径
        new_name: 新的文件夹名称
    
    Returns:
        (是否有冲突, 冲突描述)
    """
    new_path = folder_path.parent / new_name
    
    if new_path.exists():
        if new_path.is_dir():
            return True, f"目标文件夹已存在: {new_path}"
        else:
            return True, f"目标路径已被文件占用: {new_path}"
    
    return False, ""


def rename_folder(folder_path: Path, new_name: str, force: bool = False) -> bool:
    """
    重命名单个文件夹
    
    Args:
        folder_path: 原文件夹路径
        new_name: 新的文件夹名称
        force: 是否强制重命名（如果目标存在）
    
    Returns:
        重命名是否成功
    """
    try:
        new_path = folder_path.parent / new_name
        
        # 检查冲突
        has_conflict, conflict_msg = check_rename_conflicts(folder_path, new_name)
        if has_conflict and not force:
            logging.warning(f"跳过重命名 {folder_path} -> {new_name}: {conflict_msg}")
            return False
        
        # 如果强制模式且目标存在，先删除或移动目标
        if has_conflict and force:
            logging.warning(f"强制模式：将覆盖现有目标 {new_path}")
            if new_path.is_dir():
                import shutil
                shutil.rmtree(new_path)
            else:
                new_path.unlink()
        
        # 执行重命名
        folder_path.rename(new_path)
        logging.info(f"成功重命名: {folder_path} -> {new_path}")
        return True
        
    except PermissionError as e:
        logging.error(f"权限错误，无法重命名 {folder_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"重命名失败 {folder_path}: {e}")
        return False


def preview_operations(processed_folders: List[Path]) -> None:
    """
    预览将要执行的重命名操作
    
    Args:
        processed_folders: 要处理的文件夹列表
    """
    print("\n=== 预览重命名操作 ===")
    print(f"找到 {len(processed_folders)} 个需要处理的文件夹:\n")
    
    for i, folder in enumerate(processed_folders, 1):
        new_name = get_new_name(folder)
        has_conflict, conflict_msg = check_rename_conflicts(folder, new_name)
        
        status = "⚠️  冲突" if has_conflict else "✅ 可执行"
        print(f"{i:2d}. {status}")
        print(f"    原名称: {folder}")
        print(f"    新名称: {folder.parent / new_name}")
        if has_conflict:
            print(f"    冲突详情: {conflict_msg}")
        print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量重命名文件夹，移除[processed]前缀",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python batch_rename_folders.py                    # 处理当前目录
  python batch_rename_folders.py /path/to/dir       # 处理指定目录
  python batch_rename_folders.py -r                 # 递归处理所有子目录
  python batch_rename_folders.py --preview          # 只预览，不执行
  python batch_rename_folders.py --force            # 强制重命名，覆盖冲突
        """
    )
    
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='要处理的目录路径 (默认: 当前目录)'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='递归处理所有子目录'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='只预览操作，不执行重命名'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重命名，如果目标存在则覆盖'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 显示运行参数
    logging.info(f"开始批量重命名操作")
    logging.info(f"目标目录: {os.path.abspath(args.directory)}")
    logging.info(f"递归模式: {args.recursive}")
    logging.info(f"预览模式: {args.preview}")
    logging.info(f"强制模式: {args.force}")
    
    # 查找所有需要处理的文件夹
    processed_folders = find_processed_folders(args.directory, args.recursive)
    
    if not processed_folders:
        print("❌ 没有找到带有[processed]前缀的文件夹")
        logging.info("没有找到需要处理的文件夹")
        return
    
    # 预览操作
    preview_operations(processed_folders)
    
    if args.preview:
        print("🔍 预览模式：不会执行实际的重命名操作")
        return
    
    # 确认执行
    if not args.force:
        response = input(f"\n是否继续执行重命名操作？(y/N): ").strip().lower()
        if response not in ['y', 'yes', '是', '确定']:
            print("❌ 操作已取消")
            return
    
    # 执行重命名
    print("\n=== 开始执行重命名 ===")
    success_count = 0
    failed_count = 0
    
    for folder in processed_folders:
        new_name = get_new_name(folder)
        if rename_folder(folder, new_name, args.force):
            success_count += 1
            print(f"✅ 重命名成功: {folder.name} -> {new_name}")
        else:
            failed_count += 1
            print(f"❌ 重命名失败: {folder.name}")
    
    # 显示结果统计
    print(f"\n=== 操作完成 ===")
    print(f"成功: {success_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"总计: {len(processed_folders)} 个")
    
    if failed_count > 0:
        print(f"\n请查看日志文件 'batch_rename.log' 了解失败详情")
    
    logging.info(f"批量重命名操作完成 - 成功: {success_count}, 失败: {failed_count}")


if __name__ == "__main__":
    main()
