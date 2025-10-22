#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Liblib.art 模型批量上传工具

功能：
1. 读取Excel/CSV文件中的模型信息
2. 批量上传模型到 liblib.art 平台
3. 获取并记录 model_uuid 和 version_uuid
4. 自动保存进度，支持断点续传

依赖：
- requests: HTTP请求库
- pandas: Excel文件处理
- openpyxl: Excel格式支持

安装依赖：
pip install requests pandas openpyxl

用法：
python upload_to_liblib.py

Excel表格格式要求：
支持的列名（中英文自动识别）：
- request_id: 任务ID
- 模型名称/model_name: 模型名称
- 模型版本名称/version_name: 版本名称  
- model_size: 模型大小
- online_path/source_path: 模型文件路径（会自动添加域名前缀）
- status: 状态
- 触发词/trigger_word: 触发词

脚本会自动添加三列：
- model_uuid: 模型UUID（通过liblib.art的model/save接口获取）
- version_uuid: 版本UUID（通过getByUuid接口获取，validation任务需要此值）
- model_hash: 模型SHA256哈希（唯一性标识和API参数）

注意：
- 脚本只操作英文列名model_uuid和version_uuid
- 现有Excel中的中文列"模型uuid"和"模型文件uuid"与此脚本无关
- 如果version_uuid列已有值，说明该行已处理完成，会被跳过
"""

import csv
import json
import os
import time
import requests
from datetime import datetime
try:
    import pandas as pd
except ImportError:
    print("警告：pandas未安装，只能处理CSV文件。如需处理Excel文件，请运行：pip install pandas openpyxl")
    pd = None


def read_data_file(file_path):
    """
    Read data from CSV or Excel file and return rows as list
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.xlsx' or file_ext == '.xls':
        if pd is None:
            raise ImportError("需要安装pandas和openpyxl来处理Excel文件。请运行：pip install pandas openpyxl")
        
        # Read Excel file
        df = pd.read_excel(file_path)
        # Convert DataFrame to list of lists (similar to CSV reader)
        rows = [df.columns.tolist()]  # Header row
        rows.extend(df.values.tolist())  # Data rows
        return rows
    
    elif file_ext == '.csv':
        # Read CSV file
        rows = []
        with open(file_path, 'r', encoding='utf-8', newline='') as handle:
            csv_reader = csv.reader(handle)
            rows = list(csv_reader)
        return rows
    
    else:
        raise ValueError(f"不支持的文件格式：{file_ext}。支持的格式：.csv, .xlsx, .xls")


def save_json_backup(data, backup_type, model_name, version_name):
    """
    Save JSON backup data
    """
    try:
        # Create backup directory if it doesn't exist
        backup_dir = "json_backups"
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_version_name = "".join(c for c in version_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        filename = f"{backup_type}_{safe_model_name}_{safe_version_name}_{timestamp}.json"
        filepath = os.path.join(backup_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"[SAVE] JSON备份已保存: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"[WARN] JSON备份保存失败: {str(e)}")
        return None


def upload_model_to_liblib(model_name, version_name, model_source, file_name, model_size, trigger_word, file_path=None, existing_hash=None):
    """
    Upload model to liblib.art and return model_uuid and version_uuid
    
    Args:
        file_path: 本地文件路径，用于计算真实的SHA256哈希值
        existing_hash: 已有的模型哈希值，如果提供则优先使用
    """
    # 计算或使用文件的SHA256哈希值
    model_source_hash = "a77be31f12401b3feb9d2dfe4e7ed42f83d7c95413e1fc4a3aa23e3b1848c1cb"  # 默认值
    
    # 优先使用已有的哈希值
    if existing_hash and existing_hash.strip():
        model_source_hash = existing_hash.strip()
        print(f"[OK] 使用已有模型哈希: {model_source_hash}")
    elif file_path and os.path.exists(file_path):
        try:
            from file_hash_utils import calculate_file_hash
            calculated_hash = calculate_file_hash(file_path)
            if calculated_hash:
                model_source_hash = calculated_hash
                print(f"[OK] 计算得到文件SHA256哈希: {model_source_hash}")
            else:
                print(f"[WARN] 计算文件哈希失败，使用默认值: {file_path}")
        except ImportError:
            print("[WARN] file_hash_utils不可用，使用默认哈希值")
        except Exception as e:
            print(f"[WARN] 计算文件哈希异常: {e}，使用默认哈希值")
    else:
        if file_path:
            print(f"[WARN] 文件不存在，使用默认哈希值: {file_path}")
        else:
            print("[WARN] 未提供文件路径和已有哈希，使用默认哈希值")

    # Build the JSON data payload - 完全按照用户提供的正常请求格式
    json_data = {
        "sendStatus": 1,
        "name": model_name,
        "modelType": 5,
        "isOriginal": 1,
        "versions": [{
            "vae": "none",
            "showType": "0",
            "vipUsed": 0,
            "weight": 1,
            "isEncrypted": 0,
            "exclusiveDisabeld": False,
            "showTypeDisabeld": False,
            "attachment": {
                "modelSource": model_source,
                "modelSourceName": file_name,
                "modelSourceHash": model_source_hash,
                "modelSourceSize": int(model_size) if model_size.isdigit() else model_size
            },
            "name": version_name,
            "baseType": 35,
            "cfg": 1,
            "ckpt": ["21251055"],
            "versionDesc": "<p>wavespeed版</p>",
            "noTriggerWord": 0 if trigger_word else 1,
            "triggerWord": trigger_word,
            "noHdSamplerMethods": 1,
            "loraDes": "离开地球",
            "imageGroup": {
                "imagesV2": None,
                "images": [{
                    "file": "img/081e9f07d9bd4c2ba090efde163518f9/327fce8978522adf1b22e61647edd763c9f0c2230e27b56e760aa97f971b5195.jpg",
                    "url": "https://liblibai-online.liblib.cloud/img/081e9f07d9bd4c2ba090efde163518f9/d27ad0a6fdbdbc5717dcb364b15607fc1f3e62dcb7a4b6f25ff0812996d91880.png",
                    "generateInfo": {
                        "prompt": "",
                        "negativePrompt": "",
                        "steps": "",
                        "size": "",
                        "seed": "",
                        "model": [],
                        "modelHash": "",
                        "sampler": "",
                        "cfgScale": "",
                        "clipSkip": "",
                        "metainformation": '{"AIGC":{},"comfy-prompt":{},"comfy-workflow":{},"Hash collection":[]}',
                        "samplingMethod": "",
                        "samplingStep": ""
                    },
                    "originalInfo": {
                        "AIGC": {},
                        "comfy-prompt": {},
                        "comfy-workflow": {},
                        "Hash collection": []
                    },
                    "height": 1536,
                    "width": 1536,
                    "isValid": True,
                    "imageUrl": "https://liblibai-online.liblib.cloud/img/081e9f07d9bd4c2ba090efde163518f9/d27ad0a6fdbdbc5717dcb364b15607fc1f3e62dcb7a4b6f25ff0812996d91880.png",
                    "originalName": "e101f61726ae1b146d7b103b2bb4c90151beed33999ab8c156b29ec11c88a348 (1).jpg",
                    "models": [],
                    "uid": "__AUTO__1756479777652_0__"
                }]
            },
            "exclusive": 0,
            "privilege": [0],
            "teamId": 1335,
            "versionIntro": f'{{"noTriggerWord":{1 if not trigger_word else 0},"loraDes":"离开地球","weight":1,"vae":"none","cfg":1,"noHdSamplerMethods":1,"ckpt":["21251055"],"samplingStep":30}}'
        }],
        "tagsV2": {}
    }
    
    # Prepare headers
    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'zh-CN,zh;q=0.9',
        'cache-control': 'no-cache',
        'content-type': 'application/json',
        'origin': 'https://www.liblib.art',
        'pragma': 'no-cache',
        'priority': 'u=1, i',
        'referer': 'https://www.liblib.art/uploadmodel',
        'sec-ch-ua': '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'token': 'd6f491db1e294056b1ae316e629ddb0a172081e9',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
        'webid': '1745248738942dctsuplo'
    }
    
    # Get current timestamp
    timestamp = int(time.time() * 1000)
    url = f"https://api2.liblib.art/api/www/model/save?timestamp={timestamp}"
    
    try:
        print(f"正在上传模型：{model_name} - {version_name}")
        
        # Send POST request
        response = requests.post(url, headers=headers, json=json_data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print(f"API响应：{result}")
        
        # 保存model/save API响应的JSON备份
        save_json_backup(result, "model_save", model_name, version_name)
        
        if result.get('code') == 0 and result.get('data'):
            # 正确提取model_uuid
            data = result['data']
            if isinstance(data, dict) and 'uuid' in data:
                model_uuid = data['uuid']
            elif isinstance(data, str):
                model_uuid = data
            else:
                model_uuid = str(data)
            
            print(f"成功获取model_uuid: {model_uuid}")
            
            # Get version_uuid
            version_uuid = get_version_uuid(model_uuid, model_name, version_name)
            
            return {
                'success': True,
                'model_uuid': model_uuid,
                'version_uuid': version_uuid,
                'model_hash': model_source_hash,
                'message': '上传成功'
            }
        elif result.get('code') == 500:
            return {
                'success': False,
                'model_uuid': None,
                'version_uuid': None,
                'model_hash': None,
                'message': f"服务异常：{result.get('msg', 'liblib.art API服务暂时不可用，请稍后重试')}"
            }
        else:
            return {
                'success': False,
                'model_uuid': None,
                'version_uuid': None,
                'model_hash': None,
                'message': f"上传失败：{result.get('msg', '未知错误')} (code: {result.get('code', 'N/A')})"
            }
            
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误：{str(e)}")
        return {
            'success': False,
            'model_uuid': None,
            'version_uuid': None,
            'model_hash': None,
            'message': f"网络错误：{str(e)}"
        }
    except Exception as e:
        print(f"上传过程中发生错误：{str(e)}")
        return {
            'success': False,
            'model_uuid': None,
            'version_uuid': None,
            'model_hash': None,
            'message': f"未知错误：{str(e)}"
        }


def get_version_uuid(model_uuid, model_name="", version_name="", max_retries=3):
    """
    Get version_uuid from model_uuid with retry mechanism
    """
    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'zh-CN,zh;q=0.9',
        'cache-control': 'no-cache',
        'content-type': 'application/json',
        'origin': 'https://www.liblib.art',
        'pragma': 'no-cache',
        'priority': 'u=1, i',
        'referer': f'https://www.liblib.art/modelinfo/{model_uuid}?from=personal_page',
        'sec-ch-ua': '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'token': 'd6f491db1e294056b1ae316e629ddb0a172081e9',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
        'webid': '1745248738942dctsuplo'
    }
    
    for attempt in range(max_retries):
        try:
            timestamp = int(time.time() * 1000)
            url = f"https://api2.liblib.art/api/www/model/getByUuid/{model_uuid}?timestamp={timestamp}"
            
            print(f"正在获取version_uuid for model: {model_uuid} (尝试 {attempt + 1}/{max_retries})")
            
            # 使用POST方法，发送空的JSON数据
            response = requests.post(url, headers=headers, json={}, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            print(f"getByUuid API响应：{result}")
            
            # 保存getByUuid API响应的JSON备份
            if model_name and version_name:
                save_json_backup(result, "get_by_uuid", model_name, version_name)
            
            if result.get('code') == 0 and result.get('data'):
                data = result['data']
                print(f"调试信息 - data结构：{data}")
                
                if data and 'versions' in data:
                    versions = data['versions']
                    print(f"调试信息 - versions数量：{len(versions) if versions else 0}")
                    
                    if versions and len(versions) > 0:
                        first_version = versions[0]
                        print(f"调试信息 - 第一个version结构：{first_version}")
                        
                        version_uuid = first_version.get('uuid')
                        if version_uuid:
                            print(f"成功获取version_uuid: {version_uuid}")
                            return version_uuid
                        else:
                            print("警告：第一个version中没有找到uuid字段")
                            print(f"可用字段：{list(first_version.keys()) if isinstance(first_version, dict) else 'Not a dict'}")
                            return None
                    else:
                        print("警告：versions数组为空")
                        return None
                else:
                    print("警告：返回数据中没有versions信息")
                    print(f"data中的可用字段：{list(data.keys()) if isinstance(data, dict) else 'data不是字典'}")
                    return None
            elif result.get('code') == 500:
                # 服务异常，等待后重试
                print(f"服务异常(code 500)，等待5秒后重试...")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    print("达到最大重试次数，获取version_uuid失败")
                    return None
            else:
                print(f"获取version_uuid失败：{result.get('msg', '未知错误')}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"获取version_uuid网络错误：{str(e)}")
            if attempt < max_retries - 1:
                print(f"等待3秒后重试...")
                time.sleep(3)
                continue
            else:
                print("达到最大重试次数，网络请求失败")
                return None
        except Exception as e:
            print(f"获取version_uuid过程中发生错误：{str(e)}")
            return None
    
    return None


def save_excel_with_uuids(file_path, data_rows):
    """
    Save updated data back to Excel file with backup
    """
    if pd is None:
        print("错误：需要pandas库来保存Excel文件")
        return False
    
    try:
        # Create backup before saving
        if os.path.exists(file_path):
            backup_path = file_path.replace('.xlsx', f'_backup_{int(time.time())}.xlsx')
            try:
                import shutil
                shutil.copy2(file_path, backup_path)
                print(f"📁 创建备份文件：{backup_path}")
            except Exception as e:
                print(f"[WARN] 备份文件创建失败：{str(e)}")
        
        # 调试信息：检查rows数据中的UUID
        print(f"[DEBUG] 调试：检查rows数据中的UUID更新情况...")
        header = data_rows[0]
        model_uuid_col = header.index('model_uuid') if 'model_uuid' in header else -1
        version_uuid_col = header.index('version_uuid') if 'version_uuid' in header else -1
        print(f"[DEBUG] 调试：model_uuid列索引: {model_uuid_col}, version_uuid列索引: {version_uuid_col}")
        
        # 检查rows中有UUID的行
        rows_with_uuid = 0
        for i, row in enumerate(data_rows[1:], 1):
            if (model_uuid_col >= 0 and version_uuid_col >= 0 and 
                len(row) > max(model_uuid_col, version_uuid_col) and
                row[model_uuid_col] and row[version_uuid_col] and
                str(row[model_uuid_col]).strip() and str(row[version_uuid_col]).strip() and
                str(row[model_uuid_col]) != 'nan' and str(row[version_uuid_col]) != 'nan'):
                rows_with_uuid += 1
                print(f"[DEBUG] 调试：行{i}包含UUID: model_uuid={row[model_uuid_col]}, version_uuid={row[version_uuid_col]}")
        print(f"[DEBUG] 调试：rows数据中包含完整UUID的行数: {rows_with_uuid}")
        
        # Convert rows back to DataFrame with explicit data type handling
        print(f"[DEBUG] 调试：转换rows为DataFrame...")
        try:
            # 确保所有行的长度一致
            max_cols = len(data_rows[0])
            for i, row in enumerate(data_rows[1:]):
                while len(row) < max_cols:
                    row.append('')
                # 确保UUID列的数据类型为字符串
                if model_uuid_col >= 0 and len(row) > model_uuid_col:
                    row[model_uuid_col] = str(row[model_uuid_col]) if row[model_uuid_col] is not None else ''
                if version_uuid_col >= 0 and len(row) > version_uuid_col:
                    row[version_uuid_col] = str(row[version_uuid_col]) if row[version_uuid_col] is not None else ''
            
            df = pd.DataFrame(data_rows[1:], columns=data_rows[0])
            print(f"[DEBUG] 调试：DataFrame创建成功，形状: {df.shape}")
            
            # 确保UUID列的数据类型
            if 'model_uuid' in df.columns:
                df['model_uuid'] = df['model_uuid'].astype(str)
                df['model_uuid'] = df['model_uuid'].replace(['nan', 'None'], '')
            if 'version_uuid' in df.columns:
                df['version_uuid'] = df['version_uuid'].astype(str)
                df['version_uuid'] = df['version_uuid'].replace(['nan', 'None'], '')
            
        except Exception as e:
            print(f"[ERROR] DataFrame转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 调试信息：检查DataFrame中的UUID数据
        print(f"[DEBUG] 调试：检查DataFrame中的UUID数据...")
        if 'model_uuid' in df.columns and 'version_uuid' in df.columns:
            # 检查非空且非'nan'的UUID
            mask = ((df['model_uuid'].notna()) & (df['model_uuid'] != '') & (df['model_uuid'] != 'nan') &
                    (df['version_uuid'].notna()) & (df['version_uuid'] != '') & (df['version_uuid'] != 'nan'))
            uuid_count = mask.sum()
            print(f"[DEBUG] 调试：DataFrame中包含有效UUID的行数: {uuid_count}")
            if uuid_count > 0:
                sample_rows = df[mask].head(3)
                print(f"[DEBUG] 调试：样本UUID行:")
                for idx, row in sample_rows.iterrows():
                    print(f"   行{idx}: model_uuid={row['model_uuid']}, version_uuid={row['version_uuid']}")
        
        # Save to Excel
        df.to_excel(file_path, index=False)
        print(f"[SAVE] 成功保存更新后的数据到：{file_path}")
        
        # 验证保存结果
        print(f"[DEBUG] 调试：验证保存结果...")
        df_verify = pd.read_excel(file_path)
        if 'model_uuid' in df_verify.columns and 'version_uuid' in df_verify.columns:
            verify_mask = ((df_verify['model_uuid'].notna()) & (df_verify['model_uuid'] != '') &
                          (df_verify['version_uuid'].notna()) & (df_verify['version_uuid'] != ''))
            uuid_count_saved = verify_mask.sum()
            print(f"[DEBUG] 调试：保存后文件中包含有效UUID的行数: {uuid_count_saved}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 保存Excel文件时发生错误：{str(e)}")
        return False


def upload_to_liblib(data_file):
    """
    Main function to process CSV or Excel file and upload models to liblib.art
    """
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"错误：找不到文件 {data_file}")
        return
    
    try:
        # Read data from file (CSV or Excel)
        rows = read_data_file(data_file)
        
        print(f"成功读取文件：{data_file}")
        print(f"共找到 {len(rows)} 行数据")
        
        # Check and add UUID columns if they don't exist
        header = rows[0]
        model_uuid_col = -1
        version_uuid_col = -1
        model_hash_col = -1
        
        # Find or add model_uuid column (只使用英文列名，用于写入API返回的UUID)
        if 'model_uuid' in header:
            model_uuid_col = header.index('model_uuid')
            print(f"[OK] 找到model_uuid列 (索引: {model_uuid_col})")
        else:
            header.append('model_uuid')
            model_uuid_col = len(header) - 1
            # Add empty values for existing rows
            for row in rows[1:]:
                row.append('')
            print(f"➕ 新增model_uuid列 (索引: {model_uuid_col})")
        
        # Find or add version_uuid column (只使用英文列名，用于写入API返回的UUID)
        if 'version_uuid' in header:
            version_uuid_col = header.index('version_uuid')
            print(f"[OK] 找到version_uuid列 (索引: {version_uuid_col})")
        else:
            header.append('version_uuid')
            version_uuid_col = len(header) - 1
            # Add empty values for existing rows
            for row in rows[1:]:
                row.append('')
            print(f"➕ 新增version_uuid列 (索引: {version_uuid_col})")
        
        # Find or add model_hash column (模型SHA256哈希值，用于唯一性标识和API参数)
        if 'model_hash' in header:
            model_hash_col = header.index('model_hash')
            print(f"[OK] 找到model_hash列 (索引: {model_hash_col})")
        else:
            header.append('model_hash')
            model_hash_col = len(header) - 1
            # Add empty values for existing rows
            for row in rows[1:]:
                row.append('')
            print(f"➕ 新增model_hash列 (索引: {model_hash_col})")
        
        print(f"列索引映射：model_uuid={model_uuid_col}, version_uuid={version_uuid_col}, model_hash={model_hash_col}")
        
        # Process each row
        successful_uploads = 0
        failed_uploads = 0
        
        for i in range(len(rows)):
            # Skip header row
            if i == 0:
                print(f"标题行：{rows[i]}")
                continue
            
            # 直接操作原始rows[i]，避免副本问题
            row = rows[i]
            
            # Ensure row has enough columns
            while len(row) < len(header):
                row.append('')
            
            # Handle potential NaN values from Excel
            row[:] = [str(cell) if cell is not None and str(cell) != 'nan' else '' for cell in row]
            
            # 检查version_uuid列是否已有值（如果有说明已经调用过API）
            existing_model_uuid = row[model_uuid_col].strip() if row[model_uuid_col] else ""
            existing_version_uuid = row[version_uuid_col].strip() if row[version_uuid_col] else ""
            
            # 如果version_uuid已经有值，说明已经上传过API，可以跳过
            if existing_version_uuid and existing_version_uuid != 'nan':
                # 显示跳过原因，包含request_id信息（如果有的话）
                request_id_info = ""
                try:
                    request_id_idx = header.index('request_id') if 'request_id' in header else -1
                    if request_id_idx >= 0 and request_id_idx < len(row):
                        request_id_value = row[request_id_idx]
                        if request_id_value and str(request_id_value) != 'nan':
                            request_id_info = f" (TaskID: {request_id_value})"
                except:
                    pass
                print(f"[OK] 第 {i+1} 行已有version_uuid，跳过：{existing_version_uuid[:8]}...{request_id_info}")
                continue
            else:
                print(f"[INFO] 第 {i+1} 行version_uuid为空，需要调用API获取 (model_uuid: {existing_model_uuid or '无'})")
            
            # Check if row has enough base columns (移动到数据提取后检查)
            
            # Extract data from columns using column names instead of hardcoded indices
            try:
                # 获取列索引
                model_name_col = header.index('模型名称') if '模型名称' in header else (header.index('model_name') if 'model_name' in header else 0)
                version_name_col = header.index('模型版本名称') if '模型版本名称' in header else (header.index('version_name') if 'version_name' in header else 1)
                model_size_col = header.index('model_size') if 'model_size' in header else 2
                source_path_col = header.index('online_path') if 'online_path' in header else (header.index('source_path') if 'source_path' in header else 3)
                trigger_word_col = header.index('触发词') if '触发词' in header else (header.index('trigger_word') if 'trigger_word' in header else 5)
                
                # 提取数据
                model_name = str(row[model_name_col]).strip() if model_name_col < len(row) else ""
                version_name = str(row[version_name_col]).strip() if version_name_col < len(row) else ""
                model_size = str(row[model_size_col]).strip() if model_size_col < len(row) else ""
                source_path = str(row[source_path_col]).strip() if source_path_col < len(row) else ""
                trigger_word = str(row[trigger_word_col]).strip() if trigger_word_col < len(row) else ""
                
                file_name = os.path.basename(source_path)  # Get filename from model source path
                model_source = "https://liblibai-online.liblib.cloud/" + source_path
                
            except (ValueError, IndexError) as e:
                print(f"[ERROR] 第 {i+1} 行列解析失败: {e}")
                print(f"   可用列: {header}")
                failed_uploads += 1
                continue
            
            # 尝试获取本地文件路径和已有哈希
            local_file_path = None
            existing_model_hash = None
            
            try:
                # 查找file_path列
                if 'file_path' in header:
                    file_path_col = header.index('file_path')
                    if file_path_col < len(row):
                        local_file_path = str(row[file_path_col]).strip()
                        if local_file_path == 'nan' or not local_file_path:
                            local_file_path = None
                
                # 查找已有的模型哈希
                if model_hash_col >= 0 and model_hash_col < len(row):
                    existing_model_hash = str(row[model_hash_col]).strip()
                    if existing_model_hash == 'nan' or not existing_model_hash:
                        existing_model_hash = None
            except:
                pass
            
            # Skip empty rows
            if not model_name or not version_name or not source_path:
                print(f"[WARN] 警告：第 {i+1} 行关键数据为空，跳过")
                print(f"   model_name: '{model_name}', version_name: '{version_name}', source_path: '{source_path}'")
                failed_uploads += 1
                continue
            
            print(f"\n{'='*50}")
            print(f"处理第 {i+1} 行：{model_name} - {version_name}")
            if local_file_path:
                print(f"本地文件路径: {local_file_path}")
            if existing_model_hash:
                print(f"已有模型哈希: {existing_model_hash[:16]}...")
            print(f"{'='*50}")
            
            # Upload model
            result = upload_model_to_liblib(model_name, version_name, model_source, file_name, model_size, trigger_word, local_file_path, existing_model_hash)
            
            # 详细打印返回值进行调试
            print(f"[DEBUG] upload_model_to_liblib 返回值调试:")
            print(f"   result类型: {type(result)}")
            print(f"   result内容: {result}")
            print(f"   success: {result.get('success', 'KEY_NOT_FOUND')}")
            if 'message' in result:
                print(f"   message: {result['message']}")
            
            if result['success']:
                model_uuid = result['model_uuid']
                version_uuid = result['version_uuid']
                model_hash = result.get('model_hash')
                
                print(f"[OK] 成功上传：{model_name}")
                print(f"   model_uuid: {model_uuid}")
                print(f"   version_uuid: {version_uuid}")
                print(f"   model_hash: {model_hash}")
                
                # 只有在两个UUID都成功获取后才更新Excel
                if model_uuid and version_uuid:
                    # Update row with UUIDs and hash
                    print(f"[DEBUG] 更新前行数据调试：model_uuid_col={model_uuid_col}, version_uuid_col={version_uuid_col}")
                    print(f"[DEBUG] 更新前行长度: {len(row)}, 需要的最小长度: {max(model_uuid_col, version_uuid_col) + 1}")
                    
                    # 确保行有足够的列
                    while len(rows[i]) <= max(model_uuid_col, version_uuid_col):
                        rows[i].append('')
                    
                    # 直接修改原始rows[i]，确保更改被保存
                    rows[i][model_uuid_col] = model_uuid
                    rows[i][version_uuid_col] = version_uuid
                    if model_hash:
                        rows[i][model_hash_col] = model_hash
                    
                    print(f"[DEBUG] 更新后验证：rows[{i}][{model_uuid_col}]={rows[i][model_uuid_col]}, rows[{i}][{version_uuid_col}]={rows[i][version_uuid_col]}")
                    print(f"[SAVE] 保存到Excel：model_uuid={model_uuid}, version_uuid={version_uuid}, model_hash={model_hash[:16] if model_hash else 'None'}...")
                    
                    # Save progress after each successful upload
                    save_excel_with_uuids(data_file, rows)
                    successful_uploads += 1
                elif model_uuid and not version_uuid:
                    print(f"[WARN] 警告：model_uuid获取成功但version_uuid获取失败，尝试重新获取...")
                    print(f"   model_uuid: {model_uuid}")
                    
                    # 等待3秒后重试获取version_uuid
                    time.sleep(3)
                    retry_version_uuid = get_version_uuid(model_uuid, model_name, version_name, max_retries=2)
                    
                    if retry_version_uuid:
                        print(f"[OK] 重试成功获取version_uuid: {retry_version_uuid}")
                        rows[i][model_uuid_col] = model_uuid
                        rows[i][version_uuid_col] = retry_version_uuid
                        
                        print(f"[SAVE] 保存UUID到Excel：model_uuid={model_uuid}, version_uuid={retry_version_uuid}")
                        save_excel_with_uuids(data_file, rows)
                        successful_uploads += 1
                    else:
                        print(f"[ERROR] 重试后仍无法获取version_uuid，暂不保存到Excel")
                        print(f"[SAVE] 保存临时进度记录...")
                        
                        # 保存不完整的结果到临时文件以便后续处理
                        temp_record = {
                            "timestamp": datetime.now().isoformat(),
                            "model_name": model_name,
                            "version_name": version_name,
                            "model_uuid": model_uuid,
                            "version_uuid": None,
                            "status": "incomplete_uuid"
                        }
                        save_json_backup(temp_record, "incomplete_uuid", model_name, version_name)
                        failed_uploads += 1
                else:
                    print(f"[ERROR] UUID获取不完整，跳过保存")
                    failed_uploads += 1
                    
            else:
                print(f"[ERROR] 上传失败：{model_name}")
                print(f"   错误信息：{result.get('message', '未知错误')}")
                print(f"   完整失败信息: {result}")
                failed_uploads += 1
            
            # Add delay between requests to avoid rate limiting
            time.sleep(2)
        
        # Final save
        save_excel_with_uuids(data_file, rows)
        
        print(f"\n{'='*50}")
        print(f"上传完成！")
        print(f"成功：{successful_uploads} 个")
        print(f"失败：{failed_uploads} 个")
        print(f"{'='*50}")
                
    except ImportError as e:
        print(f"导入错误：{str(e)}")
    except Exception as e:
        print(f"错误：处理文件时发生异常 - {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    # Check if filename provided as command line argument
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        # Default file names to try
        default_files = ['model_upload_data.csv', 'model_summary.xlsx', 'upload.xlsx', 'upload.csv', 'model_summary.csv']
        file_name = None

        for default_file in default_files:
            if os.path.exists(default_file):
                file_name = default_file
                print(f"找到默认文件：{file_name}")
                break

        if file_name is None:
            print("未找到默认文件，请指定文件名：")
            print("用法：python upload_to_liblib.py <文件名.xlsx|.csv>")
            print("支持的文件格式：.xlsx, .xls, .csv")
            sys.exit(1)

    upload_to_liblib(file_name)