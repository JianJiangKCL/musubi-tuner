#!/usr/bin/env python3
"""
Langfuse Observations 获取 - 使用 REST API
基于 https://api.reference.langfuse.com/#tag/observations/get/api/public/observations
"""
import os
import requests
import json
from datetime import datetime
import time
import base64

# 设置环境变量
LANGFUSE_SECRET_KEY = "sk-lf-4ba25923-cae5-4be2-a5b6-196910a06cad"
LANGFUSE_PUBLIC_KEY = "pk-lf-f0f5ff00-cecb-48bc-b2a6-644c18b52feb"
LANGFUSE_HOST = "https://cloud.langfuse.com"


def get_observations_via_api(page=1, limit=10, trace_id=None, name=None, user_id=None, type=None, parent_observation_id=None):
    """
    通过 REST API 获取 observations
    参考: https://api.reference.langfuse.com/#tag/observations/get/api/public/observations
    """
    url = f"{LANGFUSE_HOST}/api/public/observations"
    
    # 构建查询参数
    params = {
        "page": page,
        "limit": limit
    }
    
    # 添加可选的过滤参数
    if trace_id:
        params["traceId"] = trace_id
    if name:
        params["name"] = name
    if user_id:
        params["userId"] = user_id
    if type:
        params["type"] = type
    if parent_observation_id:
        params["parentObservationId"] = parent_observation_id
    
    # 设置认证头 - 使用正确的 Basic Auth 格式
    auth_string = f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}"
    auth_bytes = auth_string.encode('ascii')
    auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
    
    headers = {
        "Authorization": f"Basic {auth_b64}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"\n请求 URL: {url}")
        print(f"请求参数: {params}")
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API 请求失败: {response.status_code}")
            print(f"响应内容: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("请求超时")
        return None
    except Exception as e:
        print(f"请求失败: {e}")
        return None


def get_single_observation_via_api(observation_id):
    """
    通过 REST API 获取单个 observation
    """
    url = f"{LANGFUSE_HOST}/api/public/observations/{observation_id}"
    
    # 设置认证头 - 使用正确的 Basic Auth 格式
    auth_string = f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}"
    auth_bytes = auth_string.encode('ascii')
    auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
    
    headers = {
        "Authorization": f"Basic {auth_b64}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"获取单个 observation 失败: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"请求失败: {e}")
        return None


def main():
    print("=== Langfuse Observations REST API 获取工具 ===\n")
    print(f"API 文档: https://api.reference.langfuse.com/#tag/observations/get/api/public/observations")
    
    # 1. 尝试获取 observations 列表
    print("\n1. 获取 Observations 列表 (使用 REST API)...")
    
    # 尝试不同的参数组合
    test_configs = [
        {"limit": 5, "page": 1},
        {"limit": 1, "page": 1},
        {"limit": 10, "page": 1, "type": "GENERATION"},
        {"limit": 10, "page": 1, "type": "SPAN"},
    ]
    
    observations_data = None
    
    for config in test_configs:
        print(f"\n尝试配置: {config}")
        result = get_observations_via_api(**config)
        
        if result:
            observations_data = result
            print(f"✓ 成功获取数据!")
            
            # 显示返回的数据结构
            if "data" in result:
                print(f"  - 返回 {len(result['data'])} 个 observations")
                print(f"  - 总页数: {result.get('meta', {}).get('totalPages', 'N/A')}")
                print(f"  - 总数量: {result.get('meta', {}).get('totalItems', 'N/A')}")
                
                # 显示第一个 observation 的字段
                if result['data']:
                    first_obs = result['data'][0]
                    print(f"\n2. 第一个 Observation 的字段:")
                    print("=" * 60)
                    
                    for key, value in sorted(first_obs.items()):
                        if isinstance(value, (dict, list)):
                            print(f"  {key}: {type(value).__name__}")
                        else:
                            print(f"  {key}: {value}")
                    
                    print("=" * 60)
                    print(f"✓ 总共 {len(first_obs)} 个字段")
                    
                    # 保存第一个 observation 的 ID
                    first_obs_id = first_obs.get('id')
                    
                    # 3. 获取单个 observation 的详细信息
                    if first_obs_id:
                        print(f"\n3. 获取单个 Observation 详情 (ID: {first_obs_id})...")
                        single_obs = get_single_observation_via_api(first_obs_id)
                        
                        if single_obs:
                            print("✓ 成功获取单个 observation")
                            print("\n通过单独 API 获取的额外字段:")
                            
                            # 比较字段差异
                            single_keys = set(single_obs.keys())
                            list_keys = set(first_obs.keys())
                            
                            extra_keys = single_keys - list_keys
                            if extra_keys:
                                for key in sorted(extra_keys):
                                    print(f"  + {key}: {single_obs[key]}")
                            else:
                                print("  (没有额外字段)")
                break
        else:
            print("✗ 获取失败")
        
        time.sleep(1)  # 避免请求过快
    
    # 4. 如果有数据，分析所有字段
    if observations_data and observations_data.get('data'):
        print(f"\n4. 分析所有 Observations 的字段...")
        
        all_fields = set()
        field_types = {}
        
        for obs in observations_data['data']:
            for key, value in obs.items():
                all_fields.add(key)
                if key not in field_types:
                    field_types[key] = type(value).__name__
        
        # 5. 导出字段信息
        print("\n5. 导出 Observation 字段信息...")
        
        output_file = "observation_rest_api_fields.json"
        field_info = {
            "api_endpoint": f"{LANGFUSE_HOST}/api/public/observations",
            "total_fields": len(all_fields),
            "field_names": sorted(list(all_fields)),
            "field_types": field_types,
            "sample_observation": observations_data['data'][0] if observations_data['data'] else {},
            "api_parameters": {
                "page": "页码 (默认: 1)",
                "limit": "每页数量 (默认: 50)",
                "traceId": "按 trace ID 过滤",
                "name": "按名称过滤",
                "userId": "按用户 ID 过滤",
                "type": "按类型过滤 (GENERATION/SPAN/EVENT)",
                "parentObservationId": "按父 observation ID 过滤"
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(field_info, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 字段信息已保存到: {output_file}")
        print(f"  - 总字段数: {len(all_fields)}")
        print(f"  - 字段列表: {', '.join(sorted(list(all_fields))[:10])}...")
    else:
        print("\n✗ 未能获取到任何 observations 数据")
    
    print("\n" + "=" * 60)
    print("总结：")
    print("1. REST API 端点: GET /api/public/observations")
    print("2. 支持分页和多种过滤参数")
    print("3. 返回数据包含 data 数组和 meta 信息")
    print("4. 单个获取: GET /api/public/observations/{observationId}")
    print("=" * 60)


if __name__ == "__main__":
    main()
