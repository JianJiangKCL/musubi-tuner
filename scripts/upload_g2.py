import oss2
import os
import uuid
import json
import requests
from io import BytesIO
import math

# 创建缓存目录
os.makedirs('results', exist_ok=True)

# 初始化缓存
oss_url_cache = {}
cache_file = 'results/oss_url_cache.json'
if os.path.exists(cache_file):
    try:
        with open(cache_file, 'r') as f:
            oss_url_cache = json.load(f)
    except:
        pass

internal_endpoint = os.environ.get("OSS_ENDPOINT", "oss-cn-beijing.aliyuncs.com")
access_key_id = os.environ.get("OSS_ACCESS_KEY_ID")
access_key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
_bucket_name = os.environ.get("OSS_BUCKET_NAME", "liblibai-tmp-image")

if not access_key_id or not access_key_secret:
    raise ValueError("OSS_ACCESS_KEY_ID and OSS_ACCESS_KEY_SECRET environment variables must be set")

_auth = oss2.Auth(access_key_id=access_key_id, access_key_secret=access_key_secret)
bucket = oss2.Bucket(_auth, internal_endpoint, _bucket_name)

def upload_large_file_multipart(bucket, object_name, file_path, part_size=10*1024*1024):
    """
    使用分片上传大文件到OSS
    
    Args:
        bucket: OSS bucket对象
        object_name: OSS对象名
        file_path: 本地文件路径
        part_size: 分片大小，默认10MB
    """
    try:
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        part_count = math.ceil(file_size / part_size)
        
        print(f"文件将分为 {part_count} 个分片上传，每片 {part_size/1024/1024:.1f} MB")
        
        # 初始化分片上传
        upload_id = bucket.init_multipart_upload(object_name).upload_id
        print(f"分片上传初始化成功，Upload ID: {upload_id}")
        
        parts = []
        
        # 上传每个分片
        with open(file_path, 'rb') as f:
            for part_number in range(1, part_count + 1):
                # 读取分片数据
                offset = (part_number - 1) * part_size
                size = min(part_size, file_size - offset)
                f.seek(offset)
                data = f.read(size)
                
                # 上传分片
                print(f"上传分片 {part_number}/{part_count} ({size/1024/1024:.1f} MB)")
                result = bucket.upload_part(object_name, upload_id, part_number, data)
                parts.append(oss2.models.PartInfo(part_number, result.etag))
        
        # 完成分片上传
        print("正在合并分片...")
        bucket.complete_multipart_upload(object_name, upload_id, parts)
        print("分片上传完成！")
        
    except Exception as e:
        print(f"分片上传失败: {str(e)}")
        # 尝试取消分片上传
        try:
            bucket.abort_multipart_upload(object_name, upload_id)
            print("已取消未完成的分片上传")
        except:
            pass
        raise e

def check_url_validity(url, timeout=10):
    """检查URL是否有效可访问，增强版本能处理更多边缘情况"""
    # 对于已知的OSS域名，直接认为有效
    if 'liblibai-tmp-image.liblib.cloud' in url:
        return True
    
    try:
        # 首先尝试HEAD请求
        response = requests.head(url, timeout=timeout, verify=False, allow_redirects=True)
        if response.status_code == 200:
            return True
    except:
        pass
    
    try:
        # 如果HEAD请求失败，尝试GET请求的前几个字节
        response = requests.get(url, timeout=timeout, verify=False, stream=True, headers={'Range': 'bytes=0-1023'})
        if response.status_code in [200, 206]:  # 206是部分内容响应
            return True
    except:
        pass
    
    try:
        # 如果Range请求也失败，尝试完整GET请求的第一个chunk
        # 这对某些有访问控制的URL可能有效
        response = requests.get(url, timeout=timeout, verify=False, stream=True)
        if response.status_code == 200:
            # 尝试读取第一个chunk来验证
            try:
                chunk = next(response.iter_content(chunk_size=1024))
                if chunk:
                    return True
            except:
                pass
    except:
        pass
    
    # 对于特定的域名，我们采用更宽松的策略
    # 如果是seedance、vidu、tongyi等已知服务，即使访问检查失败也尝试下载
    known_domains = [
        'ark-content-generation-cn-beijing.tos-cn-beijing.volces.com',  # seedance
        'prod-ss-vidu.s3.cn-northwest-1.amazonaws.com.cn',  # vidu
        'dashscope-result-wlcb-acdr-1.oss-cn-wulanchabu-acdr-1.aliyuncs.com'  # tongyi
    ]
    
    for domain in known_domains:
        if domain in url:
            print(f"⚠️ 已知域名URL访问检查失败，但仍将尝试下载: {domain}")
            return True
    
    return False

def get_file_extension_from_url(url):
    """从URL获取文件扩展名"""
    # 移除查询参数
    url_path = url.split('?')[0]
    # 获取扩展名
    ext = os.path.splitext(url_path)[1].lower()
    if not ext:
        # 如果没有扩展名，根据URL内容判断
        if 'video' in url.lower() or '.mp4' in url.lower():
            return '.mp4'
        elif 'image' in url.lower() or any(img_ext in url.lower() for img_ext in ['.jpg', '.jpeg', '.png', '.gif']):
            return '.png'
        elif 'zip' in url.lower() or '.zip' in url.lower():
            return '.zip'
    return ext or '.mp4'  # 默认为mp4

def upload_url(source, file_type='auto'):
    """上传文件到OSS，支持本地文件路径或远程URL，支持图片和视频
    
    Args:
        source: 文件路径或URL
        file_type: 'image', 'video', 或 'auto' (自动检测)
    """
    try:
        # 判断输入是URL还是本地文件路径
        is_url = source.startswith('http://') or source.startswith('https://')
        
        # 如果是URL，先检查缓存或OSS地址
        if is_url:
            # 检查URL是否已经上传过
            if source in oss_url_cache:
                cached_url = oss_url_cache[source]
                print(f"使用缓存的 OSS URL: {cached_url}")
                return cached_url
                
            # 如果已经是OSS URL，直接返回
            if 'liblibai-tmp-image.liblib.cloud' in source:
                print(f"已经是OSS地址，无需上传: {source}")
                return source
                
            # 检查URL有效性
            if not check_url_validity(source):
                print(f"URL无效或已过期: {source}")
                return None
                
            # 下载远程文件到本地
            temp_dir = os.path.join(os.getcwd(), 'static', 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # 根据URL确定文件扩展名
            file_ext = get_file_extension_from_url(source)
            if file_type == 'auto':
                # 自动检测文件类型
                if file_ext in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm']:
                    file_type = 'video'
                elif file_ext in ['.zip', '.rar', '.7z', '.tar', '.gz']:
                    file_type = 'zip'
                else:
                    file_type = 'image'
            
            file_name = f"{uuid.uuid4()}{file_ext}"
            local_path = os.path.join(temp_dir, file_name)
            
            print(f"下载远程{file_type}文件: {source}")
            try:
                response = requests.get(source, timeout=60, verify=False, stream=True)
                if response.status_code != 200:
                    print(f"下载远程文件失败，状态码: {response.status_code}")
                    return None
                    
                # 流式下载，适合大文件
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"文件已保存到本地: {local_path}")
                
                # 更新source为本地文件路径，继续处理
                source = local_path
            except Exception as download_error:
                print(f"下载文件失败: {download_error}")
                return None
        
        # 处理本地文件
        if not is_url or os.path.exists(source):
            # 检查本地文件缓存
            if source in oss_url_cache and os.path.exists(source):
                cached_url = oss_url_cache[source]
                print(f"使用缓存的 OSS URL: {cached_url}")
                return cached_url
            
            # 确定文件类型
            if file_type == 'auto':
                file_ext = os.path.splitext(source)[1].lower()
                if file_ext in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm']:
                    file_type = 'video'
                elif file_ext in ['.zip', '.rar', '.7z', '.tar', '.gz']:
                    file_type = 'zip'
                else:
                    file_type = 'image'
            
            # 生成OSS对象名
            if file_type == 'video':
                object_name = f"sd-videos/{str(uuid.uuid4())}.mp4"
            elif file_type == 'zip':
                original_ext = os.path.splitext(source)[1].lower()
                object_name = f"training-data/{str(uuid.uuid4())}{original_ext}"
            else:
                object_name = f"sd-images/{str(uuid.uuid4())}.png"
                
            # 上传文件到OSS
            print(f"开始上传{file_type}文件到OSS: {source}")
            
            # 获取文件大小
            file_size = os.path.getsize(source)
            print(f"文件大小: {file_size / 1024 / 1024:.1f} MB")
            
            # 对于大文件使用分片上传，小文件使用简单上传
            if file_size > 100 * 1024 * 1024:  # 大于100MB使用分片上传
                print("使用分片上传...")
                upload_large_file_multipart(bucket, object_name, source)
            else:
                print("使用简单上传...")
                bucket.put_object_from_file(object_name, source)
            
            print(f"上传文件 {source} 到OSS成功")
            
            # 完整的 URL
            full_url = 'https://liblibai-tmp-image.liblib.cloud/' + object_name
            print(f"{file_type}文件OSS地址：{full_url}")
            
            # 记录到缓存
            original_source = source
            if is_url:
                # 获取原始URL（下载前的URL）
                for key, value in locals().items():
                    if key == 'source' and isinstance(value, str) and value.startswith('http'):
                        original_source = value
                        break
            
            oss_url_cache[original_source] = full_url
            oss_url_cache[source] = full_url  # 同时缓存本地路径
            
            # 保存缓存
            with open('results/oss_url_cache.json', 'w') as f:
                json.dump(oss_url_cache, f, indent=2)
            
            # 如果是临时下载的文件，清理它
            if is_url and os.path.exists(source) and 'temp' in source:
                try:
                    os.remove(source)
                    print(f"临时文件已删除: {source}")
                except Exception as e:
                    print(f"清理临时文件失败: {e}")
            
            return full_url
        else:
            print(f"文件不存在: {source}")
            return None
            
    except Exception as e:
        print(f"上传过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def upload_video_url(video_url):
    """专门用于上传视频URL的便捷函数"""
    return upload_url(video_url, file_type='video')

def upload_image_url(image_url):
    """专门用于上传图片URL的便捷函数"""
    return upload_url(image_url, file_type='image')

def upload_zip_file(zip_path):
    """专门用于上传zip文件的便捷函数"""
    return upload_url(zip_path, file_type='zip')

def upload_model_file(model_file_path, model_dir_uuid=None):
    """
    专门用于上传模型文件的函数
    
    Args:
        model_file_path: 模型文件的本地路径
        model_dir_uuid: 模型目录UUID，如果为None则自动生成。
                       同一个数据集的多个模型文件应使用相同的model_dir_uuid
    
    Returns:
        tuple: (oss_url, model_dir_uuid) 返回OSS地址和模型目录UUID
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(model_file_path):
            print(f"模型文件不存在: {model_file_path}")
            return None, None
        
        # 检查文件扩展名
        file_ext = os.path.splitext(model_file_path)[1].lower()
        supported_extensions = ['.safetensors', '.ckpt', '.pt', '.bin', '.pth', '.gguf', '.sft']
        
        if file_ext not in supported_extensions:
            print(f"不支持的模型文件格式: {file_ext}")
            print(f"支持的格式: {', '.join(supported_extensions)}")
            return None, None
        
        # 生成UUID（去掉连字符）
        if model_dir_uuid is None:
            model_dir_uuid = str(uuid.uuid4()).replace("-", "")
            print(f"生成新的模型目录UUID: {model_dir_uuid}")
        else:
            print(f"使用指定的模型目录UUID: {model_dir_uuid}")
        
        model_file_uuid = str(uuid.uuid4()).replace("-", "")
        print(f"模型文件UUID: {model_file_uuid}")
        
        # 生成OSS对象名
        object_name = f"models/chaowei/{model_dir_uuid}/{model_file_uuid}{file_ext}"
        print(f"OSS对象名: {object_name}")
        
        # 获取文件大小
        file_size = os.path.getsize(model_file_path)
        print(f"模型文件大小: {file_size / 1024 / 1024:.1f} MB")
        
        # 开始上传
        print(f"开始上传模型文件到OSS: {model_file_path}")
        
        # 对于大文件使用分片上传，小文件使用简单上传
        if file_size > 100 * 1024 * 1024:  # 大于100MB使用分片上传
            print("使用分片上传...")
            upload_large_file_multipart(bucket, object_name, model_file_path)
        else:
            print("使用简单上传...")
            bucket.put_object_from_file(object_name, model_file_path)
        
        print(f"模型文件上传成功: {model_file_path}")
        
        # 生成完整的URL（使用liblib.cloud域名）
        full_url = 'https://liblibai-tmp-image.liblib.cloud/' + object_name
        print(f"模型文件OSS地址: {full_url}")
        
        return full_url, model_dir_uuid
        
    except Exception as e:
        print(f"上传模型文件失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def upload_model_files_batch(model_file_paths, model_dir_uuid=None):
    """
    批量上传同一个数据集的多个模型文件
    
    Args:
        model_file_paths: 模型文件路径列表
        model_dir_uuid: 模型目录UUID，如果为None则自动生成。
                       同一批次的所有文件使用相同的model_dir_uuid
    
    Returns:
        tuple: (upload_results, model_dir_uuid)
               upload_results: [{'file_path': str, 'oss_url': str, 'success': bool}, ...]
               model_dir_uuid: 使用的模型目录UUID
    """
    upload_results = []
    
    if not model_file_paths:
        print("没有提供模型文件路径")
        return upload_results, None
    
    # 如果没有提供model_dir_uuid，生成一个新的
    if model_dir_uuid is None:
        model_dir_uuid = str(uuid.uuid4()).replace("-", "")
        print(f"为批量上传生成模型目录UUID: {model_dir_uuid}")
    
    print(f"开始批量上传 {len(model_file_paths)} 个模型文件")
    print("-" * 60)
    
    for i, file_path in enumerate(model_file_paths, 1):
        print(f"\n[{i}/{len(model_file_paths)}] 上传模型文件: {os.path.basename(file_path)}")
        
        oss_url, _ = upload_model_file(file_path, model_dir_uuid)
        
        result = {
            'file_path': file_path,
            'oss_url': oss_url,
            'success': oss_url is not None
        }
        upload_results.append(result)
        
        if result['success']:
            print(f"✅ 上传成功: {os.path.basename(file_path)}")
        else:
            print(f"❌ 上传失败: {os.path.basename(file_path)}")
    
    # 统计结果
    success_count = sum(1 for r in upload_results if r['success'])
    print(f"\n📊 批量上传完成:")
    print(f"   总文件数: {len(model_file_paths)}")
    print(f"   成功上传: {success_count}")
    print(f"   上传失败: {len(model_file_paths) - success_count}")
    print(f"   模型目录UUID: {model_dir_uuid}")
    
    return upload_results, model_dir_uuid

def migrate_history_images(history_data):
    """将历史任务中的图片上传到OSS"""
    if not history_data:
        return history_data
        
    print(f"开始迁移{len(history_data)}条历史记录的图片")
    updated = False
    
    for task in history_data:
        # 处理resultImageUrl
        if task.get('resultImageUrl') and not task.get('resultImageUrl').startswith('https://liblibai-tmp-image.liblib.cloud/'):
            oss_url = upload_url(task['resultImageUrl'])
            if oss_url != task['resultImageUrl']:
                task['resultImageUrl'] = oss_url
                updated = True
                print(f"迁移结果图片: {oss_url}")
        
        # 处理imageUrls数组
        if task.get('imageUrls') and isinstance(task['imageUrls'], list):
            for i, img_url in enumerate(task['imageUrls']):
                if img_url and not img_url.startswith('https://liblibai-tmp-image.liblib.cloud/'):
                    oss_url = upload_url(img_url)
                    if oss_url != img_url:
                        task['imageUrls'][i] = oss_url
                        updated = True
                        print(f"迁移上传图片: {oss_url}")
    
    print(f"历史记录图片迁移完成，{'有' if updated else '无'}更新")
    return history_data, updated
