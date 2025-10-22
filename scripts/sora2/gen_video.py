
#!/usr/bin/env python3
"""
OpenAI Sora-2 视频生成脚本
使用与“连接测试脚本”类似的结构：
- 环境变量配置
- 创建 OpenAI 客户端
- 发起 Sora-2 视频生成请求
- 可选轮询任务状态
- 可选下载生成的视频

环境变量：
  - OPENAI_API_KEY (必需)
  - OPENAI_BASE_URL (可选，默认 https://api.openai.com/v1)

用法示例：
  python3 scripts/sora2/gen_video.py \
    --prompt "A cool cat rides a motorcycle at night, cinematic lighting" \
    --output /tmp/sora2_cat.mp4
"""

import os
import sys
import json
import time
import argparse
import traceback
from typing import Optional, Dict, Any

try:
    from openai import OpenAI
except ImportError as e:
    print(f"缺少必要的依赖包: {e}")
    print("请安装: pip install openai")
    sys.exit(1)

try:
    import requests
except ImportError as e:
    print(f"缺少必要的依赖包: {e}")
    print("请安装: pip install requests")
    sys.exit(1)


def load_config_from_env() -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    if not api_key:
        print("❌ 未设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)
    return {
        "api_key": api_key,
        "base_url": base_url,
        "client_type": "openai",
    }


def create_openai_client(config: Dict[str, Any]) -> Optional[OpenAI]:
    try:
        print("正在创建 OpenAI 客户端...")
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=120.0,
            max_retries=3,
        )
        print("✅ OpenAI 客户端创建成功")
        return client
    except Exception as e:
        print(f"❌ 创建 OpenAI 客户端失败: {str(e)}")
        print(f"错误详情:\n{traceback.format_exc()}")
        return None


def sora_create_video(client: OpenAI, model: str, prompt: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """调用 Sora-2 视频生成接口。返回原始响应字典。"""
    extra = extra or {}
    print(f"\n正在创建视频任务: model={model}")
    try:
        start_time = time.time()
        # 与已有示例保持兼容：最小参数 model + prompt
        resp = client.videos.create(model=model, prompt=prompt, **extra)  # type: ignore[attr-defined]
        elapsed = time.time() - start_time
        # 将响应对象尽量序列化为 dict
        data = resp if isinstance(resp, dict) else json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
        print(f"✅ 创建成功，用时 {elapsed:.2f}s")
        print_safe_video_info(data)
        return data
    except Exception as e:
        print(f"❌ 创建视频任务失败: {str(e)}")
        print(f"错误详情:\n{traceback.format_exc()}")
        raise


def print_safe_video_info(data: Dict[str, Any]) -> None:
    video_id = data.get("id") or data.get("video_id") or data.get("task_id")
    status = data.get("status") or data.get("state")
    print(f"任务ID: {video_id}")
    print(f"任务状态: {status}")
    # 尝试打印资源 URL（如果立即可用）
    url = extract_first_video_url(data)
    if url:
        print(f"视频URL: {url}")


def extract_first_video_url(data: Dict[str, Any]) -> Optional[str]:
    # 常见字段尝试：assets、video、output、result 等
    candidates = [
        data.get("video"),
        data.get("output"),
        data.get("result"),
        data.get("assets"),
        data.get("data"),
    ]
    for c in candidates:
        if not c:
            continue
        if isinstance(c, str) and c.startswith("http"):
            return c
        if isinstance(c, dict):
            # 常见 key 尝试
            for k in ("url", "video_url", "mp4", "file", "download_url"):
                v = c.get(k)
                if isinstance(v, str) and v.startswith("http"):
                    return v
        if isinstance(c, list):
            for item in c:
                if isinstance(item, str) and item.startswith("http"):
                    return item
                if isinstance(item, dict):
                    for k in ("url", "video_url", "mp4", "file", "download_url"):
                        v = item.get(k)
                        if isinstance(v, str) and v.startswith("http"):
                            return v
    return None


def extract_file_ids(data: Any) -> list[str]:
    """在任意嵌套结构中提取疑似文件ID（以 file_ 开头）。"""
    found: list[str] = []
    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and v.startswith("file_"):
                    found.append(v)
                else:
                    _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, str):
            if node.startswith("file_"):
                found.append(node)
    _walk(data)
    # 去重保持顺序
    seen = set()
    unique: list[str] = []
    for fid in found:
        if fid not in seen:
            seen.add(fid)
            unique.append(fid)
    return unique


def download_via_file_id(client: OpenAI, file_id: str, output_path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"尝试通过 file_id 下载: {file_id}\n保存到: {output_path}")
        # 优先使用 SDK 的流式保存（如果可用）
        content = client.files.content(file_id)
        if hasattr(content, "stream_to_file"):
            content.stream_to_file(output_path)  # type: ignore[attr-defined]
            print("✅ 文件下载完成 (stream_to_file)")
            return True
        # 兼容：作为上下文管理器读取
        try:
            with client.files.content(file_id) as r:  # type: ignore
                if hasattr(r, "stream_to_file"):
                    r.stream_to_file(output_path)  # type: ignore[attr-defined]
                    print("✅ 文件下载完成 (context stream_to_file)")
                    return True
                data = r.read() if hasattr(r, "read") else None
                if data is not None:
                    with open(output_path, "wb") as f:
                        f.write(data)
                    print("✅ 文件下载完成 (read)")
                    return True
        except Exception:
            pass
        # 最后回退：尝试将 content 视作 bytes
        if isinstance(content, (bytes, bytearray)):
            with open(output_path, "wb") as f:
                f.write(content)
            print("✅ 文件下载完成 (bytes)")
            return True
        print("⚠️ 文件下载方式不受支持，未保存")
        return False
    except Exception as e:
        print(f"❌ 通过 file_id 下载失败: {e}")
        return False


def try_poll_until_done(client: OpenAI, initial: Dict[str, Any], poll_interval: float = 5.0, timeout: float = 900.0) -> Dict[str, Any]:
    """尝试轮询任务直至完成（如果 API 支持 retrieve 方式）。如果不支持，原样返回 initial。"""
    video_id = initial.get("id") or initial.get("video_id") or initial.get("task_id")
    if not video_id:
        print("⚠️ 未找到任务ID，跳过轮询")
        return initial

    print(f"开始轮询任务进度（每 {poll_interval:.0f}s，一共 {timeout:.0f}s 超时）...")
    start = time.time()
    last_status = str(initial.get("status") or initial.get("state") or "unknown")
    try:
        while True:
            if time.time() - start > timeout:
                print("⏰ 轮询超时，返回最新状态")
                return initial
            time.sleep(poll_interval)
            # 不同 SDK 可能是 retrieve/get，做容错尝试
            resp = None
            try:
                if hasattr(client.videos, "retrieve"):
                    resp = client.videos.retrieve(video_id)  # type: ignore[attr-defined]
                elif hasattr(client.videos, "get"):
                    resp = client.videos.get(video_id)  # type: ignore[attr-defined]
            except Exception:
                # 如果 retrieve 不可用，则无法继续轮询
                print("⚠️ SDK 不支持 retrieve/get，停止轮询")
                return initial

            data = resp if isinstance(resp, dict) else json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
            status = str(data.get("status") or data.get("state") or "unknown")
            if status != last_status:
                last_status = status
                print(f"状态更新: {status}")
            if status.lower() in {"succeeded", "completed", "finished", "done", "success"}:
                print("✅ 任务已完成")
                return data
            if status.lower() in {"failed", "error", "cancelled"}:
                print("❌ 任务失败/取消")
                return data
    except KeyboardInterrupt:
        print("用户中断轮询")
        return initial


def download_file(url: str, output_path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"开始下载: {url}\n保存到: {output_path}")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print("✅ 下载完成")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def rest_download_by_video_id(api_key: str, base_url: str, video_id: str, output_path: str) -> bool:
    """使用 REST 端点尝试下载视频内容或解析资产列表。"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        headers = {"Authorization": f"Bearer {api_key}"}
        # 常见候选端点（不同版本/供应方实现可能不同）
        candidates = [
            f"{base_url.rstrip('/')}/videos/{video_id}/content",
            f"{base_url.rstrip('/')}/videos/{video_id}/download",
            f"{base_url.rstrip('/')}/videos/{video_id}/assets",
        ]
        for u in candidates:
            try:
                print(f"尝试 REST 下载: {u}")
                with requests.get(u, headers=headers, stream=True, timeout=120) as r:
                    if r.status_code != 200:
                        print(f"REST 下载失败: {r.status_code}")
                        continue
                    content_type = r.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        # 可能是资产列表，尝试解析其中的 URL
                        data = r.json()
                        url = extract_first_video_url(data)
                        if url:
                            return download_file(url, output_path)
                        if isinstance(data, dict):
                            for key in ("url", "download_url", "video_url"):
                                val = data.get(key)
                                if isinstance(val, str) and val.startswith("http"):
                                    return download_file(val, output_path)
                        print("REST 返回 JSON 但未发现可下载 URL")
                        continue
                    # 否则当作二进制内容保存
                    with open(output_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    print("✅ 通过 REST 下载完成")
                    return True
            except Exception as e:
                print(f"REST 尝试失败: {e}")
        return False
    except Exception as e:
        print(f"❌ REST 下载出错: {e}")
        return False


def _slugify_from_prompt(prompt: str) -> str:
    base = prompt.strip().lower()
    # Replace spaces with underscores and keep safe chars
    safe = []
    for ch in base:
        if ch.isalnum():
            safe.append(ch)
        elif ch in {" ", "-", "_"}:
            safe.append("_")
    slug = "".join(safe).strip("_")
    if not slug:
        slug = "video"
    if len(slug) > 60:
        slug = slug[:60]
    return slug


def main():
    parser = argparse.ArgumentParser(description="OpenAI Sora-2 视频生成")
    parser.add_argument("prompt", type=str, help="视频文案提示词（位置参数）")
    parser.add_argument("--model", type=str, default="sora-2", help="使用的模型，默认 sora-2")
    parser.add_argument("--output", type=str, default=None, help="下载保存路径（若接口直接返回视频URL）")
    parser.add_argument("--no-poll", action="store_true", help="不轮询任务状态，创建后直接退出")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="轮询间隔秒数")
    parser.add_argument("--timeout", type=float, default=900.0, help="轮询超时时间（秒）")
    parser.add_argument("--debug", action="store_true", help="调试模式：打印完整响应结构")
    args = parser.parse_args()

    config = load_config_from_env()
    # 安全打印部分配置信息
    masked_key = f"{config['api_key'][:6]}...{config['api_key'][-4:]}" if len(config["api_key"]) > 10 else "***"
    print("配置信息：")
    print(f"  Base URL: {config['base_url']}")
    print(f"  API Key: {masked_key}")
    print(f"  Client Type: {config['client_type']}")

    client = create_openai_client(config)
    if not client:
        sys.exit(2)

    try:
        created = sora_create_video(client, model=args.model, prompt=args.prompt)
    except Exception:
        sys.exit(3)

    # Resolve default output path if not provided
    output_path = args.output
    if not output_path:
        slug = _slugify_from_prompt(args.prompt)
        default_dir = "/mnt/cfs/jj/musubi-tuner/scripts/sora2/results"
        os.makedirs(default_dir, exist_ok=True)
        output_path = os.path.join(default_dir, f"{slug}.mp4")
        print(f"未提供 --output，使用默认路径: {output_path}")

    if args.no_poll:
        # 不轮询时，若已返回 URL 则尝试直接下载
        url = extract_first_video_url(created)
        if url:
            download_file(url, output_path)
        sys.exit(0)

    # 轮询直到完成（若 SDK 支持）
    final = try_poll_until_done(client, created, poll_interval=args.poll_interval, timeout=args.timeout)
    url = extract_first_video_url(final) or extract_first_video_url(created)
    if url:
        download_file(url, output_path)
    else:
        # 尝试 file_id 下载
        file_ids = extract_file_ids(final) or extract_file_ids(created)
        if file_ids:
            print(f"未找到 URL，尝试通过 file_id 下载: {file_ids[0]}")
            if not download_via_file_id(client, file_ids[0], output_path):
                print("⚠️ 通过 file_id 下载失败")
        else:
            # 最后尝试：通过视频ID下载内容（若 SDK 支持 videos.content(video_id) ）
            vid = final.get("id") or created.get("id")
            if vid and hasattr(client, "videos") and hasattr(client.videos, "content"):
                print(f"未找到 URL 或 file_id，尝试通过 video_id 下载内容: {vid}")
                try:
                    content = client.videos.content(vid)  # type: ignore[attr-defined]
                    if hasattr(content, "stream_to_file"):
                        content.stream_to_file(output_path)  # type: ignore[attr-defined]
                        print("✅ 通过 video_id 下载完成 (stream_to_file)")
                    else:
                        try:
                            with client.videos.content(vid) as r:  # type: ignore
                                if hasattr(r, "stream_to_file"):
                                    r.stream_to_file(output_path)  # type: ignore[attr-defined]
                                    print("✅ 通过 video_id 下载完成 (context stream_to_file)")
                                else:
                                    data = r.read() if hasattr(r, "read") else None
                                    if data is not None:
                                        with open(output_path, "wb") as f:
                                            f.write(data)
                                        print("✅ 通过 video_id 下载完成 (read)")
                                    else:
                                        print("⚠️ videos.content 返回的对象不支持读/流，未保存")
                        except Exception:
                            # 回退：若 content 是 bytes
                            if isinstance(content, (bytes, bytearray)):
                                with open(output_path, "wb") as f:
                                    f.write(content)
                                print("✅ 通过 video_id 下载完成 (bytes)")
                            else:
                                print("⚠️ 通过 video_id 下载未成功")
                except Exception as e:
                    print(f"❌ 通过 video_id 下载失败: {e}")
            else:
                print("⚠️ 未找到可下载的 URL 或 file_id，尝试 REST 兜底下载。")
                ok = rest_download_by_video_id(config["api_key"], config["base_url"], vid, output_path) if vid else False
                if not ok:
                    print("⚠️ REST 兜底下载失败，SDK 也无法通过 video_id 下载。")
            if args.debug:
                try:
                    print("—— 完整响应（最终）——")
                    print(json.dumps(final, ensure_ascii=False, indent=2))
                except Exception:
                    print(str(final))
                try:
                    print("—— 完整响应（初始）——")
                    print(json.dumps(created, ensure_ascii=False, indent=2))
                except Exception:
                    print(str(created))

    sys.exit(0)


if __name__ == "__main__":
    main()
