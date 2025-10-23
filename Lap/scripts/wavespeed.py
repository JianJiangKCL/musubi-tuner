import os
import sys
import requests
import json
import time
import hashlib
import uuid
import asyncio
import argparse
import importlib.util
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

# Load upload_g2 module for image uploads
def load_upload_g2():
    """Load the upload_g2 module from the scripts directory"""
    # Try to find upload_g2.py in common locations
    possible_paths = [
        "/mnt/cfs/jj/musubi-tuner/scripts/upload_g2.py",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../scripts/upload_g2.py"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_g2.py"),
    ]
    
    for upload_g2_path in possible_paths:
        upload_g2_path = os.path.abspath(upload_g2_path)
        if os.path.exists(upload_g2_path):
            spec = importlib.util.spec_from_file_location("upload_g2", upload_g2_path)
            upload_g2 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(upload_g2)
            return upload_g2
    
    raise FileNotFoundError("Could not find upload_g2.py in expected locations")

# Load the upload module
try:
    upload_g2 = load_upload_g2()
except Exception as e:
    print(f"Warning: Could not load upload_g2 module: {e}")
    upload_g2 = None

def wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora):
    print("Hello from WaveSpeedAI!")
    API_KEY = os.getenv("WAVESPEED_API_KEY")
    print(f"API_KEY: {API_KEY}")

    url = "https://api.wavespeed.ai/api/v3/wavespeed-ai/wan-2.2/i2v-720p-lora"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "last_image": "",
        "high_noise_loras": [
            {
            "path": high_noise_lora,
            "scale": 1
            }
        ],
        "low_noise_loras": [
            {
            "path": low_noise_lora,
            "scale": 1
            }
        ],
        "duration": 8,
        "seed": 42
    }

    begin = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        print(f"Task submitted successfully. Request ID: {request_id}")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Poll for results
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                print(f"Task completed in {end - begin} seconds.")
                url = result["outputs"][0]
                print(f"Task completed. URL: {url}")
                break
            elif status == "failed":
                print(f"Task failed: {result.get('error')}")
                break
            else:
                print(f"Task still processing. Status: {status}")
        else:
            print(f"Error: {response.status_code}, {response.text}")
            break

        time.sleep(0.1)


def wavespeed_generate_high_noise_only(image, prompt, high_noise_lora):
    print("Hello from WaveSpeedAI!")
    API_KEY = os.getenv("WAVESPEED_API_KEY")
    print(f"API_KEY: {API_KEY}")

    url = "https://api.wavespeed.ai/api/v3/wavespeed-ai/wan-2.2/i2v-720p-lora"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "last_image": "",
        "high_noise_loras": [
            {
            "path": high_noise_lora,
            "scale": 1
            }
        ],
        "low_noise_loras": [],  # No low noise lora
        "duration": 8,
        "seed": 42
    }

    begin = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        print(f"Task submitted successfully. Request ID: {request_id}")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Poll for results
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                print(f"Task completed in {end - begin} seconds.")
                url = result["outputs"][0]
                print(f"Task completed. URL: {url}")
                break
            elif status == "failed":
                print(f"Task failed: {result.get('error')}")
                break
            else:
                print(f"Task still processing. Status: {status}")
        else:
            print(f"Error: {response.status_code}, {response.text}")
            break

        time.sleep(0.1)


def wavespeed_generate_simple(image, prompt, lora):
    print("Hello from WaveSpeedAI!")
    API_KEY = os.getenv("WAVESPEED_API_KEY")
    print(f"API_KEY: {API_KEY}")

    url = "https://api.wavespeed.ai/api/v3/wavespeed-ai/wan-2.2/i2v-720p-lora"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "last_image": "",
        "loras": [
            {
            "path": lora,
            "scale": 1
            }
        ],
        "high_noise_loras": [],
        "low_noise_loras": [],
        "duration": 5,
        "seed": -1
    }

    begin = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        print(f"Task submitted successfully. Request ID: {request_id}")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Poll for results
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                print(f"Task completed in {end - begin} seconds.")
                url = result["outputs"][0]
                print(f"Task completed. URL: {url}")
                break
            elif status == "failed":
                print(f"Task failed: {result.get('error')}")
                break
            else:
                print(f"Task still processing. Status: {status}")
        else:
            print(f"Error: {response.status_code}, {response.text}")
            break

        time.sleep(0.1)


def submit_wavespeed_task(image: str, prompt: str, lora: str, api_key: str) -> Optional[str]:
    """
    Submit a single wavespeed generation task and return the request ID.
    
    Args:
        image: URL or path to the image
        prompt: Text prompt for generation
        lora: URL to the lora checkpoint
        api_key: WaveSpeed API key
        
    Returns:
        Request ID if successful, None otherwise
    """
    url = "https://api.wavespeed.ai/api/v3/wavespeed-ai/wan-2.2/i2v-720p-lora"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "last_image": "",
        "high_noise_loras": [
            {
                "path": lora,
                "scale": 1
            }
        ],
        "low_noise_loras": [
            {
                "path": lora,
                "scale": 1
            }
        ],
        "loras": [],
        "duration": 8,
        "seed": 42
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()["data"]
            request_id = result["id"]
            print(f"✓ Task submitted successfully. Request ID: {request_id}, Prompt: {prompt[:50]}...")
            return request_id
        else:
            print(f"✗ Error submitting task: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"✗ Exception submitting task: {e}")
        return None


def poll_wavespeed_result(request_id: str, api_key: str, max_retries: int = 600) -> Optional[str]:
    """
    Poll for the result of a wavespeed generation task.
    
    Args:
        request_id: The request ID to poll
        api_key: WaveSpeed API key
        max_retries: Maximum number of polling attempts
        
    Returns:
        URL of the generated video if successful, None otherwise
    """
    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    begin = time.time()
    for i in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                result = response.json()["data"]
                status = result["status"]
                
                if status == "completed":
                    end = time.time()
                    video_url = result["outputs"][0]
                    print(f"✓ Task {request_id} completed in {end - begin:.2f}s. URL: {video_url}")
                    return video_url
                elif status == "failed":
                    print(f"✗ Task {request_id} failed: {result.get('error')}")
                    return None
                else:
                    if i % 10 == 0:  # Print status every 10 iterations
                        print(f"⋯ Task {request_id} still processing. Status: {status}")
            else:
                print(f"✗ Error polling task {request_id}: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            print(f"✗ Exception polling task {request_id}: {e}")
            return None
        
        time.sleep(1)  # Wait 1 second between polls
    
    print(f"✗ Task {request_id} timed out after {max_retries} attempts")
    return None


def upload_image_to_oss(image_path: str) -> Optional[str]:
    """
    Upload an image to OSS and return the URL.
    
    Args:
        image_path: Local path to the image file
        
    Returns:
        OSS URL if successful, None otherwise
    """
    if upload_g2 is None:
        print(f"✗ upload_g2 module not available, cannot upload {image_path}")
        return None
    
    try:
        print(f"📤 Uploading image: {os.path.basename(image_path)}")
        oss_url = upload_g2.upload_url(image_path, file_type='image')
        if oss_url:
            print(f"✓ Image uploaded: {oss_url}")
            return oss_url
        else:
            print(f"✗ Failed to upload image: {image_path}")
            return None
    except Exception as e:
        print(f"✗ Exception uploading image {image_path}: {e}")
        return None


def download_video_from_url(video_url: str, output_path: str) -> bool:
    """
    Download a video from a URL to a local path.
    
    Args:
        video_url: URL of the video to download
        output_path: Local path where the video should be saved
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"⬇️  Downloading video: {os.path.basename(output_path)}")
        
        # Download with streaming to handle large files
        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get total file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Write to file in chunks
        with open(output_path, 'wb') as f:
            downloaded = 0
            chunk_size = 8192
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        file_size = os.path.getsize(output_path)
        print(f"✓ Video downloaded: {output_path} ({file_size / 1024 / 1024:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"✗ Exception downloading video from {video_url}: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return False


async def process_single_item(item: Dict, lora: str, api_key: str, output_video_dir: str, executor: ThreadPoolExecutor) -> Dict:
    """
    Process a single item from the JSONL file asynchronously.
    
    Args:
        item: Dictionary containing video_path and caption
        lora: URL to the lora checkpoint
        api_key: WaveSpeed API key
        output_video_dir: Directory to save downloaded videos
        executor: ThreadPoolExecutor for running blocking operations
        
    Returns:
        Dictionary containing the original item data plus result URL
    """
    loop = asyncio.get_event_loop()
    
    # Extract first frame path (convert clips to first_frames)
    video_path = item.get('video_path', '')
    first_frame_path = video_path.replace('/clips/', '/first_frames/').replace('.mp4', '.jpg')
    
    # Get the video filename from the original video path
    video_filename = os.path.basename(video_path)
    local_video_path = os.path.join(output_video_dir, video_filename)
    
    if not os.path.exists(first_frame_path):
        print(f"✗ First frame not found: {first_frame_path}")
        return {
            **item,
            'first_frame_path': first_frame_path,
            'image_url': None,
            'request_id': None,
            'video_url': None,
            'local_video_path': None,
            'error': 'First frame not found'
        }
    
    # Upload image to OSS
    image_url = await loop.run_in_executor(
        executor, upload_image_to_oss, first_frame_path
    )
    
    if image_url is None:
        return {
            **item,
            'first_frame_path': first_frame_path,
            'image_url': None,
            'request_id': None,
            'video_url': None,
            'local_video_path': None,
            'error': 'Failed to upload image'
        }
    
    caption = item.get('caption', '')
    
    # Submit task with uploaded image URL
    request_id = await loop.run_in_executor(
        executor, submit_wavespeed_task, image_url, caption, lora, api_key
    )
    
    if request_id is None:
        return {
            **item,
            'first_frame_path': first_frame_path,
            'image_url': image_url,
            'request_id': None,
            'video_url': None,
            'local_video_path': None,
            'error': 'Failed to submit task'
        }
    
    # Poll for result
    video_url = await loop.run_in_executor(
        executor, poll_wavespeed_result, request_id, api_key
    )
    
    if video_url is None:
        return {
            **item,
            'first_frame_path': first_frame_path,
            'image_url': image_url,
            'request_id': request_id,
            'video_url': None,
            'local_video_path': None,
            'error': 'Video generation failed'
        }
    
    # Download the generated video
    download_success = await loop.run_in_executor(
        executor, download_video_from_url, video_url, local_video_path
    )
    
    return {
        **item,
        'first_frame_path': first_frame_path,
        'image_url': image_url,
        'request_id': request_id,
        'video_url': video_url,
        'local_video_path': local_video_path if download_success else None,
        'error': None if download_success else 'Video download failed'
    }


async def process_batch(items: List[Dict], lora: str, api_key: str, output_video_dir: str, batch_size: int = 8) -> List[Dict]:
    """
    Process a batch of items concurrently.
    
    Args:
        items: List of items to process
        lora: URL to the lora checkpoint
        api_key: WaveSpeed API key
        output_video_dir: Directory to save downloaded videos
        batch_size: Number of concurrent requests
        
    Returns:
        List of results
    """
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        tasks = [
            process_single_item(item, lora, api_key, output_video_dir, executor)
            for item in items
        ]
        results = await asyncio.gather(*tasks)
    
    return results


def process_jsonl_file(jsonl_path: str, lora_url: str, output_path: str, output_video_dir: str, batch_size: int = 8):
    """
    Process the JSONL file in batches.
    
    Args:
        jsonl_path: Path to the input JSONL file
        lora_url: URL to the lora checkpoint
        output_path: Path to the output JSONL file
        output_video_dir: Directory to save downloaded videos
        batch_size: Number of concurrent requests per batch
    """
    API_KEY = os.getenv("WAVESPEED_API_KEY")
    if not API_KEY:
        print("Error: WAVESPEED_API_KEY not found in environment variables")
        return
    
    # Create output video directory if it doesn't exist
    os.makedirs(output_video_dir, exist_ok=True)
    
    print(f"Reading items from {jsonl_path}...")
    
    # Read all items from JSONL
    items = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
    
    print(f"Found {len(items)} items to process")
    print(f"Processing in batches of {batch_size}...")
    print(f"Videos will be saved to: {output_video_dir}")
    
    # Load existing results for checkpoint resume
    all_results = []
    processed_video_paths = set()
    
    if os.path.exists(output_path):
        print(f"\n🔄 Found existing output file: {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        result = json.loads(line)
                        all_results.append(result)
                        # Track which videos have been processed
                        if 'video_path' in result:
                            processed_video_paths.add(result['video_path'])
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse existing result line: {e}")
        
        print(f"✓ Loaded {len(all_results)} existing results")
        print(f"✓ Resuming from checkpoint...")
    else:
        print(f"\n🆕 Starting fresh processing...")
    
    # Filter out already processed items
    items_to_process = [item for item in items if item.get('video_path') not in processed_video_paths]
    skipped = len(items) - len(items_to_process)
    
    if skipped > 0:
        print(f"⏭️  Skipping {skipped} already processed items")
    
    print(f"📝 {len(items_to_process)} items remaining to process")
    
    if len(items_to_process) == 0:
        print("\n🎉 All items already processed!")
        return
    
    # Process in batches
    for i in range(0, len(items_to_process), batch_size):
        batch = items_to_process[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(items_to_process) + batch_size - 1) // batch_size
        
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)...")
        print(f"Overall progress: {len(all_results)}/{len(items)} completed, {len(items_to_process) - i} remaining")
        print(f"{'='*60}")
        
        # Run async batch processing
        results = asyncio.run(process_batch(batch, lora_url, API_KEY, output_video_dir, batch_size))
        all_results.extend(results)
        
        # Save intermediate results after each batch
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"✓ Batch {batch_num}/{total_batches} completed. Results saved to {output_path}")
        print(f"📊 Current session: {i + len(batch)}/{len(items_to_process)}, Total: {len(all_results)}/{len(items)}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Total items processed: {len(all_results)}/{len(items)}")
    successful = sum(1 for r in all_results if r.get('video_url'))
    downloaded = sum(1 for r in all_results if r.get('local_video_path'))
    failed = len(all_results) - successful
    print(f"Successful generations: {successful}")
    print(f"Downloaded videos: {downloaded}")
    print(f"Failed: {failed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Results saved to: {output_path}")
    print(f"Videos saved to: {output_video_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate videos from captions using WaveSpeed API with LoRA checkpoint.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s https://liblibai-tmp-image.liblib.cloud/models/chaowei/9e72e9d05b3949268387b152401962c9/8beb364f35b847efa7d7c61b12c9bdba.safetensors
        """
    )
    
    parser.add_argument(
        'lora_url',
        type=str,
        help='URL to the LoRA checkpoint (e.g., https://liblibai-tmp-image.liblib.cloud/models/...)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='/mnt/cfs/jj/musubi-tuner/test_outputs/tmp_data/preprocessing/filtered_clips_processed.jsonl',
        help='Path to input JSONL file (default: filtered_clips_processed.jsonl)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='/mnt/cfs/jj/musubi-tuner/test_outputs/tmp_data/preprocessing/wavespeed_results.jsonl',
        help='Path to output JSONL file (default: wavespeed_results.jsonl)'
    )
    
    parser.add_argument(
        '--video-output-dir',
        type=str,
        default='/mnt/cfs/jj/musubi-tuner/test_outputs/tmp_data/video_outputs/allvideos_20s',
        help='Directory to save downloaded videos (default: video_outputs/allvideos_20s)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Number of concurrent requests per batch (default: 8)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create video output directory
    os.makedirs(args.video_output_dir, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Input JSONL: {args.input}")
    print(f"  Output JSONL: {args.output}")
    print(f"  Video output dir: {args.video_output_dir}")
    print(f"  LoRA URL: {args.lora_url}")
    print(f"  Batch size: {args.batch_size}")
    print()
    
    process_jsonl_file(args.input, args.lora_url, args.output, args.video_output_dir, args.batch_size)


if __name__ == "__main__":
    main()
