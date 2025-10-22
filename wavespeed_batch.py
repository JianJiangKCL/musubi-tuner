import os
import requests
import json
import time
import hashlib
import uuid
import subprocess
import re
from urllib.parse import urlparse
from pathlib import Path
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dotenv import load_dotenv

load_dotenv()

# Thread-safe printing
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

def wavespeed_generate_batch(image, prompt, high_noise_lora, low_noise_lora, output_name):
    """Generate video using Wavespeed API with high and low noise LoRAs"""
    safe_print(f"Processing: {output_name}")
    safe_print("Hello from WaveSpeedAI!")
    API_KEY = os.getenv("WAVESPEED_API_KEY")
    
    if not API_KEY:
        safe_print("Error: WAVESPEED_API_KEY not found in environment variables")
        return None
    
    safe_print(f"API_KEY: {API_KEY[:10]}...")

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
        "duration": 5,
        "seed": 42
    }

    begin = time.time()
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()["data"]
            request_id = result["id"]
            safe_print(f"Task submitted successfully. Request ID: {request_id}")
        else:
            safe_print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        safe_print(f"Error submitting request: {e}")
        return None

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Poll for results
    while True:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                result = response.json()["data"]
                status = result["status"]

                if status == "completed":
                    end = time.time()
                    safe_print(f"Task completed in {end - begin:.2f} seconds.")
                    video_url = result["outputs"][0]
                    safe_print(f"Task completed. URL: {video_url}")
                    return video_url
                elif status == "failed":
                    safe_print(f"Task failed: {result.get('error')}")
                    return None
                else:
                    safe_print(f"Task still processing. Status: {status}")
            else:
                safe_print(f"Error: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            safe_print(f"Error polling results: {e}")
            return None

        time.sleep(5)  # Poll every 5 seconds instead of 0.1 to be more reasonable


def upload_lora_model(lora_path):
    """Upload LoRA model and return the OSS URL"""
    safe_print(f"Uploading LoRA model: {lora_path}")
    
    try:
        # Use the upload_model_cli.py script
        upload_script = "/mnt/cfs/jj/musubi-tuner/scripts/upload_model_cli.py"
        result = subprocess.run([
            "python3", upload_script, lora_path, "--quiet"
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            # Extract the final URL from the output
            # The --quiet flag should only output the URL, but let's be safe
            output_lines = result.stdout.strip().split('\n')
            
            # Find the last line that looks like a URL
            oss_url = None
            for line in reversed(output_lines):
                line = line.strip()
                if line.startswith('https://') and 'liblibai-tmp-image.liblib.cloud' in line:
                    oss_url = line
                    break
            
            if oss_url:
                safe_print(f"✅ Upload successful: {oss_url}")
                return oss_url
            else:
                safe_print(f"❌ Upload failed: Could not extract URL from output")
                safe_print(f"Output was: {result.stdout}")
                return None
        else:
            safe_print(f"❌ Upload failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        safe_print("❌ Upload timed out after 10 minutes")
        return None
    except Exception as e:
        safe_print(f"❌ Upload error: {e}")
        return None


def process_single_task(task_info):
    """Process a single video generation task"""
    task_number, lora_name, lora_path, oss_url, image_index, image_url, caption = task_info
    
    output_name = f"{lora_name}_image{image_index}"
    
    safe_print(f"\n--- Task {task_number} ---")
    safe_print(f"Processing {output_name}")
    safe_print(f"Image URL: {image_url}")
    
    try:
        # Generate video using the OSS URL for both high and low noise
        generation_start = datetime.now()
        video_url = wavespeed_generate_batch(
            image=image_url,
            prompt=caption,
            high_noise_lora=oss_url,
            low_noise_lora=oss_url,
            output_name=output_name
        )
        generation_end = datetime.now()
        
        # Base result structure
        result = {
            'task_number': task_number,
            'lora_name': lora_name,
            'lora_path': lora_path,
            'lora_oss_url': oss_url,
            'image_index': image_index,
            'image_url': image_url,
            'prompt': caption,
            'video_url': video_url,
            'output_name': output_name,
            'generation_start_time': generation_start.isoformat(),
            'generation_end_time': generation_end.isoformat(),
            'generation_duration_seconds': (generation_end - generation_start).total_seconds(),
            'timestamp': datetime.now().isoformat()
        }
        
        if video_url:
            result['status'] = 'success'
            safe_print(f"✅ Successfully generated video for {output_name}")
            safe_print(f"   Video URL: {video_url}")
        else:
            result['status'] = 'failed'
            result['video_url'] = None
            safe_print(f"❌ Failed to generate video for {output_name}")
        
        return result
        
    except Exception as e:
        safe_print(f"❌ Error processing {output_name}: {e}")
        result = {
            'task_number': task_number,
            'lora_name': lora_name,
            'lora_path': lora_path,
            'lora_oss_url': oss_url,
            'image_index': image_index,
            'image_url': image_url,
            'prompt': caption,
            'video_url': None,
            'output_name': output_name,
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        return result


def load_captions_from_jsonl(jsonl_path):
    """Load captions from a JSONL file"""
    captions = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'caption' in data:
                    captions.append(data['caption'])
        return captions
    except Exception as e:
        print(f"Error loading captions from {jsonl_path}: {e}")
        return []


def find_lora_files(folder_path):
    """Find all LoRA files in subfolders, prioritizing main LoRA files without numbers"""
    lora_files = []
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return []
    
    # Look for .safetensors files in subfolders
    for subfolder in folder_path.iterdir():
        if subfolder.is_dir():
            subfolder_name = subfolder.name
            safetensors_files = list(subfolder.glob("*.safetensors"))
            
            if safetensors_files:
                # Look for the main LoRA file (without numbers)
                # Priority: {subfolder_name}-lora.safetensors
                preferred_name = f"{subfolder_name}-lora.safetensors"
                preferred_file = subfolder / preferred_name
                
                if preferred_file.exists():
                    lora_file = preferred_file
                    safe_print(f"Found main LoRA: {subfolder_name} -> {lora_file}")
                else:
                    # Filter out files with numbers (like -000001.safetensors)
                    main_files = []
                    for f in safetensors_files:
                        # Check if filename ends with -XXXXXX pattern (6 digits)
                        stem = f.stem  # filename without extension
                        
                        # Skip files that match pattern: *-000001, *-000002, etc.
                        if re.search(r'-\d{6}$', stem):
                            safe_print(f"  Skipping numbered file: {f.name}")
                            continue
                        
                        main_files.append(f)
                    
                    if main_files:
                        # Take the first non-numbered file
                        lora_file = main_files[0]
                        safe_print(f"Found LoRA (non-numbered): {subfolder_name} -> {lora_file}")
                    else:
                        # Fallback to first file if no main file found
                        lora_file = safetensors_files[0]
                        safe_print(f"Found LoRA (fallback): {subfolder_name} -> {lora_file}")
                        safe_print(f"  Warning: Only numbered files found, using fallback")
                
                lora_files.append({
                    'name': subfolder_name,
                    'path': str(lora_file),
                    'subfolder': str(subfolder)
                })
            else:
                safe_print(f"Warning: No .safetensors files found in {subfolder}")
    
    return lora_files


def get_caption_for_lora(lora_name, datasets_folder):
    """Get caption for a specific LoRA from corresponding JSONL file"""
    jsonl_path = Path(datasets_folder) / f"{lora_name}.jsonl"
    
    if not jsonl_path.exists():
        safe_print(f"Warning: Caption file {jsonl_path} not found for {lora_name}")
        return f"Default prompt for {lora_name}"
    
    captions = load_captions_from_jsonl(jsonl_path)
    if captions:
        # Use the first caption found
        return captions[0]
    else:
        return f"Default prompt for {lora_name}"


def process_batch_generation():
    """Main function to process all LoRA models and generate videos"""
    # Configuration
    lora_folder = "/mnt/cfs/jj/musubi-tuner/lora_outputs/to_improve"
    datasets_folder = "/mnt/cfs/jj/musubi-tuner/datasets/to_improve"
    
    # Two test images
    images = [
        "https://liblibai-tmp-image.liblib.cloud/sd-images/81836526-458b-44a9-b065-1ef36f326a65.png",
        "https://liblibai-tmp-image.liblib.cloud/sd-images/850eb8ae-2e82-4b1b-bf3f-236a46e96b20.png"
    ]
    
    # Find all LoRA files
    lora_files = find_lora_files(lora_folder)
    
    if not lora_files:
        print("No LoRA files found!")
        return
    
    print(f"Found {len(lora_files)} LoRA models to process")
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"wavespeed_batch_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize batch metadata
    batch_metadata = {
        "batch_start_time": datetime.now().isoformat(),
        "total_lora_models": len(lora_files),
        "total_images": len(images),
        "total_tasks": len(lora_files) * len(images),
        "images_used": images,
        "lora_folder": lora_folder,
        "datasets_folder": datasets_folder
    }
    
    results = []
    
    total_tasks = len(lora_files) * len(images)
    current_task = 0
    
    # Phase 1: Upload all LoRA models in parallel
    safe_print(f"\n{'='*60}")
    safe_print("PHASE 1: UPLOADING ALL LORA MODELS")
    safe_print(f"{'='*60}")
    
    # Upload all LoRAs in parallel (max 3 concurrent uploads to avoid overwhelming the system)
    upload_tasks = [(lora_info['name'], lora_info['path']) for lora_info in lora_files]
    uploaded_loras = {}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit upload tasks
        future_to_lora = {
            executor.submit(upload_lora_model, lora_path): lora_name 
            for lora_name, lora_path in upload_tasks
        }
        
        for future in as_completed(future_to_lora):
            lora_name = future_to_lora[future]
            lora_path = next(info['path'] for info in lora_files if info['name'] == lora_name)
            
            try:
                oss_url = future.result()
                if oss_url:
                    uploaded_loras[lora_name] = {
                        'path': lora_path,
                        'oss_url': oss_url,
                        'caption': get_caption_for_lora(lora_name, datasets_folder)
                    }
                    safe_print(f"✅ {lora_name} uploaded successfully")
                else:
                    safe_print(f"❌ {lora_name} upload failed")
                    # Add failed upload results for both images
                    for i in range(len(images)):
                        current_task += 1
                        result = {
                            'task_number': current_task,
                            'lora_name': lora_name,
                            'lora_path': lora_path,
                            'lora_oss_url': None,
                            'image_index': i + 1,
                            'image_url': images[i],
                            'prompt': 'N/A - Upload failed',
                            'video_url': None,
                            'output_name': f"{lora_name}_image{i+1}",
                            'status': 'upload_failed',
                            'timestamp': datetime.now().isoformat()
                        }
                        results.append(result)
            except Exception as e:
                safe_print(f"❌ {lora_name} upload error: {e}")
    
    safe_print(f"\nUploaded {len(uploaded_loras)} out of {len(lora_files)} LoRA models")
    
    # Phase 2: Generate videos in parallel for all uploaded LoRAs
    if uploaded_loras:
        safe_print(f"\n{'='*60}")
        safe_print("PHASE 2: GENERATING VIDEOS IN PARALLEL")
        safe_print(f"{'='*60}")
        
        # Create all video generation tasks
        video_tasks = []
        for lora_name, lora_data in uploaded_loras.items():
            for i, image_url in enumerate(images):
                current_task += 1
                task_info = (
                    current_task,
                    lora_name,
                    lora_data['path'],
                    lora_data['oss_url'],
                    i + 1,
                    image_url,
                    lora_data['caption']
                )
                video_tasks.append(task_info)
        
        safe_print(f"Starting {len(video_tasks)} video generation tasks...")
        
        # Process video generation tasks in parallel (max 4 concurrent to avoid API limits)
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_task = {
                executor.submit(process_single_task, task_info): task_info 
                for task_info in video_tasks
            }
            
            completed_count = 0
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    safe_print(f"Completed {completed_count}/{len(video_tasks)} tasks")
                except Exception as e:
                    task_info = future_to_task[future]
                    safe_print(f"❌ Task failed with exception: {e}")
                    # Create error result
                    error_result = {
                        'task_number': task_info[0],
                        'lora_name': task_info[1],
                        'lora_path': task_info[2],
                        'lora_oss_url': task_info[3],
                        'image_index': task_info[4],
                        'image_url': task_info[5],
                        'prompt': task_info[6],
                        'video_url': None,
                        'output_name': f"{task_info[1]}_image{task_info[4]}",
                        'status': 'error',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(error_result)
    else:
        safe_print("❌ No LoRAs were uploaded successfully. Skipping video generation.")
    
    # Complete batch metadata
    batch_metadata.update({
        "batch_end_time": datetime.now().isoformat(),
        "batch_duration_seconds": (datetime.now() - datetime.fromisoformat(batch_metadata["batch_start_time"])).total_seconds(),
        "results_summary": {
            "total_tasks": total_tasks,
            "successful": sum(1 for r in results if r['status'] == 'success'),
            "failed": sum(1 for r in results if r['status'] in ['failed', 'error']),
            "upload_failed": sum(1 for r in results if r['status'] == 'upload_failed'),
            "success_rate": sum(1 for r in results if r['status'] == 'success') / total_tasks * 100 if total_tasks > 0 else 0
        }
    })
    
    # Save comprehensive results to JSON file
    comprehensive_results = {
        "metadata": batch_metadata,
        "results": results
    }
    
    results_file = output_dir / "batch_results_comprehensive.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
    
    # Also save a simple results-only file for easier processing
    simple_results_file = output_dir / "batch_results_simple.json"
    with open(simple_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save successful results only (for quick access to working videos)
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        success_file = output_dir / "successful_videos.json"
        with open(success_file, 'w', encoding='utf-8') as f:
            json.dump(successful_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total tasks: {total_tasks}")
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] in ['failed', 'error'])
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/total_tasks*100:.1f}%")
    print(f"\nResults saved to:")
    print(f"  📊 Comprehensive: {results_file}")
    print(f"  📋 Simple: {simple_results_file}")
    if successful_results:
        print(f"  ✅ Success only: {success_file}")
    print(f"  📁 Output directory: {output_dir}")
    
    # Print successful results
    print(f"\n{'='*60}")
    print("SUCCESSFUL GENERATIONS:")
    print(f"{'='*60}")
    for result in results:
        if result['status'] == 'success':
            print(f"✅ {result['output_name']}")
            print(f"   LoRA: {result['lora_name']}")
            print(f"   LoRA OSS: {result['lora_oss_url']}")
            print(f"   Prompt: {result['prompt'][:80]}...")
            print(f"   Video: {result['video_url']}")
            print(f"   Duration: {result['generation_duration_seconds']:.1f}s")
            print()
    
    # Print upload failures
    upload_failures = [r for r in results if r['status'] == 'upload_failed']
    if upload_failures:
        print(f"\n{'='*60}")
        print("UPLOAD FAILURES:")
        print(f"{'='*60}")
        failed_loras = set(r['lora_name'] for r in upload_failures)
        for lora_name in failed_loras:
            print(f"❌ {lora_name} - Failed to upload to OSS")


if __name__ == "__main__":
    print("Starting Wavespeed Batch Processing...")
    print("This will process all LoRA models in /mnt/cfs/jj/musubi-tuner/lora_outputs/to_improve")
    print("and generate videos for 2 different images using captions from datasets/to_improve")
    print()
    
    # Ask for confirmation
    response = input("Do you want to continue? (y/N): ")
    if response.lower() in ['y', 'yes']:
        process_batch_generation()
    else:
        print("Batch processing cancelled.")
