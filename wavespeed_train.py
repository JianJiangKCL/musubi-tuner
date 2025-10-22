import os
import requests
import json
import time
import zipfile
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def load_training_data(jsonl_path):
    """Load training data from JSONL file"""
    training_data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            training_data.append(data)
    return training_data


def create_training_zip(training_data, output_zip_path):
    """Create a ZIP file containing videos and caption files for training"""
    temp_dir = Path("temp_training_data")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Copy videos and create caption files
        for i, item in enumerate(training_data):
            video_path = Path(item['video_path'])
            caption = item['caption']
            
            # Copy video file
            if video_path.exists():
                video_name = video_path.name
                shutil.copy2(video_path, temp_dir / video_name)
                
                # Create corresponding caption file
                caption_name = video_path.stem + ".txt"
                with open(temp_dir / caption_name, 'w') as f:
                    f.write(caption)
                    
                print(f"Added {video_name} and {caption_name}")
            else:
                print(f"Warning: Video file not found: {video_path}")
        
        # Create ZIP file
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in temp_dir.iterdir():
                zipf.write(file_path, file_path.name)
        
        print(f"Created training ZIP: {output_zip_path}")
        return True
        
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def upload_to_temp_host(zip_path):
    """Upload ZIP file to a temporary file hosting service"""
    print("Uploading to temporary file hosting service...")
    
    # Try multiple temporary file hosting services
    services = [
        {
            'name': 'tmpfiles.org',
            'url': 'https://tmpfiles.org/api/v1/upload',
            'method': 'tmpfiles'
        },
        {
            'name': '0x0.st',
            'url': 'https://0x0.st',
            'method': '0x0'
        }
    ]
    
    for service in services:
        try:
            print(f"Trying {service['name']}...")
            
            with open(zip_path, 'rb') as f:
                if service['method'] == 'tmpfiles':
                    files = {'file': f}
                    response = requests.post(service['url'], files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('status') == 'success':
                            file_url = result.get('data', {}).get('url')
                            # Convert tmpfiles.org URL to direct download format
                            if 'tmpfiles.org' in file_url:
                                # Change from http://tmpfiles.org/ID/filename to https://tmpfiles.org/dl/ID/filename
                                file_url = file_url.replace('http://tmpfiles.org/', 'https://tmpfiles.org/dl/')
                            print(f"File uploaded successfully to {service['name']}: {file_url}")
                            return file_url
                
                elif service['method'] == '0x0':
                    files = {'file': ('training.zip', f, 'application/zip')}
                    response = requests.post(service['url'], files=files)
                    
                    if response.status_code == 200:
                        file_url = response.text.strip()
                        print(f"File uploaded successfully to {service['name']}: {file_url}")
                        return file_url
                        
        except Exception as e:
            print(f"Failed to upload to {service['name']}: {e}")
            continue
    
    print("All upload services failed")
    return None


def main():
    print("Hello from WaveSpeedAI Training!")
    API_KEY = os.getenv("WAVESPEED_API_KEY")
    print(f"API_KEY: {API_KEY}")
    
    if not API_KEY:
        print("Error: WAVESPEED_API_KEY not found in environment variables")
        return

    # Load training data
    data_path = "/mnt/cfs/jj/musubi-tuner/Chaowei/datasets_sameTag/Earth_Zoom_Out_sameTag/video_caption_rectified_v3.jsonl"
    training_data = load_training_data(data_path)
    print(f"Loaded {len(training_data)} training samples")
    
    # Display training data info
    for i, item in enumerate(training_data):
        print(f"Sample {i+1}: {os.path.basename(item['video_path'])}")
        print(f"Caption preview: {item['caption'][:100]}...")
        print()

    # Create and upload training ZIP file
    zip_path = "earth_zoom_out_training.zip"
    print("Creating training ZIP file...")
    
    if not create_training_zip(training_data, zip_path):
        print("Failed to create training ZIP file")
        return
    
    print("Uploading training data...")
    data_url = upload_to_temp_host(zip_path)
    if not data_url:
        print("Failed to upload training data")
        print("Please upload the ZIP file manually to a file hosting service and update the data_url variable")
        print(f"ZIP file created: {zip_path}")
        return
    
    # Clean up local ZIP file
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"Cleaned up local ZIP file: {zip_path}")

    url = "https://api.wavespeed.ai/api/v3/wavespeed-ai/wan-2.2-i2v-lora-trainer"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    
    # Training configuration based on Earth zoom-out dataset
    payload = {
        "data": data_url,  # URL of uploaded ZIP file
        "trigger_word": "earth_zoom_out",  # Trigger word for the Earth zoom-out effect
        "steps": 500,  # Increased steps for better training with video data
        "learning_rate": 0.0001,  # Lower learning rate for video training
        "lora_rank": 64,  # Higher rank for better video quality
    }

    print("Starting training with configuration:")
    print(f"- Data URL: {data_url}")
    print(f"- Trigger word: {payload['trigger_word']}")
    print(f"- Steps: {payload['steps']}")
    print(f"- Learning rate: {payload['learning_rate']}")
    print(f"- LoRA rank: {payload['lora_rank']}")
    print(f"- Training samples: {len(training_data)}")
    print()

    begin = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        print(f"Training task submitted successfully. Request ID: {request_id}")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Poll for results
    print("Polling for training completion...")
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                print(f"Training completed in {end - begin:.2f} seconds ({(end - begin)/60:.2f} minutes).")
                
                # Handle multiple outputs if available
                outputs = result.get("outputs", [])
                if outputs:
                    for i, output_url in enumerate(outputs):
                        print(f"Model output {i+1}: {output_url}")
                else:
                    print("No outputs available")
                break
            elif status == "failed":
                error_msg = result.get('error', 'Unknown error')
                print(f"Training failed: {error_msg}")
                break
            else:
                elapsed = time.time() - begin
                print(f"Training in progress. Status: {status} (elapsed: {elapsed:.1f}s)")
        else:
            print(f"Error polling status: {response.status_code}, {response.text}")
            break

        time.sleep(5)  # Poll every 5 seconds for training tasks


if __name__ == "__main__":
    main()
