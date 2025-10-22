import os
import requests
import json
import time
import hashlib
import uuid
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

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
        "duration": 8,
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


if __name__ == "__main__":
    '''
    高低噪
    '''
    # image = "https://liblibai-tmp-image.liblib.cloud/img/081e9f07d9bd4c2ba090efde163518f9/7eb0ca3f-44c9-4c50-9041-5f062241de8b.png"
    # prompt = "红发女性面向镜头,她的手臂从肩膀开始逐渐浮现裂纹状纹路,纹路如同MedusaGorgona般向颈部和脸部蔓延,同时她的头发从发梢开始变为灰白色并逐渐向上延伸,随着裂纹覆盖她的脖颈,脸颊乃至手部,她的发色完全转为银白色,最终全身都布满了类似石像的裂纹纹理,最终她完全化身石像保持静止,镜头缓慢拉远，展现她凝固的瞬间，光影柔和而神秘，营造出超现实的氛围。"
    # high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/87d76a57229d8e1d99afc995c0d52322fa042388b8191c3b1aa8fd39865d22cf.safetensors"
    # low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/87d76a57229d8e1d99afc995c0d52322fa042388b8191c3b1aa8fd39865d22cf.safetensors"
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)

    # '''
    # 单lora
    # '''
    # image = "https://liblibai-tmp-image.liblib.cloud/img/081e9f07d9bd4c2ba090efde163518f9/7eb0ca3f-44c9-4c50-9041-5f062241de8b.png"
    # prompt = "红发女性面向镜头,她的手臂从肩膀开始逐渐浮现裂纹状纹路,纹路如同MedusaGorgona般向颈部和脸部蔓延,同时她的头发从发梢开始变为灰白色并逐渐向上延伸,随着裂纹覆盖她的脖颈,脸颊乃至手部,她的发色完全转为银白色,最终全身都布满了类似石像的裂纹纹理,最终她完全化身石像保持静止,镜头缓慢拉远，展现她凝固的瞬间，光影柔和而神秘，营造出超现实的氛围。"
    # lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/87d76a57229d8e1d99afc995c0d52322fa042388b8191c3b1aa8fd39865d22cf.safetensors"

    high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/0622c1ab1c884125abd22ed1f566822e/dda589c37c0f4521b5ccd54dc1e8c72b.safetensors"
    low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/0622c1ab1c884125abd22ed1f566822e/dda589c37c0f4521b5ccd54dc1e8c72b.safetensors"
    image = "https://liblibai-tmp-image.liblib.cloud/sd-images/f30caee9-c678-4471-9cbe-1c4b3a912ba8.png"
    prompt = "A person gradually transforms into a stone statue as crack-like textures spread across their body from a starting point, changing their skin and clothing to gray stone material with a rough, cracked surface texture, similar to the Medusa Gorgon petrification effect, while maintaining their final pose."
    wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
    
    
    
    
    
