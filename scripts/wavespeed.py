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
        "duration": 5,
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
    #####
    # #### stone
    # high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/fdf8f61dd9a342608488a83b92449d6e/dbd1d597b5414b12a0aea57f15aefcf5.safetensors"
    # low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/fdf8f61dd9a342608488a83b92449d6e/dbd1d597b5414b12a0aea57f15aefcf5.safetensors"
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/f30caee9-c678-4471-9cbe-1c4b3a912ba8.png"
    # # # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/c07fd4eb-a4f3-4966-9df5-5b26d0f78d2a.png"
    # # # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/5433772b-9020-4da0-9105-68026bb83a99.png"
    # # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/81836526-458b-44a9-b065-1ef36f326a65.png"
    # # prompt = "A person gradually transforms into a stone statue as the effect spreads across their body from a starting point, changing their skin and clothing to gray stone material with minimal surface texture, similar to the Medusa Gorgon petrification effect, while maintaining their final pose."
    # prompt = "A person gradually transforms into a stone statue as crack-like textures spread across their body from a starting point, changing their skin and clothing to gray stone material with a rough, cracked surface texture, similar to the Medusa Gorgon petrification effect, while maintaining their final pose."
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
    
    
    
    # high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/8eafe73364c648d584f3d400f1c29cd9/4d89f87520a94d048954050b43e0ebbc.safetensors"
    # low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/8eafe73364c648d584f3d400f1c29cd9/4d89f87520a94d048954050b43e0ebbc.safetensors"
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/88fe3059-5646-43c4-a725-4201b97175cf.png"
    # # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/f30caee9-c678-4471-9cbe-1c4b3a912ba8.png"
    # prompt = "A fist rapidly approaches and punches a person's face while the camera performs a dramatic crash zoom in, causing the person's expression to change to shock or pain as particles, debris, or other materials fly from the impact point."
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
    

    
    

    # high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/54ca9fb8c4d44801b42eb68d868c4b48/1284fa85b73346a6992f6065bd48e71a.safetensors"
    # low_noise_lora  = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/54ca9fb8c4d44801b42eb68d868c4b48/1284fa85b73346a6992f6065bd48e71a.safetensors"
    # # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/81836526-458b-44a9-b065-1ef36f326a65.png"
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/e5c5522b-d2ff-4465-9e2d-351e2c049fe9.png"
    # prompt = "Black matter or shadow begins to appear on the figure from the very bottom, gradually spreading upward to cover the entire body or upper body, ultimately enveloping them completely in black, transforming into a Shadow form, black silhouette or outline, merging into the shadows."
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
    
    # high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/a4604feabba84f73802042dc80c852b6/ee242fd1507a4a0585aa7faef54fe739.safetensors"
    # low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/a4604feabba84f73802042dc80c852b6/ee242fd1507a4a0585aa7faef54fe739.safetensors"
    # # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/81836526-458b-44a9-b065-1ef36f326a65.png"
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/e5c5522b-d2ff-4465-9e2d-351e2c049fe9.png"
    # prompt = "Black matter or shadow begins to appear on the figure from the very bottom, quickly spreading upward to cover the entire body or upper body, ultimately enveloping them completely in black with the face and head fully covered by shadows, transforming into a Shadow form."
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
    
    # high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/69d0149d6a504dbabdddfcd9c0446b05/1208ba58aac34b1da59f8231a28dafa7.safetensors"
    # low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/69d0149d6a504dbabdddfcd9c0446b05/1208ba58aac34b1da59f8231a28dafa7.safetensors"
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/2b0a0c70-45ef-4d82-b6ef-872de9c4991b.png"
    # # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/f30caee9-c678-4471-9cbe-1c4b3a912ba8.png"
    # prompt = "The main character  remains still, and the animated character Buddy appears from behind them. Buddy wears the same style clothing as the main character, making gestures such as smiling, waving, moving, or placing a hand on their shoulder. The main character turns their head to look at Buddy."
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
    
    # high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/8b79bfe20c584fe6a729180c01c3f704/4031e1f2067a4815ba7382c24050a75f.safetensors"
    # low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/8b79bfe20c584fe6a729180c01c3f704/4031e1f2067a4815ba7382c24050a75f.safetensors"
    # # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/81836526-458b-44a9-b065-1ef36f326a65.png"
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/e44362f3-236c-4b84-9de9-96d145739184.png"
    # prompt = "A person stands in the foreground as a massive atomic explosion erupts in the background, creating a bright flash followed by an expanding orange-yellow mush    high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/d9e2d06edf97432ea63a218e36737906/61c0a58cd2744a149fb987f7ce5542b7.safetensors"
    # low_noise_lora = "https://d2p7pge43lyniu.cloudfront.net/output/41ecf732-bf2e-4b44-b61a-f78670bf3161-u1_i2v_A14B_separate_low_noise_lora_ce03fb69-1611-47be-8cc9-f7f4fe1ff163.safetensors"
    # high_noise_lora = "https://d2p7pge43lyniu.cloudfront.net/output/41ecf732-bf2e-4b44-b61a-f78670bf3161-u1_i2v_A14B_separate_high_noise_lora_099cacd5-716f-4d5a-8c49-d0129793d2b9.safetensors"
   
    # prompt = "A continuous zoom-out shot from a fixed vantage point in space. The camera starts with an extreme close-up view of the main character, then rapidly pulls back while maintaining the same orbital perspective. As the focal length decreases, the person quickly becomes smaller and smaller until they disappear from view entirely, absorbed into the expanding landscape. More of the surrounding terrain becomes visible: first the immediate area, then neighboring buildings or landscape features, rapidly revealing the broader region, country borders, and continental outlines. The zoom creates a smooth transition from an impossibly detailed telescopic view to a final panoramic view of Earth as a whole, showing the planet as a complete sphere floating in the darkness of space."
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
    # wavespeed_generate_high_noise_only(image, prompt, high_noise_lora)
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
    
    
    # trace 51
    # high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/9458b69dd23041af9efa563ac9f51558/9a4d846cfee2442ca7e72a687269edfb.safetensors"
    # low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/9458b69dd23041af9efa563ac9f51558/9a4d846cfee2442ca7e72a687269edfb.safetensors"
    # trace 52
    # high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/4b04fca2d81d4f9fadac481a674c0799/59a43fde2c2944c59338ea28911da6df.safetensors"
    # low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/4b04fca2d81d4f9fadac481a674c0799/59a43fde2c2944c59338ea28911da6df.safetensors"
    # shangtian666
    # high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/ac31ff3427e243d4ab0dc0e83d000a04/8c64ab70297a4db88276064694f16c6d.safetensors"
    # low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/ac31ff3427e243d4ab0dc0e83d000a04/8c64ab70297a4db88276064694f16c6d.safetensors"
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/81836526-458b-44a9-b065-1ef36f326a65.png"
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/a8de839d-2f3a-4204-a455-6b7b42572899.png"
    # prompt = "An epic, continuous camera zoom out begins, pulling back to reveal the nearby geographical features come into view, followed by the surrounding landscape stretching to the horizon. The zoom continues to accelerate, ascending through the clouds and into space, culminating in a breathtaking view of planet Earth floating silently in the cosmos."
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
     # trace 30
    high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/641638d1d8614d7f9a6b36aea2d556d1/90fa4e309d984937ad90307ed7cd2d0c.safetensors"
    low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/641638d1d8614d7f9a6b36aea2d556d1/90fa4e309d984937ad90307ed7cd2d0c.safetensors"
    prompt = "Camera zooms out from a person standing on ground. Person becomes very small then invisible. View expands to show buildings, then city, then country, then continents. Final shot shows full Earth sphere in black space. Fast continuous zoom out movement from close up person to whole planet Earth."
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/81836526-458b-44a9-b065-1ef36f326a65.png"
    image = "https://liblibai-tmp-image.liblib.cloud/sd-images/a8de839d-2f3a-4204-a455-6b7b42572899.png"
    wavespeed_generate_high_noise_only(image, prompt, high_noise_lora)
    # trace80

     
    
    # high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/d9e2d06edf97432ea63a218e36737906/61c0a58cd2744a149fb987f7ce5542b7.safetensors"
    # low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/d9e2d06edf97432ea63a218e36737906/61c0a58cd2744a149fb987f7ce5542b7.safetensors"
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/81836526-458b-44a9-b065-1ef36f326a65.png"
    # # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/850eb8ae-2e82-4b1b-bf3f-236a46e96b20.png"
    # prompt = "The building explodes, and immediately the main character disintegrates into particles and disperses."
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
    
    
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/b6f212ff-eba7-4d20-8d54-7682ac608768.png"
    # prompt = "The building explodes, and immediately the main character disintegrates into particles and disperses."
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
    
    
    
    # high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/dafffcc2a3c74576ad11d8e8fdf9d2d7/79ab5c253e9b4b17a56fd26f8c51ef5f.safetensors"
    # low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/dafffcc2a3c74576ad11d8e8fdf9d2d7/79ab5c253e9b4b17a56fd26f8c51ef5f.safetensors"
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/60d78f80-05e8-418b-a2ed-a3470edb3e57.png"
    # # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/850eb8ae-2e82-4b1b-bf3f-236a46e96b20.png"
    # prompt = "The main character's hair, facial features, and body form transform into a monkey - Turning Monkey. The scene then switches to a forest, where the transformed monkey walks on ground."
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/850eb8ae-2e82-4b1b-bf3f-236a46e96b20.png"
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
    
    
    # high_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/67883556212f4a0b8c94265533067f1a/35e672bb184c405ca9065b4805fec913.safetensors"
    # low_noise_lora = "https://liblibai-tmp-image.liblib.cloud/models/chaowei/67883556212f4a0b8c94265533067f1a/35e672bb184c405ca9065b4805fec913.safetensors"
    # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/60d78f80-05e8-418b-a2ed-a3470edb3e57.png"
    # # # image = "https://liblibai-tmp-image.liblib.cloud/sd-images/850eb8ae-2e82-4b1b-bf3f-236a46e96b20.png"
    # prompt = "一个人在环境中面带微笑，举起手中的食物送向嘴边，镜头逐渐拉近聚焦在食物被送入口中的瞬间，展现咬下和咀嚼的动作细节，最终定格在面部或唇部特写"
    # wavespeed_generate(image, prompt, high_noise_lora, low_noise_lora)
    
