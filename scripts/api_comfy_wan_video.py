import hmac
import time
import requests
from datetime import datetime
import hashlib
import uuid
import base64


class Comfy:
    def __init__(self, ak='-IS1RD71KgHhbRrEkQKUSA', sk='NwNfwchfcHzntMpm2CNzgjTwGXNBHJJb', interval=5):
        """
        :param ak
        :param sk
        :param interval 轮询间隔
        """
        self.ak = ak
        self.sk = sk
        self.time_stamp = int(datetime.now().timestamp() * 1000)  # 毫秒级时间戳
        self.signature_nonce = uuid.uuid1()  # 随机字符串
        self.signature_img = self._hash_sk(self.sk, self.time_stamp, self.signature_nonce)
        self.signature_status = self._hash_sk_status(self.sk, self.time_stamp, self.signature_nonce)
        self.interval = interval
        self.headers = {'Content-Type': 'application/json'}
        self.comfy_url = self.get_image_url(self.ak, self.signature_img, self.time_stamp, self.signature_nonce)
        self.generate_url = self.get_generate_url(self.ak, self.signature_status, self.time_stamp,
                                                  self.signature_nonce)

    def hmac_sha1(self, key, code):
        hmac_code = hmac.new(key.encode(), code.encode(), hashlib.sha1)
        return hmac_code.digest()

    def _hash_sk(self, key, s_time, ro):
        """加密sk"""
        data = "/api/generate/comfyui/app" + "&" + str(s_time) + "&" + str(ro)
        s = base64.urlsafe_b64encode(self.hmac_sha1(key, data)).rstrip(b'=').decode()
        return s

    def _hash_sk_status(self, key, s_time, ro):
        """加密sk"""
        data = "/api/generate/comfy/status" + "&" + str(s_time) + "&" + str(ro)
        s = base64.urlsafe_b64encode(self.hmac_sha1(key, data)).rstrip(b'=').decode()
        return s

    def get_image_url(self, ak, signature, time_stamp, signature_nonce):

        url = f"https://openapi.liblibai.cloud/api/generate/comfyui/app?AccessKey={ak}&Signature={signature}&Timestamp={time_stamp}&SignatureNonce={signature_nonce}"
        return url

    def get_generate_url(self, ak, signature, time_stamp, signature_nonce):

        url = f"https://openapi.liblibai.cloud/api/generate/comfy/status?AccessKey={ak}&Signature={signature}&Timestamp={time_stamp}&SignatureNonce={signature_nonce}"
        return url

    def comfy_wan_video_lora(self, image, trigger_word, prompt, high_noise_lora, low_noise_lora):
        base_json = {
            "templateUuid": "4df2efa0f18d46dc9758803e478eb51c",
            "generateParams": {
                "67": {
                    "class_type": "LoadImage",
                    "inputs": {
                        "image": image
                    }
                },
                "115": {
                    "class_type": "SeargePromptCombiner",
                    "inputs": {
                        "prompt1": trigger_word,
                        "prompt2": prompt
                    }
                },
                "117": {
                    "class_type": "WanVideoLoraSelect",
                    "inputs": {
                        "lora": high_noise_lora,
                        "strength": 1
                    }
                },
                "118": {
                    "class_type": "WanVideoLoraSelect",
                    "inputs": {
                        "lora": low_noise_lora,
                        "strength": 1
                    }
                },
                "124": {
                    "class_type": "WanVideoLoraSelect",
                    "inputs": {
                        "lora": "46598f903eee46419fac22cdf7ddcddb",
                        "strength": 0
                    }
                },
                "125": {
                    "class_type": "WanVideoLoraSelect",
                    "inputs": {
                        "lora": "46598f903eee46419fac22cdf7ddcddb",
                        "strength": 0
                    }
                },
                "workflowUuid": "6715b74a1e5140c9963c5221a91f5db0"
            }
        }        
        return self.run(base_json, self.comfy_url)

    def comfy_wan_startendframe(self):
        base_json = {
            "templateUuid": "4df2efa0f18d46dc9758803e478eb51c",
            "generateParams": {
                "3": {
                    "class_type": "LoadImage",
                    "inputs": {
                        "image": "https://liblibai-tmp-image.liblib.cloud/img/081e9f07d9bd4c2ba090efde163518f9/097297f639b8a2850be8187c2a8d9465dc1afabfb813b76f6c188effd42a34c4.png"
                    }
                },
                "4": {
                    "class_type": "LoadImage",
                    "inputs": {
                        "image": "https://liblibai-tmp-image.liblib.cloud/img/081e9f07d9bd4c2ba090efde163518f9/da0e158ebeb23a52d832b16f45c0ac43d7c60f07e36bdc0a438602c4a251cfab.png"
                    }
                },
                "51": {
                    "class_type": "WanVideoLoraSelect",
                    "inputs": {
                        "lora": "1736639c952a4652a7aa5e1a1b83a156",
                        "strength": 1
                    }
                },
                "52": {
                    "class_type": "WanVideoLoraSelect",
                    "inputs": {
                        "lora": "1736639c952a4652a7aa5e1a1b83a156",
                        "strength": 0
                    }
                },
                "53": {
                    "class_type": "WanVideoLoraSelect",
                    "inputs": {
                        "lora": "1736639c952a4652a7aa5e1a1b83a156",
                        "strength": 1
                    }
                },
                "54": {
                    "class_type": "WanVideoLoraSelect",
                    "inputs": {
                        "lora": "1736639c952a4652a7aa5e1a1b83a156",
                        "strength": 0
                    }
                },
                "55": {
                    "class_type": "SeargePromptCombiner",
                    "inputs": {
                        "prompt1": "xiaosan66",
                        "prompt2": "桌子周围的人在大笑着打扑克牌，镜头对面的女人对镜头招手。镜头前的人物手上的海报化为非常细腻的黑色粒子烟雾消散掉"
                    }
                },
                "workflowUuid": "5fc1ce80e4514173a1020ad4227ec470"
            }
        }
        self.run(base_json, self.comfy_url)

    def run(self, data, url, timeout=1000, max_retries=3):
        """
        发送任务到生图接口，直到返回video为止，失败抛出异常信息
        返回格式: {"video_url": str, "generate_uuid": str} 或 None
        """
        import time
        
        start_time = time.time()  # 记录开始时间
        # 这里提交任务，校验是否提交成功，并且获取任务ID
        # print(url)
        response = requests.post(url=url, headers=self.headers, json=data)
        response.raise_for_status()
        progress = response.json()
        # print(progress)

        if progress['code'] == 0:
            generate_uuid = progress["data"]['generateUuid']
            print(f"任务提交成功，任务ID: {generate_uuid}")
            
            # 如果获取到任务ID，执行等待生图
            consecutive_failures = 0
            while True:
                current_time = time.time()
                if (current_time - start_time) > timeout:
                    print(f"{timeout}s任务超时，已退出轮询。")
                    return {"video_url": None, "generate_uuid": generate_uuid, "error": "timeout"}

                # 添加重试机制
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        data_status = {"generateUuid": generate_uuid}
                        response = requests.post(url=self.generate_url, headers=self.headers, json=data_status, timeout=30)
                        response.raise_for_status()
                        progress = response.json()
                        consecutive_failures = 0  # 重置连续失败计数
                        break
                    except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                        retry_count += 1
                        consecutive_failures += 1
                        print(f"⚠️ 网络请求失败 (重试 {retry_count}/{max_retries}): {e}")
                        if retry_count < max_retries:
                            time.sleep(min(2 ** retry_count, 10))  # 指数退避
                        else:
                            print(f"❌ 达到最大重试次数，跳过本次轮询")
                            if consecutive_failures >= 5:
                                print(f"❌ 连续失败次数过多，任务可能已异常")
                                return None
                            continue

                # 检查任务状态
                if progress.get('code') == 0 and progress.get('data'):
                    data_obj = progress['data']
                    generate_status = data_obj.get('generateStatus', 0)                                   
                    
                    # generateStatus=5是成功
                    if generate_status == 5:
                        # 修复videoUrl取值逻辑
                        videos = data_obj.get('videos', [])
                        if videos and len(videos) > 0:
                            video_url = None
                            # 尝试多种取值方式
                            if isinstance(videos[0], dict):
                                video_url = videos[0].get('videoUrl')
                            elif isinstance(videos[0], str):
                                video_url = videos[0]
                            
                            if video_url:
                                print(f"✅ 任务成功完成，获取到视频: {video_url}")
                                return {"video_url": video_url, "generate_uuid": generate_uuid}
                            else:
                                print(f"❌ 任务完成但未获取到视频URL: {videos}")
                                return {"video_url": None, "generate_uuid": generate_uuid, "error": "no_video_url"}
                        else:
                            print(f"❌ 任务完成但videos为空: {data_obj}")
                            return {"video_url": None, "generate_uuid": generate_uuid, "error": "empty_videos"}
                    
                    # generateStatus=6或7是失败
                    elif generate_status in [6, 7]:
                        error_msg = data_obj.get('msg', '未知错误')
                        print(f"❌ 任务失败 (状态码: {generate_status}): {error_msg}")
                        return {"video_url": None, "generate_uuid": generate_uuid, "error": f"generation_failed_{generate_status}", "error_msg": error_msg}
                    
                    # 其他状态继续等待
                    else:
                        # print(f"⏳ 任务进行中 (状态: {generate_status})，等待 {self.interval} 秒...")
                        time.sleep(self.interval)
                else:
                    print(f"⚠️ 获取任务状态异常: {progress}")
                    time.sleep(self.interval)
        else:
            error_msg = f'任务提交失败,原因：{progress.get("msg", "未知错误")}'
            print(f"❌ {error_msg}")
            return {"video_url": None, "generate_uuid": None, "error": "submit_failed", "error_msg": error_msg}


def run_wan_video(image, trigger_word, prompt, high_noise_lora, low_noise_lora):
    test = Comfy()
    start_time = time.time()
    test.comfy_wan_video_lora(image, trigger_word, prompt, high_noise_lora, low_noise_lora)
    end_time = time.time()
    print("任务耗时：",end_time-start_time, "秒")

if __name__ == '__main__':
    run_wan_video(image="https://liblibai-tmp-image.liblib.cloud/sd-images/1177d089-0417-4a9f-b0c0-2fad539bc976.png", trigger_word="Attack of Clones", prompt="三个穿着黑色连帽衫和黑色针织帽的人并排站在沙漠中，背景是连绵的沙丘。随后，更多穿着相同黑色连帽衫和帽子的人从两侧涌入，形成Attack of Clones般的包围态势，将中间三人团团围住。接着，中间的人周围突然爆发出大量沙尘，沙尘如瀑布般向下倾泻，同时周围的人被震飞四散，纷纷倒地。最终，沙尘逐渐散去，原本中间的人独自站立在沙漠中，周围躺着多个同样穿着黑色连帽衫的人，地面出现明显的裂纹。", high_noise_lora="d9c0a34176444dc5904f241a2e50afd1", low_noise_lora="d9c0a34176444dc5904f241a2e50afd1")