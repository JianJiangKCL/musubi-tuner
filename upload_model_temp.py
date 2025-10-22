#!/usr/bin/env python3
import sys
import os
sys.path.append('scripts')

from upload_g2 import upload_model_file

def main():
    model_file_path = "/mnt/cfs/jj/musubi-tuner/lora_outputs/attack/high_noise_trace30/attack-of-lorac.safetensors"

    if not os.path.exists(model_file_path):
        print(f"错误：模型文件不存在: {model_file_path}")
        return

    print(f"开始上传模型文件: {model_file_path}")
    oss_url, model_dir_uuid = upload_model_file(model_file_path)

    if oss_url:
        print("\n上传成功！")
        print(f"OSS URL: {oss_url}")
        print(f"Model Dir UUID: {model_dir_uuid}")

        # 写入到文件供后续使用
        with open('model_upload_info.txt', 'w', encoding='utf-8') as f:
            f.write(f"FILE_PATH={model_file_path}\n")
            f.write(f"OSS_URL={oss_url}\n")
            f.write(f"MODEL_DIR_UUID={model_dir_uuid}\n")
            f.write(f"FILE_SIZE={os.path.getsize(model_file_path)}\n")
            f.write(f"FILE_NAME={os.path.basename(model_file_path)}\n")

        print(f"\n信息已保存到: model_upload_info.txt")
    else:
        print("上传失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
