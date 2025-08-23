import torch
import numpy as np
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video, load_image

dtype = torch.bfloat16
device = "cuda"
vae = AutoencoderKLWan.from_pretrained("/mnt/cfs/jj/ckpt/Wan2.2-T2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained("/mnt/cfs/jj/ckpt/Wan2.2-T2V-A14B-Diffusers", vae=vae, torch_dtype=dtype)
pipe.to(device)

# pipe.load_lora_weights(
#    "Kijai/WanVideo_comfy", 
#    weight_name="Wan22-Lightning/Wan2.2-Lightning_T2V-A14B-4steps-lora_HIGH_fp16.safetensors", 
#     adapter_name="lightning"
# )
# # kwargs = {}
# # kwargs["load_into_transformer_2"] = True
# # pipe.load_lora_weights(
# #    "Kijai/WanVideo_comfy", 
# #    weight_name="Wan22-Lightning/Wan2.2-Lightning_T2V-A14B-4steps-lora_LOW_fp16.safetensors", 
# #     adapter_name="lightning_2", **kwargs
# # )
# pipe.set_adapters(["lightning", "lightning_2"], adapter_weights=[1., 1.])

height = 480
width = 832

prompt = "A cute baby bee flying in a flower garden"
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=81,
    guidance_scale=1.0,
    guidance_scale_2=1.0,
    num_inference_steps=4,
    generator=torch.manual_seed(0),
).frames[0]
export_to_video(output, "t2v_out.mp4", fps=16)