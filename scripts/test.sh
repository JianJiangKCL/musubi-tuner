
CUDA_VISIBLE_DEVICES=6 python /mnt/cfs/jj/musubi-tuner/src/musubi_tuner/wan_generate_video.py \
    --dit  /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors \
    --dit_high_noise /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors \
    --lazy_loading \
    --t5 /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth \
    --task i2v-A14B \
    --prompt "A hand peels a realistic 3D sticker of a car off a flat surface，revealing the flat background underneath and emphasizing the illusion of depth." \
    --video_size 320 480 \
    --video_length 81 \
    --vae /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/Wan2.1_VAE.pth \
    --vae_cache_cpu \
    --save_path /mnt/cfs/jj/musubi-tuner/test_i2v_outputs/ \
    --attn_mode flash2 \
    --lora_weight /mnt/cfs/jj/musubi-tuner/peel_it_outputs_high_noise_frame81_uniform/peel_it-of-lorac-000150.safetensors \
    --image_path /mnt/cfs/jj/musubi-tuner/datasets/peel_it/first_frames/1.jpg

# CUDA_VISIBLE_DEVICES=1  python /mnt/cfs/jj/Wan2.2/generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir /mnt/cfs/jj/ckpt/Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "A cute baby bee flying in a flower garden"

CUDA_VISIBLE_DEVICES=7 python /mnt/cfs/jj/musubi-tuner/src/musubi_tuner/wan_generate_video.py \
    --dit  /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors \
    --dit_high_noise /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors \
    --lazy_loading \
    --t5 /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth \
    --task i2v-A14B \
    --prompt "A hand peels a realistic 3D sticker of a woman off a flat surface, revealing the blank background underneath and emphasizing the illusion of depth." \
    --video_size 320 480 \
    --video_length 81 \
    --vae /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/Wan2.1_VAE.pth \
    --vae_cache_cpu \
    --save_path /mnt/cfs/jj/musubi-tuner/test_i2v_outputs/ \
    --attn_mode flash2 \
    --lora_weight /mnt/cfs/jj/musubi-tuner/peel_it_outputs_high_noise_full_trace8/peel_it-of-lorac.safetensors \
    --image_path /mnt/cfs/jj/musubi-tuner/datasets/peel_it/first_frames/2.jpg
    /mnt/cfs/jj/musubi-tuner/test_i2v_outputs/girl.jpg


    CUDA_VISIBLE_DEVICES=5 python /mnt/cfs/jj/musubi-tuner/src/musubi_tuner/wan_generate_video.py \
    --dit  /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors \
    --dit_high_noise /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors \
    --lazy_loading \
    --t5 /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth \
    --task i2v-A14B \
    --prompt "A hand peels a realistic 3D sticker of a car off a flat surface，revealing the flat background underneath and emphasizing the illusion of depth." \
    --video_size 320 480 \
    --video_length 81 \
    --vae /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/Wan2.1_VAE.pth \
    --vae_cache_cpu \
    --save_path /mnt/cfs/jj/musubi-tuner/test_i2v_outputs/ \
    --attn_mode flash2 \
    --lora_weight /mnt/cfs/jj/musubi-tuner/peel_it_outputs_high_noise_full_trace8/peel_it-of-lorac.safetensors \
    --image_path /mnt/cfs/jj/musubi-tuner/test_i2v_outputs/test_car.jpg