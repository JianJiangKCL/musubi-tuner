#### t2v
CUDA_VISIBLE_DEVICES=0 python /mnt/cfs/jj/musubi-tuner/wan_cache_latents.py --dataset_config /mnt/cfs/jj/musubi-tuner/my_config/baby_bee_meta_json.toml --vae /mnt/cfs/jj/ckpt/Wan2.2-T2V-A14B/Wan2.1_VAE.pth --vae_dtype bfloat16 --batch_size 180 --num_workers 8

CUDA_VISIBLE_DEVICES=1 python /mnt/cfs/jj/musubi-tuner/wan_cache_text_encoder_outputs.py --dataset_config /mnt/cfs/jj/musubi-tuner/my_config/baby_bee_meta_json.toml  --t5 /mnt/cfs/jj/ckpt/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth  --batch_size 128 
# --tokenizer /mnt/cfs/jj/ckpt/Wan2.2-T2V-A14B/google/umt5-xxl
### i2v
CUDA_VISIBLE_DEVICES=3 python /mnt/cfs/jj/musubi-tuner/wan_cache_latents.py --dataset_config /mnt/cfs/jj/musubi-tuner/my_config/baby_bee_i2v_meta_json.toml --vae /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/Wan2.1_VAE.pth --vae_dtype bfloat16 --batch_size 180 --num_workers 8 --i2v

CUDA_VISIBLE_DEVICES=4 python /mnt/cfs/jj/musubi-tuner/wan_cache_text_encoder_outputs.py --dataset_config /mnt/cfs/jj/musubi-tuner/my_config/baby_bee_i2v_meta_json.toml  --t5 /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth  --batch_size 128 


CUDA_VISIBLE_DEVICES=5 python /mnt/cfs/jj/musubi-tuner/wan_cache_latents.py --dataset_config /mnt/cfs/jj/musubi-tuner/my_config/peel_it_i2v_meta_json.toml --vae /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/Wan2.1_VAE.pth --vae_dtype bfloat16 --batch_size 180 --num_workers 8 --i2v

CUDA_VISIBLE_DEVICES=4 python /mnt/cfs/jj/musubi-tuner/wan_cache_text_encoder_outputs.py --dataset_config /mnt/cfs/jj/musubi-tuner/my_config/peel_it_i2v_meta_json.toml --t5 /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth  --batch_size 128 
