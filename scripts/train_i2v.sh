#https://github.com/kohya-ss/musubi-tuner/blob/feat-wan-generate-lazy-loading/docs/wan.md
# high noise
CUDA_VISIBLE_DEVICES=4 accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 /mnt/cfs/jj/musubi-tuner/src/musubi_tuner/wan_train_network.py \
    --dit /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors \
    --dit_high_noise /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors \
    --dataset_config /mnt/cfs/jj/musubi-tuner/my_config/peel_it_i2v_meta_json.toml \
    --flash_attn \
    --mixed_precision bf16 \
    --fp8_base \
    --fp8_scaled \
    --optimizer_type adamw \
    --task i2v-A14B \
    --offload_inactive_dit \
    --vae_cache_cpu \
    --t5 /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth \
    --learning_rate 3e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 8 --persistent_data_loader_workers \
    --network_module networks.lora_wan --network_dim 16 --network_alpha 16 \
    --timestep_sampling shift --discrete_flow_shift 1.0  \
    --max_train_epochs 100 --save_every_n_epochs 1 --seed 42 \
    --output_dir /mnt/cfs/jj/musubi-tuner/peel_it_outputs_high_noise_frame81_uniform --output_name peel_it-of-lorac \
    --preserve_distribution_shape \
    --t5 /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth  \
    --vae /mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/Wan2.1_VAE.pth \
    --optimizer_args weight_decay=0.1 \
    --max_grad_norm 0 \
    --lr_scheduler polynomial \
    --lr_scheduler_power 8 \
    --lr_scheduler_min_lr_ratio="5e-5" \
    --min_timestep 875 \
    --max_timestep 1000 
    ## for low noise model training use --min_timestep 0 --max_timestep 875 
    # --min_timestep 0 \
    # --max_timestep 875 
   

    # --lr_scheduler cosine --lr_warmup_steps 0.05 \
    # --lr_scheduler_min_lr_ratio 0.1

    # --sample_prompts /mnt/cfs/jj/musubi-tuner/my_config/peel_it_prompt.txt \
    # --sample_every_n_epochs 10 \

        # --lazy_loading \
# timestamp during training can only be int


