CUDA_VISIBLE_DEVICES=1 accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 /mnt/cfs/jj/musubi-tuner/src/musubi_tuner/wan_train_network.py \
    --dit /mnt/cfs/jj/ckpt/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors \
    --dit_high_noise /mnt/cfs/jj/ckpt/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors \
    --dataset_config /mnt/cfs/jj/musubi-tuner/my_config/baby_bee_meta_json.toml --flash_attn --mixed_precision bf16 \
    --fp8_base \
    --fp8_scaled \
    --optimizer_type adamw8bit \
    --task t2v-14B \
    --timestep_boundary  0.875 \
    --offload_inactive_dit \
    --vae_cache_cpu \
    --t5 /mnt/cfs/jj/ckpt/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth \
    --learning_rate 2e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_wan --network_dim 32 \
    --timestep_sampling shift --discrete_flow_shift 7.0 \
    --max_train_epochs 100 --save_every_n_epochs 10 --seed 42 \
    --output_dir /mnt/cfs/jj/musubi-tuner/baby_bee_t2v_outputs --output_name name-of-lorac \
    --min_timestep 200 --max_timestep 800 --preserve_distribution_shape \
    --sample_prompts /mnt/cfs/jj/musubi-tuner/my_config/baby_bee_prompt.txt \
    --sample_every_n_epochs 10 \
    --t5 /mnt/cfs/jj/ckpt/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth \
    --vae /mnt/cfs/jj/ckpt/Wan2.2-T2V-A14B/Wan2.1_VAE.pth 

    #     --fp8_scaled \
    # --fp8_t5 \
        #  \
