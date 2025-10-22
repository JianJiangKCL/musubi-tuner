#!/usr/bin/env python3
"""
GPU Manager for Musubi Tuner Pipeline
Automatically manages TOML creation, caching, and training for multiple datasets.
"""

import os
import sys
import argparse
import subprocess
import threading
import time
import queue
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import GPUtil
from datetime import datetime
import random
import toml

class GPUManager:
    def __init__(self, max_gpu_memory_threshold=0.9):
        """Initialize GPU manager with memory threshold for availability check."""
        self.max_gpu_memory_threshold = max_gpu_memory_threshold
        self.gpu_lock = threading.Lock()
        
    def get_available_gpus(self) -> List[int]:
        """Get list of completely free GPU IDs (no processes running)."""
        try:
            # First try using nvidia-smi to get detailed process information
            result = subprocess.run([
                'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Get all GPU UUIDs and their running processes
                running_processes = result.stdout.strip()
                occupied_gpu_uuids = set()
                
                if running_processes:
                    for line in running_processes.split('\n'):
                        if line.strip():
                            gpu_uuid = line.split(',')[0].strip()
                            occupied_gpu_uuids.add(gpu_uuid)
                
                # Get all GPU information including UUIDs
                gpu_info_result = subprocess.run([
                    'nvidia-smi', '--query-gpu=index,uuid', 
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True)
                
                if gpu_info_result.returncode == 0:
                    available_gpus = []
                    for line in gpu_info_result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = line.split(',')
                            gpu_id = int(parts[0].strip())
                            gpu_uuid = parts[1].strip()
                            
                            # GPU is free if no processes are running on it
                            if gpu_uuid not in occupied_gpu_uuids:
                                available_gpus.append(gpu_id)
                    
                    return available_gpus
            
            # Fallback: use GPUtil but check for zero memory usage
            gpus = GPUtil.getGPUs()
            available_gpus = []
            
            for gpu in gpus:
                # Consider GPU free only if memory usage is very minimal (< 100MB)
                if gpu.memoryUsed < 100:  # Less than 100MB used
                    available_gpus.append(gpu.id)
                    
            return available_gpus
            
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            # Final fallback: try to detect total GPU count
            try:
                result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_count = len(result.stdout.strip().split('\n'))
                    print(f"Warning: Cannot determine GPU usage, assuming all {gpu_count} GPUs are available")
                    return list(range(gpu_count))
                else:
                    print("No GPUs detected")
                    return []
            except FileNotFoundError:
                print("nvidia-smi not found. No GPUs available.")
                return []


class MusubiJob:
    def __init__(self, dataset_name: str, jsonl_path: str, base_config_dir: str, 
                 cache_dir: str, output_dir: str):
        self.dataset_name = dataset_name
        self.jsonl_path = jsonl_path
        self.base_config_dir = Path(base_config_dir)
        self.cache_dir = Path(cache_dir) / dataset_name
        self.output_dir = Path(output_dir) / dataset_name
        
        # Create directories
        self.base_config_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Config file path
        self.toml_path = self.base_config_dir / f"{dataset_name}.toml"
        
        # Job states
        self.toml_created = False
        self.text_encoder_cached = False
        self.latents_cached = False
        self.training_completed = False
        
    def create_toml_config(self) -> bool:
        """Create TOML configuration file for this dataset."""
        config = {
            "general": {
                "resolution": [960, 960],
                "batch_size": 1,
                "enable_bucket": True,
                "bucket_no_upscale": False
            },
            "datasets": [{
                "video_jsonl_file": self.jsonl_path,
                "frame_extraction": "full",
                "max_frames": 230,
                "resolution": [298, 298],
                "cache_directory": str(self.cache_dir)
            }]
        }
        
        try:
            with open(self.toml_path, 'w') as f:
                toml.dump(config, f)
            print(f"✓ Created TOML config: {self.toml_path}")
            self.toml_created = True
            return True
        except Exception as e:
            print(f"✗ Failed to create TOML config: {e}")
            return False
    
    def __str__(self):
        return f"MusubiJob({self.dataset_name})"


class MusubiPipelineManager:
    def __init__(self, dataset_dir: str, config_dir: str, cache_root: str, 
                 output_root: str, max_workers: int = None):
        self.dataset_dir = Path(dataset_dir)
        self.config_dir = Path(config_dir)
        self.cache_root = Path(cache_root)
        self.output_root = Path(output_root)
        self.gpu_manager = GPUManager()
        
        # Job queues for different stages
        self.text_encoder_queue = queue.Queue()
        self.latents_queue = queue.Queue()
        self.training_queue = queue.Queue()
        
        # Active jobs tracking
        self.active_jobs = {}  # gpu_id -> (job, process, stage)
        self.completed_jobs = []
        self.failed_jobs = []
        
        # Port management for training
        self.base_port = 29500
        self.used_ports = set()
        self.port_lock = threading.Lock()
        
        # Paths to models
        self.t5_model_path = "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth"
        self.vae_model_path = "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
        self.dit_model_path = "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors"
        self.dit_high_noise_path = "/mnt/cfs/jj/ckpt/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors"
        
        self.max_workers = max_workers
        
    def discover_datasets(self) -> List[Tuple[str, str]]:
        """Discover all jsonl files in the dataset directory."""
        datasets = []
        for jsonl_file in self.dataset_dir.glob("*.jsonl"):
            dataset_name = jsonl_file.stem
            datasets.append((dataset_name, str(jsonl_file)))
        return sorted(datasets)
    
    def create_jobs(self) -> List[MusubiJob]:
        """Create MusubiJob objects for all discovered datasets."""
        datasets = self.discover_datasets()
        jobs = []
        
        print(f"\nDiscovered {len(datasets)} datasets:")
        for dataset_name, jsonl_path in datasets:
            print(f"  - {dataset_name}")
            
            job = MusubiJob(
                dataset_name=dataset_name,
                jsonl_path=jsonl_path,
                base_config_dir=self.config_dir / "to_improve",
                cache_dir=self.cache_root,
                output_dir=self.output_root
            )
            
            # Create TOML config
            if job.create_toml_config():
                jobs.append(job)
                # Add to text encoder queue (first stage)
                self.text_encoder_queue.put(job)
            else:
                self.failed_jobs.append((job, "Failed to create TOML config"))
                
        return jobs
    
    def get_free_port(self) -> int:
        """Get a free port for training, ensuring no conflicts."""
        with self.port_lock:
            port = self.base_port
            while port in self.used_ports:
                port += 1
            self.used_ports.add(port)
            return port
    
    def release_port(self, port: int):
        """Release a port back to the pool."""
        with self.port_lock:
            self.used_ports.discard(port)
    
    def run_text_encoder_caching(self, job: MusubiJob, gpu_id: int) -> bool:
        """Run text encoder caching for a job."""
        print(f"🔤 Starting text encoder caching for {job} on GPU {gpu_id}")
        
        cmd = [
            "python", "/mnt/cfs/jj/musubi-tuner/wan_cache_text_encoder_outputs.py",
            "--dataset_config", str(job.toml_path),
            "--t5", self.t5_model_path,
            "--batch_size", "128"
        ]
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Set Hugging Face environment to avoid timeouts
        env['HF_ENDPOINT'] = 'https://huggingface.co'
        env['TRANSFORMERS_OFFLINE'] = '1'
        env['HF_DATASETS_OFFLINE'] = '1'
        
        try:
            # Activate conda environment
            conda_cmd = f"conda run -n musu {' '.join(cmd)}"
            
            process = subprocess.Popen(
                conda_cmd,
                shell=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Track active job
            with self.gpu_manager.gpu_lock:
                self.active_jobs[gpu_id] = (job, process, 'text_encoder')
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"✓ Completed text encoder caching for {job}")
                job.text_encoder_cached = True
                # Move to latents queue
                self.latents_queue.put(job)
                return True
            else:
                print(f"✗ Failed text encoder caching for {job}")
                print(f"Error: {stderr}")
                self.failed_jobs.append((job, f"Text encoder caching failed: {stderr}"))
                return False
                
        except Exception as e:
            print(f"✗ Exception in text encoder caching for {job}: {e}")
            self.failed_jobs.append((job, f"Text encoder exception: {str(e)}"))
            return False
        finally:
            with self.gpu_manager.gpu_lock:
                if gpu_id in self.active_jobs:
                    del self.active_jobs[gpu_id]
    
    def run_latents_caching(self, job: MusubiJob, gpu_id: int) -> bool:
        """Run latents caching for a job."""
        print(f"🎨 Starting latents caching for {job} on GPU {gpu_id}")
        
        cmd = [
            "python", "/mnt/cfs/jj/musubi-tuner/wan_cache_latents.py",
            "--dataset_config", str(job.toml_path),
            "--vae", self.vae_model_path,
            "--vae_dtype", "bfloat16",
            "--batch_size", "180",
            "--num_workers", "8",
            "--i2v"
        ]
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Set Hugging Face environment to avoid timeouts
        env['HF_ENDPOINT'] = 'https://huggingface.co'
        env['TRANSFORMERS_OFFLINE'] = '1'
        env['HF_DATASETS_OFFLINE'] = '1'
        
        try:
            # Activate conda environment
            conda_cmd = f"conda run -n musu {' '.join(cmd)}"
            
            process = subprocess.Popen(
                conda_cmd,
                shell=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Track active job
            with self.gpu_manager.gpu_lock:
                self.active_jobs[gpu_id] = (job, process, 'latents')
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"✓ Completed latents caching for {job}")
                job.latents_cached = True
                # Move to training queue
                self.training_queue.put(job)
                return True
            else:
                print(f"✗ Failed latents caching for {job}")
                print(f"Error: {stderr}")
                self.failed_jobs.append((job, f"Latents caching failed: {stderr}"))
                return False
                
        except Exception as e:
            print(f"✗ Exception in latents caching for {job}: {e}")
            self.failed_jobs.append((job, f"Latents exception: {str(e)}"))
            return False
        finally:
            with self.gpu_manager.gpu_lock:
                if gpu_id in self.active_jobs:
                    del self.active_jobs[gpu_id]
    
    def run_training(self, job: MusubiJob, gpu_id: int) -> bool:
        """Run training for a job."""
        port = self.get_free_port()
        print(f"🚀 Starting training for {job} on GPU {gpu_id} with port {port}")
        
        cmd = [
            "accelerate", "launch",
            "--config_file", "/mnt/cfs/jj/musubi-tuner/single_gpu_config.yaml",
            "--num_cpu_threads_per_process", "1",
            "--mixed_precision", "bf16",
            "--main_process_port", str(port),
            "/mnt/cfs/jj/musubi-tuner/src/musubi_tuner/wan_train_network.py",
            "--dit", self.dit_model_path,
            "--dit_high_noise", self.dit_high_noise_path,
            "--dataset_config", str(job.toml_path),
            "--flash_attn",
            "--mixed_precision", "bf16",
            "--fp8_base",
            "--fp8_scaled",
            "--optimizer_type", "adamw",
            "--task", "i2v-A14B",
            "--offload_inactive_dit",
            "--vae_cache_cpu",
            "--t5", self.t5_model_path,
            "--max_data_loader_n_workers", "8",
            "--persistent_data_loader_workers",
            "--network_module", "networks.lora_wan",
            "--network_dim", "32",
            "--network_alpha", "32",
            "--timestep_sampling", "shift",
            "--discrete_flow_shift", "1.0",
            "--max_train_epochs", "100",
            "--save_every_n_epochs", "1",
            "--seed", "42",
            "--output_dir", str(job.output_dir),
            "--output_name", f"{job.dataset_name}-lora",
            "--preserve_distribution_shape",
            "--vae", self.vae_model_path,
            "--optimizer_args", "weight_decay=0.1",
            "--max_grad_norm", "0",
            "--learning_rate", "3e-4",
            "--lr_scheduler", "polynomial",
            "--lr_scheduler_power", "8",
            "--lr_scheduler_min_lr_ratio", "5e-5",
            "--min_timestep", "875",
            "--max_timestep", "1000",
            "--gradient_checkpointing"
        ]
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Set Hugging Face environment to avoid timeouts
        env['HF_ENDPOINT'] = 'https://huggingface.co'
        env['TRANSFORMERS_OFFLINE'] = '1'
        env['HF_DATASETS_OFFLINE'] = '1'
        
        try:
            # Activate conda environment
            conda_cmd = f"conda run -n musu {' '.join(cmd)}"
            
            process = subprocess.Popen(
                conda_cmd,
                shell=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Track active job
            with self.gpu_manager.gpu_lock:
                self.active_jobs[gpu_id] = (job, process, 'training')
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"✓ Completed training for {job}")
                job.training_completed = True
                self.completed_jobs.append(job)
                return True
            else:
                print(f"✗ Failed training for {job}")
                print(f"Error: {stderr}")
                self.failed_jobs.append((job, f"Training failed: {stderr}"))
                return False
                
        except Exception as e:
            print(f"✗ Exception in training for {job}: {e}")
            self.failed_jobs.append((job, f"Training exception: {str(e)}"))
            return False
        finally:
            self.release_port(port)
            with self.gpu_manager.gpu_lock:
                if gpu_id in self.active_jobs:
                    del self.active_jobs[gpu_id]
    
    def worker_thread(self, stage: str):
        """Worker thread that processes jobs from specific queue."""
        if stage == 'text_encoder':
            job_queue = self.text_encoder_queue
            run_func = self.run_text_encoder_caching
        elif stage == 'latents':
            job_queue = self.latents_queue
            run_func = self.run_latents_caching
        elif stage == 'training':
            job_queue = self.training_queue
            run_func = self.run_training
        else:
            return
            
        while True:
            try:
                # Get available GPUs
                available_gpus = self.gpu_manager.get_available_gpus()
                
                # Filter out GPUs that are currently running jobs
                free_gpus = [gpu for gpu in available_gpus if gpu not in self.active_jobs]
                
                if not free_gpus:
                    time.sleep(5)
                    continue
                    
                # Get next job from queue
                try:
                    job = job_queue.get(timeout=1)
                except queue.Empty:
                    continue
                    
                # Assign job to first available GPU
                gpu_id = free_gpus[0]
                
                print(f"📌 Assigning {job.dataset_name} ({stage}) to GPU {gpu_id}")
                
                # Mark GPU as occupied immediately
                with self.gpu_manager.gpu_lock:
                    if gpu_id in self.active_jobs:
                        # GPU already occupied, put job back in queue
                        job_queue.put(job)
                        continue
                    
                    # Reserve the GPU
                    self.active_jobs[gpu_id] = (job, None, stage)
                
                # Run job in separate thread
                job_thread = threading.Thread(
                    target=run_func,
                    args=(job, gpu_id)
                )
                job_thread.daemon = True
                job_thread.start()
                
            except KeyboardInterrupt:
                print(f"Worker thread ({stage}) interrupted")
                break
            except Exception as e:
                print(f"Worker thread ({stage}) error: {e}")
                time.sleep(5)
    
    def run(self):
        """Main execution method."""
        print("=== Musubi Tuner GPU Pipeline Manager ===")
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Config directory: {self.config_dir}")
        print(f"Cache root: {self.cache_root}")
        print(f"Output root: {self.output_root}")
        
        # Check available GPUs
        available_gpus = self.gpu_manager.get_available_gpus()
        if not available_gpus:
            print("No free GPUs available!")
            return
            
        print(f"Available GPUs: {available_gpus}")
        
        # Create jobs
        jobs = self.create_jobs()
        if not jobs:
            print("No jobs to process!")
            return
            
        print(f"\nCreated {len(jobs)} jobs")
        
        # Start worker threads for each stage
        stages = ['text_encoder', 'latents', 'training']
        workers = []
        
        for stage in stages:
            worker = threading.Thread(
                target=self.worker_thread,
                args=(stage,)
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)
            print(f"Started worker thread for {stage}")
            
        # Monitor progress
        try:
            while True:
                # Check if all jobs are completed
                if len(self.completed_jobs) + len(self.failed_jobs) >= len(jobs):
                    print("\nAll jobs processed!")
                    break
                    
                time.sleep(10)
                
                # Print status
                text_q = self.text_encoder_queue.qsize()
                latents_q = self.latents_queue.qsize()
                training_q = self.training_queue.qsize()
                active_count = len(self.active_jobs)
                completed_count = len(self.completed_jobs)
                failed_count = len(self.failed_jobs)
                
                print(f"\n📊 Status: Active: {active_count}, Completed: {completed_count}, Failed: {failed_count}")
                print(f"   Queues - Text: {text_q}, Latents: {latents_q}, Training: {training_q}")
                
                # Show active jobs
                if self.active_jobs:
                    print("🔄 Active jobs:")
                    for gpu_id, (job, _, stage) in self.active_jobs.items():
                        print(f"  GPU {gpu_id}: {job.dataset_name} ({stage})")
                        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
        # Final summary
        print("\n=== Final Summary ===")
        print(f"Completed: {len(self.completed_jobs)}")
        if self.completed_jobs:
            print("Completed jobs:")
            for job in self.completed_jobs:
                print(f"  ✓ {job.dataset_name}")
                
        print(f"\nFailed: {len(self.failed_jobs)}")
        if self.failed_jobs:
            print("Failed jobs:")
            for job, error in self.failed_jobs:
                print(f"  ✗ {job.dataset_name}: {error}")


def main():
    parser = argparse.ArgumentParser(description="GPU Manager for Musubi Tuner Pipeline")
    parser.add_argument("--dataset_dir", 
                       default="/mnt/cfs/jj/musubi-tuner/datasets/to_improve",
                       help="Directory containing jsonl datasets")
    parser.add_argument("--config_dir",
                       default="/mnt/cfs/jj/musubi-tuner/my_config",
                       help="Directory for TOML configs")
    parser.add_argument("--cache_root",
                       default="/mnt/cfs/jj/musubi-tuner/cache",
                       help="Root directory for caches")
    parser.add_argument("--output_root",
                       default="/mnt/cfs/jj/musubi-tuner/lora_outputs",
                       help="Root directory for training outputs")
    parser.add_argument("--max_workers", type=int,
                       help="Maximum number of concurrent workers")
    
    args = parser.parse_args()
    
    # Create and run manager
    manager = MusubiPipelineManager(
        dataset_dir=args.dataset_dir,
        config_dir=args.config_dir,
        cache_root=args.cache_root,
        output_root=args.output_root,
        max_workers=args.max_workers
    )
    
    manager.run()
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED")
    print("="*50)


if __name__ == "__main__":
    main()
