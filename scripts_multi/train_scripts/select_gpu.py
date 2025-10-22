#!/usr/bin/env python3
import sys

try:
    import GPUtil
    
    # Get available GPUs
    max_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    memory_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    
    # Get GPUs with memory usage below threshold
    gpus = GPUtil.getGPUs()
    available_gpus = []
    
    for gpu in gpus:
        if gpu.memoryUtil < memory_threshold:
            available_gpus.append((gpu.id, gpu.memoryUtil, gpu.load))
    
    # Sort by memory usage (lowest first)
    available_gpus.sort(key=lambda x: (x[1], x[2]))
    
    # Print available GPU IDs (space-separated)
    if available_gpus:
        gpu_ids = [str(g[0]) for g in available_gpus[:max_gpus]]
        print(' '.join(gpu_ids))
    else:
        # If no GPUs available, use all GPUs
        all_gpu_ids = [str(gpu.id) for gpu in gpus[:max_gpus]]
        print(' '.join(all_gpu_ids))
        
except ImportError:
    # If GPUtil not available, fall back to detecting all GPUs
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True)
        gpu_ids = result.stdout.strip().split('\n')
        print(' '.join(gpu_ids[:int(sys.argv[1]) if len(sys.argv) > 1 else 8]))
    except:
        # Final fallback
        print('0 1 2 3 4 5 6 7')
