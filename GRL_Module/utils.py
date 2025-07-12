import os
import psutil
import time
import torch
import numpy as np
from datetime import timedelta

def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model or Stable-Baselines3 model
    
    Returns:
        int: Total number of trainable parameters
    """
    # For Stable-Baselines3 PPO model, extract the PyTorch model
    if hasattr(model, 'policy') and hasattr(model.policy, 'parameters'):
        return sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    # For regular PyTorch models
    elif hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return 0

def get_memory_usage():
    """
    Get current RAM usage.
    
    Returns:
        dict: Contains RAM usage in MB and percentage
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024  # Convert to MB
    memory_percent = process.memory_percent()
    
    return {
        "memory_usage_mb": memory_usage_mb,
        "memory_percent": memory_percent
    }

def get_gpu_usage():
    """
    Get current GPU usage if available.
    
    Returns:
        list: List of dicts with GPU usage info, or None if no GPU is available
    """
    if not torch.cuda.is_available():
        return None
    
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        gpu_stats = {
            "device_id": i,
            "name": torch.cuda.get_device_name(i),
            "memory_allocated_mb": torch.cuda.memory_allocated(i) / 1024 / 1024,
            "memory_reserved_mb": torch.cuda.memory_reserved(i) / 1024 / 1024,
            "max_memory_mb": torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
        }
        
        # Calculate percentage
        if gpu_stats["max_memory_mb"] > 0:
            gpu_stats["utilization_percent"] = (gpu_stats["memory_allocated_mb"] / gpu_stats["max_memory_mb"]) * 100
        else:
            gpu_stats["utilization_percent"] = 0
            
        gpu_info.append(gpu_stats)
    
    return gpu_info

class TrainingMonitor:
    """
    Monitor and record training metrics including time, memory, and GPU usage.
    """
    def __init__(self, log_interval=1000):
        self.start_time = None
        self.end_time = None
        self.log_interval = log_interval
        self.logs = {
            "timestamps": [],
            "ram_usage": [],
            "gpu_usage": [],
            "timesteps": []
        }
    
    def start(self):
        """Start monitoring training"""
        self.start_time = time.time()
        # Log initial resource usage
        self._log_resources(0)
        return self.start_time
        
    def end(self):
        """End monitoring training"""
        self.end_time = time.time()
        # Log final resource usage
        self._log_resources(None)  # None indicates final step
        return self.end_time
    
    def _log_resources(self, timestep):
        """Log current resource usage"""
        timestamp = time.time()
        ram_usage = get_memory_usage()
        gpu_usage = get_gpu_usage()
        
        self.logs["timestamps"].append(timestamp)
        self.logs["ram_usage"].append(ram_usage)
        self.logs["gpu_usage"].append(gpu_usage)
        self.logs["timesteps"].append(timestep)
    
    def log_progress(self, timestep):
        """Log progress during training at specified intervals"""
        if timestep % self.log_interval == 0:
            self._log_resources(timestep)
    
    def get_total_time(self):
        """Get total training time in seconds"""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def get_formatted_time(self):
        """Get total training time as a formatted string"""
        total_seconds = self.get_total_time()
        return str(timedelta(seconds=total_seconds))
    
    def get_summary(self):
        """Get a summary of the training resources"""
        if not self.logs["ram_usage"]:
            return {"error": "No logs recorded"}
        
        # Calculate RAM statistics
        ram_usage_mb = [log["memory_usage_mb"] for log in self.logs["ram_usage"]]
        ram_stats = {
            "min_mb": min(ram_usage_mb),
            "max_mb": max(ram_usage_mb),
            "avg_mb": np.mean(ram_usage_mb),
            "final_mb": ram_usage_mb[-1]
        }
        
        # Calculate GPU statistics if available
        gpu_stats = None
        if self.logs["gpu_usage"][0] is not None:
            gpu_stats = []
            gpu_count = len(self.logs["gpu_usage"][0])
            
            for gpu_idx in range(gpu_count):
                allocated_mb = [log[gpu_idx]["memory_allocated_mb"] if log else 0 for log in self.logs["gpu_usage"] if log]
                if allocated_mb:
                    gpu_stats.append({
                        "device_id": gpu_idx,
                        "name": self.logs["gpu_usage"][0][gpu_idx]["name"],
                        "min_mb": min(allocated_mb),
                        "max_mb": max(allocated_mb),
                        "avg_mb": np.mean(allocated_mb),
                        "final_mb": allocated_mb[-1] if allocated_mb else 0
                    })
        
        return {
            "total_time_seconds": self.get_total_time(),
            "formatted_time": self.get_formatted_time(),
            "ram_usage": ram_stats,
            "gpu_usage": gpu_stats
        }
    
    def print_summary(self):
        """Print a summary of the training resources"""
        summary = self.get_summary()
        
        print("\n===== TRAINING RESOURCE SUMMARY =====")
        print(f"Total training time: {summary['formatted_time']}")
        
        print("\nRAM Usage:")
        print(f"  Min: {summary['ram_usage']['min_mb']:.2f} MB")
        print(f"  Max: {summary['ram_usage']['max_mb']:.2f} MB")
        print(f"  Avg: {summary['ram_usage']['avg_mb']:.2f} MB")
        print(f"  Final: {summary['ram_usage']['final_mb']:.2f} MB")
        
        if summary.get('gpu_usage'):
            print("\nGPU Usage:")
            for gpu in summary['gpu_usage']:
                print(f"  GPU {gpu['device_id']} ({gpu['name']}):")
                print(f"    Min: {gpu['min_mb']:.2f} MB")
                print(f"    Max: {gpu['max_mb']:.2f} MB")
                print(f"    Avg: {gpu['avg_mb']:.2f} MB")
                print(f"    Final: {gpu['final_mb']:.2f} MB")
        else:
            print("\nNo GPU usage recorded or no GPU available")
    
    def save_logs(self, path):
        """Save monitoring logs to a JSON file"""
        import json
        
        # Convert numpy values to Python native types
        serializable_logs = {
            "timestamps": self.logs["timestamps"],
            "ram_usage": self.logs["ram_usage"],
            "gpu_usage": [],  # We'll convert GPU logs separately
            "timesteps": self.logs["timesteps"],
            "summary": self.get_summary()
        }
        
        # Convert GPU logs if they exist
        for gpu_log in self.logs["gpu_usage"]:
            if gpu_log is None:
                serializable_logs["gpu_usage"].append(None)
            else:
                serializable_gpu_log = []
                for gpu_device in gpu_log:
                    serializable_gpu_device = {k: float(v) if isinstance(v, np.number) else v 
                                              for k, v in gpu_device.items()}
                    serializable_gpu_log.append(serializable_gpu_device)
                serializable_logs["gpu_usage"].append(serializable_gpu_log)
        
        with open(path, 'w') as f:
            json.dump(serializable_logs, f, indent=4)