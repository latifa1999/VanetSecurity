import os
import psutil
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class ResourceTracker:
    """
    A class for tracking GPU and RAM usage during model inference.
    """
    def __init__(self, log_dir='resource_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.gpu_memory_usage = []
        self.ram_usage = []
        self.timestamps = []
        self.start_time = None
        
    def start(self):
        """Start tracking resources"""
        self.start_time = time.time()
        self.gpu_memory_usage = []
        self.ram_usage = []
        self.timestamps = []
        
    def update(self):
        """Record current resource usage"""
        # Record elapsed time
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)
        
        # Record RAM usage
        process = psutil.Process(os.getpid())
        ram_info = process.memory_info()
        ram_usage_gb = ram_info.rss / (1024 ** 3)  # Convert to GB
        self.ram_usage.append(ram_usage_gb)
        
        # Record GPU memory usage if available
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            self.gpu_memory_usage.append(gpu_memory_used)
        else:
            self.gpu_memory_usage.append(0)
            
    def get_summary(self):
        """Get summary statistics of resource usage"""
        summary = {
            "ram_max_gb": max(self.ram_usage) if self.ram_usage else 0,
            "ram_mean_gb": np.mean(self.ram_usage) if self.ram_usage else 0,
            "gpu_max_gb": max(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
            "gpu_mean_gb": np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
            "duration_seconds": self.timestamps[-1] if self.timestamps else 0
        }
        return summary
        
    def plot_usage(self, model_name, test_set_name):
        """Generate plots of resource usage over time"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{test_set_name}_resource_usage_{timestamp}.png"
        filepath = os.path.join(self.log_dir, filename)
        
        plt.figure(figsize=(12, 8))
        
        # Plot RAM usage
        plt.subplot(2, 1, 1)
        plt.plot(self.timestamps, self.ram_usage, 'b-', label='RAM Usage (GB)')
        plt.title(f'RAM Usage During {model_name} Testing on {test_set_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('RAM (GB)')
        plt.grid(True)
        plt.legend()
        
        # Plot GPU usage
        plt.subplot(2, 1, 2)
        plt.plot(self.timestamps, self.gpu_memory_usage, 'r-', label='GPU Memory Usage (GB)')
        plt.title(f'GPU Memory Usage During {model_name} Testing on {test_set_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('GPU Memory (GB)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath
        
    def save_log(self, model_name, test_set_name):
        """Save resource usage data to a log file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{test_set_name}_resource_log_{timestamp}.txt"
        filepath = os.path.join(self.log_dir, filename)
        
        summary = self.get_summary()
        
        with open(filepath, 'w') as f:
            f.write(f"Resource usage for {model_name} on {test_set_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Summary Statistics:\n")
            f.write(f"Max RAM Usage: {summary['ram_max_gb']:.4f} GB\n")
            f.write(f"Mean RAM Usage: {summary['ram_mean_gb']:.4f} GB\n")
            f.write(f"Max GPU Memory Usage: {summary['gpu_max_gb']:.4f} GB\n")
            f.write(f"Mean GPU Memory Usage: {summary['gpu_mean_gb']:.4f} GB\n")
            f.write(f"Test Duration: {summary['duration_seconds']:.2f} seconds\n\n")
            
            f.write("Detailed Measurements:\n")
            f.write("Time(s),RAM(GB),GPU(GB)\n")
            for i in range(len(self.timestamps)):
                f.write(f"{self.timestamps[i]:.2f},{self.ram_usage[i]:.4f},{self.gpu_memory_usage[i]:.4f}\n")
                
        return filepath

# Function to get current GPU utilization percentage
def get_gpu_utilization():
    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return info.gpu  # GPU utilization percentage
        except (ImportError, pynvml.NVMLError):
            # If pynvml is not available or there's an error
            return None
    return None