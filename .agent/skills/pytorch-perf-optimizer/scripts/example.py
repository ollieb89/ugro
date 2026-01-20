import torch
import torch.nn as nn
import time
from scripts.setup_perf import setup_pytorch_performance, optimize_model

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 10)
        )

    def forward(self, x):
        return self.net(x)

def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on {device}...")
    
    # 1. Baseline
    model = SimpleModel().to(device)
    x = torch.randn(32, 3, 32, 32).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    start = time.time()
    for _ in range(100):
        _ = model(x)
    baseline_time = time.time() - start
    print(f"Baseline (100 iterations): {baseline_time:.4f}s")
    
    # 2. Optimized
    setup_pytorch_performance()
    model = SimpleModel().to(device)
    model = optimize_model(model)
    
    # Warmup (critical for torch.compile)
    print("Optimization warmup (compiling)...")
    for _ in range(10):
        _ = model(x)
        
    start = time.time()
    for _ in range(100):
        _ = model(x)
    opt_time = time.time() - start
    print(f"Optimized (100 iterations): {opt_time:.4f}s")
    print(f"Speedup: {baseline_time / opt_time:.2f}x")

if __name__ == "__main__":
    benchmark()
