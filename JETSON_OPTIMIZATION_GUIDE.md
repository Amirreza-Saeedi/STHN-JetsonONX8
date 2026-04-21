# Jetson Optimization Guide for my_myevaluate.py

## Critical Optimizations for Jetson (Limited GPU/CPU resources)

### 1. **MEMORY OPTIMIZATION** ⭐ HIGH PRIORITY
#### Current Issue:
- `four_point_org_single` created on GPU every iteration inside try block
- Models kept in float32 (uses 2x memory vs float16)

#### Solutions:
```python
# Move tensor creation outside loop
four_point_org_single = torch.tensor(
    [[[[0, 0], [args.resize_width - 1, 0]],
    [[0, args.resize_width - 1], [args.resize_width - 1, args.resize_width - 1]]]],
    device="cuda:0",
    dtype=torch.float16
)  # Create ONCE

# Use model.half() for float16 inference
model.netG.half()
model.netG_fine.half()
```

### 2. **BATCH PROCESSING** ⭐ HIGH PRIORITY
#### Current Issue:
- Processing single images one at a time
- Maximum throughput not achieved

#### Solution:
```python
# Load multiple images into batch
batch_size = 4  # Adjust based on Jetson VRAM
for batch_start in range(0, N, batch_size):
    batch_end = min(batch_start + batch_size, N)
    img1_batch = []
    img2_batch = []
    
    for i in range(batch_start, batch_end):
        img1 = F.to_tensor(Image.open(img1_path).convert("RGB"))
        img2 = (base_transform(query_transform(Image.open(img2_path))))
        img1_batch.append(img1)
        img2_batch.append(img2)
    
    img1_batch = torch.stack(img1_batch).to("cuda:0")
    img2_batch = torch.stack(img2_batch).to("cuda:0")
    
    with torch.no_grad():
        model.set_input(img1_batch, img2_batch)
        model.forward()
```

### 3. **IMAGE PREPROCESSING OPTIMIZATION** ⭐ HIGH PRIORITY
#### Current Issue:
- Images loaded to CPU first, then to GPU
- Slow PIL operations on CPU

#### Solutions:
```python
# Load directly to GPU with proper format
import torchvision.io
img1 = torchvision.io.read_image(img1_path).float().div(255).unsqueeze(0).to("cuda:0")

# Or use cv2 which is faster
import cv2
img1_cv = cv2.imread(img1_path)
img1 = torch.from_numpy(img1_cv).permute(2, 0, 1).float().div(255).unsqueeze(0).to("cuda:0")

# Pre-allocate transforms to avoid recreating
```

### 4. **QUANTIZATION** ⭐ HIGH PRIORITY
#### Solution - Use INT8 Quantization:
```python
# For TensorRT (if using ONNX)
from torch.quantization import quantize_dynamic
model.netG = quantize_dynamic(
    model.netG,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
```

### 5. **GPU MEMORY MANAGEMENT**
#### Current Issue:
- No GPU memory cleanup between iterations
- Cache not cleared

#### Solutions:
```python
# At start of loop iteration
torch.cuda.empty_cache()

# Use memory efficient inference
torch.cuda.reset_peak_memory_stats()

# Monitor memory
def log_gpu_mem():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
```

### 6. **REMOVE/OPTIMIZE PRUNING CODE**
#### Current Issue:
- Pruning happens at startup (slow on Jetson)
- Complex surgery operations

#### Options:
```python
# Option A: Skip pruning for inference-only
if args.identity:
    pass
else:
    # Only prune if explicitly needed
    if hasattr(args, 'enable_pruning') and args.enable_pruning:
        # Do pruning
        pass

# Option B: Pre-prune model before deployment
# Save pruned model as checkpoint, load directly
```

### 7. **ASYNC I/O OPTIMIZATION**
#### Solution - Background image loading:
```python
from concurrent.futures import ThreadPoolExecutor
import queue

def load_images_async(paths_queue, results_queue, num_workers=2):
    def worker():
        while True:
            try:
                i, img1_path, img2_path = paths_queue.get(timeout=1)
                img1 = F.to_tensor(Image.open(img1_path).convert("RGB")).unsqueeze(0)
                img2 = (base_transform(query_transform(Image.open(img2_path)))).unsqueeze(0)
                results_queue.put((i, img1, img2))
            except queue.Empty:
                break
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(num_workers):
            executor.submit(worker)

# Usage: prefetch next batch while processing current
```

### 8. **INFERENCE SPEEDUPS**
#### Solutions:
```python
# Use TorchScript (if model compatible)
model.netG = torch.jit.script(model.netG)

# Or trace model
example_input = (torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda())
traced_model = torch.jit.trace(model.netG, example_input)

# Disable grad (you already do this, good!)
torch.set_grad_enabled(False)

# Use cudnn benchmarking
torch.backends.cudnn.benchmark = True
```

### 9. **DATAFRAME OPERATIONS OPTIMIZATION**
#### Current Issue:
- Building list in loop, converting to DataFrame at end (inefficient for large N)

#### Solution:
```python
# Stream write to Excel or CSV instead
# Or use numpy array and convert once
all_corners = np.zeros((N, 11), dtype=np.float32)
# Fill array
df = pd.DataFrame(all_corners, columns=columns)
```

### 10. **TIMING MEASUREMENT OVERHEAD**
#### Current Issue:
- `time.time()` called for every iteration (overhead)

#### Solution:
```python
# Profile at 10% samples instead of 100%
if i % 10 == 0:
    start_time = time.time()
    # ... inference ...
    times.append(time.time() - start_time)
else:
    # ... inference without timing ...
    model.set_input(img1, img2)
    model.forward()
```

## Implementation Priority

1. **First**: Model to float16 + move tensor creation outside loop
2. **Second**: Batch processing + optimize image loading
3. **Third**: Async I/O or TorchScript tracing
4. **Fourth**: Remove/defer pruning operations
5. **Fifth**: Fine-tune batch_size based on Jetson VRAM (2GB/4GB/8GB?)

## Jetson-Specific Settings

```python
# Add to your script
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# For Jetson Xavier/Orin specific optimization
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Check Jetson memory
def get_jetson_info():
    import subprocess
    result = subprocess.run(['nvidia-smi', '-q', '-d', 'MEMORY'], capture_output=True, text=True)
    print(result.stdout)
```

## Expected Performance Gains

- **Model to float16**: 1.8-2.0x faster inference + 50% less memory
- **Batch processing** (batch_size=4): 2-3x throughput improvement
- **Optimized image loading**: 30-40% faster I/O
- **Async I/O**: Additional 20-30% improvement
- **Combined**: **3-5x faster** overall

---

Would you like me to implement any of these optimizations?
