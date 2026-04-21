# 🔧 Troubleshooting Guide - Jetson Optimization

## Common Issues & Solutions

### 1. ❌ CUDA Out of Memory Error

**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XXX GiB
```

**Causes:**
- `BATCH_SIZE` is too large for your Jetson
- Other GPU processes running
- Model weights not loaded properly

**Solutions:**

**Option A: Reduce Batch Size** (Recommended)
```python
# In my_myevaluate_optimized.py, line 398:
BATCH_SIZE = 2  # ← Try reducing by 1
# For example, if you had BATCH_SIZE = 4, try BATCH_SIZE = 2
```

**Option B: Clear GPU Memory**
```bash
# Kill other GPU processes
sudo pkill -f nvidia-smi

# Or restart Jetson
sudo reboot
```

**Option C: Check Available Memory**
```bash
# Run on Jetson:
nvidia-smi

# Look for "Free" memory in output
# If Free < (Model size × batch_size × 2), reduce batch_size
```

**Test if it works:**
```bash
python my_myevaluate_optimized.py
# If it completes first batch, you're good!
```

---

### 2. ❌ Script is Very Slow

**Symptoms:**
- 0.5-1 FPS (same as original)
- Batch time increasing with each iteration
- GPU memory usage growing

**Possible Causes:**
- `BATCH_SIZE` set too high (GPU throttling)
- Float16 not being used
- Image loading bottleneck

**Solutions:**

**Option A: Check Float16 is Enabled**
```bash
# Add this to see if float16 is working:
# Edit my_myevaluate_optimized.py around line 250:

print(next(model.netG.parameters()).dtype)  # Should print: torch.float16
```

**Option B: Reduce Batch Size**
```python
# If batch_size=4, try batch_size=2
BATCH_SIZE = 2
```

**Option C: Check Image Loading Time**
```python
# Monitor where time is spent. In optimized script, watch:
[GPU Memory During processing] messages
# If time increases, image loading is bottleneck
# Switch to async loading (see JETSON_OPTIMIZATION_GUIDE.md)
```

---

### 3. ❌ Model Loading Fails

**Error Message:**
```
FileNotFoundError: [Errno 2] No such file or directory: '...'
```

**Causes:**
- Incorrect model path in arguments
- Model file doesn't exist
- Wrong working directory

**Solutions:**

**Option A: Check Model Path**
```bash
# Check if model exists:
ls -la js_models/1536_two_stages/STHN.onnx
# If not found, update args.eval_model path
```

**Option B: Check Working Directory**
```bash
# Make sure you're in the right directory:
pwd  # Should be: /path/to/STHN-JetsonONX8/local_pipeline

cd /path/to/STHN-JetsonONX8/local_pipeline
python my_myevaluate_optimized.py
```

**Option C: Use Absolute Paths**
```python
# In my_myevaluate_optimized.py, modify parser args:
args.eval_model = "/absolute/path/to/model.pth"
```

---

### 4. ❌ Results Different from Original

**Symptoms:**
- Different corner point values
- Excel output has different numbers

**Causes:**
- Float16 precision differences (expected)
- Random seed differences
- Batch processing order

**Solutions:**

**This is NORMAL** - Float16 has lower precision than float32
```python
# Precision differences are typically < 0.1% 
# This is acceptable for real-world applications

# If you need exact match, disable float16:
# model.netG.float()  # Not recommended on Jetson
```

**To verify it's just precision:**
```python
# Check if differences are small:
import numpy as np
original_result = [...]
optimized_result = [...]
diff = np.abs(np.array(original_result) - np.array(optimized_result))
print(f"Max difference: {diff.max():.6f}")  # Should be < 0.01
```

---

### 5. ❌ ImportError: No module named 'torch'

**Error Message:**
```
ImportError: No module named 'torch'
```

**Causes:**
- PyTorch not installed for Jetson
- Wrong Python version
- Virtual environment not activated

**Solutions:**

**Option A: Install PyTorch for Jetson**
```bash
# For Jetson Orin (JetPack 5.x):
pip install torch torchvision torchaudio

# Or use official wheels:
# wget https://developer.download.nvidia.com/compute/redist/jp/v5x/pytorch/torch-xxx.whl
# pip install torch-xxx.whl
```

**Option B: Verify Python Environment**
```bash
python --version  # Should be Python 3.8+
which python  # Check if correct Python
which pip  # Check if correct pip
```

**Option C: Check Virtual Environment**
```bash
# If using venv:
source venv/bin/activate

# If using conda:
conda activate jetson-env
```

---

### 6. ❌ Pruning Takes Too Long at Startup

**Symptoms:**
- Script waits 30-60 seconds before starting
- Lots of pruning output

**Causes:**
- `ENABLE_PRUNING = True`
- Large model being pruned

**Solutions:**

**Option A: Disable Pruning** (Recommended for inference)
```python
# In my_myevaluate_optimized.py, line 400:
ENABLE_PRUNING = False  # ← Change to False
```

**Option B: Pre-prune Your Model**
```python
# Run pruning once, save the model, then load it:
# 1. Run with ENABLE_PRUNING = True (wait for it)
# 2. Save the pruned model: torch.save(model, 'pruned_model.pth')
# 3. Next time, load from pruned model (skip pruning)
```

---

### 7. ❌ GPU Temperature Getting Too High

**Symptoms:**
- Warning: GPU temperature > 80°C
- Performance drops during inference

**Causes:**
- High batch size causing thermal stress
- Poor ventilation on Jetson
- Long continuous inference

**Solutions:**

**Option A: Reduce Batch Size**
```python
BATCH_SIZE = 1  # Lower batch size = less heat
```

**Option B: Add Cooling**
```bash
# Improve airflow around Jetson
# Add fan if not present
# Ensure heatsink is properly installed
```

**Option C: Monitor Temperature**
```bash
# Watch GPU temp while running:
watch -n 1 "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader"

# If > 85°C, reduce batch_size or add cooling
```

**Option D: Reduce Clock Speed** (Advanced)
```bash
# Temporary (until reboot):
sudo jetson_clocks --show  # See current clocks

# Or set max clocks:
sudo jetson_clocks  # Use max performance

# Or limit clocks:
sudo nvpmodel -m 2  # Lower power mode
```

---

### 8. ❌ Excel File Not Created / Empty Results

**Error:**
```
📁 Saved corner points to js_excels/dehat.xlsx
# But file is empty or doesn't exist
```

**Causes:**
- No successful predictions
- Directory doesn't exist
- Permission denied

**Solutions:**

**Option A: Create Directory**
```bash
mkdir -p js_excels
ls -la js_excels/
```

**Option B: Check Permissions**
```bash
ls -la js_excels/
# Should show: drwxr-xr-x or similar

# If not writable, fix:
chmod 755 js_excels/
```

**Option C: Check If Predictions Ran**
```python
# Look for this in output:
# ✅ Done for image 1, batch_time=0.45s

# If no successful predictions, check:
# 1. Image paths are correct
# 2. Model loaded successfully
# 3. CUDA is working
```

---

### 9. ❌ Cannot Find Image Files

**Error Message:**
```
OSError: cannot identify image file ...
```

**Causes:**
- Wrong image path
- File corrupted
- File format not supported

**Solutions:**

**Option A: Verify Image Paths**
```bash
# Check if images exist:
ls -la js_datasets/Dehat/satellite/1.tif
ls -la js_datasets/Dehat/thermal/1_1.tif

# If not found, check directory structure:
find . -name "*.tif" | head -20
```

**Option B: Check Image Format**
```bash
# Verify TIF files are valid:
file js_datasets/Dehat/satellite/1.tif

# Should say: TIFF image
```

**Option C: Copy Images to Expected Location**
```bash
# If images are in different location:
cp -r /path/to/images/* js_datasets/Dehat/
```

---

### 10. ❌ Model Outputs Look Wrong

**Symptoms:**
- Predictions are all zeros
- Corner points are out of bounds
- Nonsensical values

**Causes:**
- Model not loaded correctly
- Model not in eval mode
- Wrong input format

**Solutions:**

**Option A: Verify Model in Eval Mode**
```python
# This is done automatically, but verify:
model.netG.eval()  # Should be eval, not train
print(model.netG.training)  # Should print: False
```

**Option B: Check Model Outputs Directly**
```python
# Add debug output:
print(f"Model output range: min={four_pred.min():.3f}, max={four_pred.max():.3f}")
# Should be reasonable values, not all zeros

# If all zeros, model isn't running properly
```

**Option C: Verify Input Shapes**
```python
print(f"Input img1 shape: {img1_batch.shape}")  # Should be [B, 3, H, W]
print(f"Input img2 shape: {img2_batch.shape}")  # Should be [B, 3, H, W]

# If wrong, image loading has issues
```

---

## 🆘 Still Having Issues?

### Debug Checklist

- [ ] BATCH_SIZE is appropriate for your Jetson
- [ ] `ENABLE_PRUNING = False` for faster startup
- [ ] Image files exist in correct locations
- [ ] Model weights loaded successfully
- [ ] GPU memory available (`nvidia-smi`)
- [ ] Float16 conversion successful
- [ ] Output directory writable

### Get Diagnostic Info

```bash
# Run this to gather info for troubleshooting:
echo "=== Jetson Info ==="
cat /etc/nv_tegra_release

echo "=== GPU Info ==="
nvidia-smi -q

echo "=== Python & PyTorch ==="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "print(f'CUDA Available: {torch.cuda.is_available()}')"

echo "=== Directory Structure ==="
ls -la js_datasets/Dehat/ | head -10
ls -la js_excels/
```

### Minimal Test Script

```python
# Save as test_setup.py
import torch
import cv2
from PIL import Image

print("✅ PyTorch imported")
print(f"   GPU available: {torch.cuda.is_available()}")
print(f"   GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Test image loading
try:
    img = cv2.imread("js_datasets/Dehat/satellite/1.tif")
    print(f"✅ Image loaded: shape={img.shape}")
except Exception as e:
    print(f"❌ Image loading failed: {e}")

# Test float16
try:
    t = torch.randn(1, 3, 256, 256).cuda().half()
    print(f"✅ Float16 tensor created: {t.dtype}")
except Exception as e:
    print(f"❌ Float16 failed: {e}")

print("\n✅ Setup OK - Ready to run optimized script!")
```

Run it:
```bash
python test_setup.py
```

---

## 📞 Need More Help?

1. **Check the docs:**
   - `QUICK_START.md` - Quick reference
   - `OPTIMIZATION_SUMMARY.md` - Technical details
   - `JETSON_OPTIMIZATION_GUIDE.md` - Advanced tuning

2. **Run the diagnostic:**
   - Use debug checklist above
   - Run `test_setup.py`
   - Check `nvidia-smi` output

3. **Reduce complexity:**
   - Set `BATCH_SIZE = 1`
   - Set `ENABLE_PRUNING = False`
   - Test with just 1 image

---

## ✨ Common Success Indicators

After running successfully, you should see:

```
🚀 Starting inference with batch_size=2

✅ Done for image 1, batch_time=0.45s
✅ Done for image 11, batch_time=0.45s
...

============================================================
📊 PERFORMANCE SUMMARY (Jetson Optimized)
============================================================
Total images processed: 108
Batch size: 2
Average batch time: 0.45 sec
Average time per image: 0.225 sec
Throughput: 4.44 FPS
============================================================

[GPU Memory Final] 0.55GB / 4.00GB

📁 Saved corner points to js_excels/dehat.xlsx
```

If you see this, **congratulations!** Your optimization is working! 🎉
