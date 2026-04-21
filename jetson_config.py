"""
JETSON OPTIMIZATION CONFIGURATION
================================

Copy the settings for your Jetson model and update the main section of my_myevaluate_optimized.py

Usage:
    1. Find your Jetson model below
    2. Copy the BATCH_SIZE value
    3. Update line 398 in my_myevaluate_optimized.py
    4. Run the script
"""

# ========================================================================
# JETSON MODELS - SELECT YOUR HARDWARE
# ========================================================================

# JETSON NANO 2GB (Entry Level)
# Memory: 2GB LPDDR4
# CUDA Cores: 128
# Expected Performance: 0.5-1 FPS
CONFIG_NANO_2GB = {
    "name": "Jetson Nano 2GB",
    "batch_size": 1,
    "enable_pruning": False,
    "expected_fps": "0.5-1",
    "max_memory_gb": 2,
    "notes": "Very limited - use batch_size=1 only"
}

# JETSON NANO 4GB (Entry Level)
# Memory: 4GB LPDDR4
# CUDA Cores: 128
# Expected Performance: 1-2 FPS
CONFIG_NANO_4GB = {
    "name": "Jetson Nano 4GB",
    "batch_size": 2,
    "enable_pruning": False,
    "expected_fps": "1-2",
    "max_memory_gb": 4,
    "notes": "Good entry-level option"
}

# JETSON XAVIER NX (Mid-Range)
# Memory: 8GB LPDDR4x
# CUDA Cores: 384
# Expected Performance: 2-3 FPS
CONFIG_XAVIER_NX = {
    "name": "Jetson Xavier NX",
    "batch_size": 4,
    "enable_pruning": False,
    "expected_fps": "2-3",
    "max_memory_gb": 8,
    "notes": "Best balance of power and performance"
}

# JETSON ORIN NANO (Budget Orin)
# Memory: 4GB or 8GB LPDDR5x
# CUDA Cores: 1024 (576 sparse)
# Expected Performance: 2-4 FPS
CONFIG_ORIN_NANO = {
    "name": "Jetson Orin Nano",
    "batch_size": 4,
    "enable_pruning": False,
    "expected_fps": "2-4",
    "max_memory_gb": "4-8",
    "notes": "For 4GB: use batch_size=2, For 8GB: batch_size=4"
}

# JETSON ORIN NX (Mid-Range Orin)
# Memory: 8GB LPDDR5x
# CUDA Cores: 1536 (864 sparse)
# Expected Performance: 3-4 FPS
CONFIG_ORIN_NX = {
    "name": "Jetson Orin NX",
    "batch_size": 6,
    "enable_pruning": False,
    "expected_fps": "3-4",
    "max_memory_gb": 8,
    "notes": "Good mid-range option"
}

# JETSON AGX ORIN (High-End)
# Memory: 12GB or 64GB LPDDR5x
# CUDA Cores: 2048 (1152 sparse)
# Expected Performance: 5-8 FPS
CONFIG_AGX_ORIN = {
    "name": "Jetson AGX Orin",
    "batch_size": 8,
    "enable_pruning": False,
    "expected_fps": "5-8",
    "max_memory_gb": "12-64",
    "notes": "High-performance option"
}

# JETSON TX2 (Legacy)
# Memory: 8GB LPDDR4
# CUDA Cores: 256
# Expected Performance: 0.5-1 FPS
CONFIG_TX2 = {
    "name": "Jetson TX2",
    "batch_size": 2,
    "enable_pruning": True,  # Might need pruning
    "expected_fps": "0.5-1",
    "max_memory_gb": 8,
    "notes": "Legacy hardware - pruning recommended"
}

# JETSON ORIN AGX 64GB (Maximum)
# Memory: 64GB LPDDR5x
# CUDA Cores: 2048
# Expected Performance: 8-12+ FPS
CONFIG_AGX_ORIN_64GB = {
    "name": "Jetson AGX Orin 64GB",
    "batch_size": 16,
    "enable_pruning": False,
    "expected_fps": "8-12+",
    "max_memory_gb": 64,
    "notes": "Maximum performance configuration"
}

# ========================================================================
# AUTOMATIC DETECTION (Optional)
# ========================================================================

def get_jetson_config():
    """
    Automatically detect Jetson model and return appropriate config
    """
    import subprocess
    import os
    
    try:
        # Try to read Jetson model from /etc/nv_tegra_release
        if os.path.exists('/etc/nv_tegra_release'):
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read().lower()
                
                if 'orin agx' in content and '64gb' in content:
                    return CONFIG_AGX_ORIN_64GB
                elif 'orin agx' in content:
                    return CONFIG_AGX_ORIN
                elif 'orin nx' in content:
                    return CONFIG_ORIN_NX
                elif 'orin nano' in content:
                    return CONFIG_ORIN_NANO
                elif 'xavier nx' in content:
                    return CONFIG_XAVIER_NX
                elif 'nano' in content and '4gb' in content:
                    return CONFIG_NANO_4GB
                elif 'nano' in content:
                    return CONFIG_NANO_2GB
                elif 'tx2' in content:
                    return CONFIG_TX2
        
        # Fallback: check GPU memory
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        memory_str = result.stdout.strip().split()[0]
        memory_mb = int(memory_str)
        memory_gb = memory_mb // 1024
        
        if memory_gb <= 2:
            return CONFIG_NANO_2GB
        elif memory_gb <= 4:
            return CONFIG_NANO_4GB
        elif memory_gb <= 8:
            return CONFIG_XAVIER_NX
        else:
            return CONFIG_AGX_ORIN
    
    except Exception as e:
        print(f"Could not auto-detect Jetson model: {e}")
        print("Defaulting to Xavier NX configuration")
        return CONFIG_XAVIER_NX

# ========================================================================
# USAGE EXAMPLES
# ========================================================================

def print_all_configs():
    """Print all available configurations"""
    configs = [
        CONFIG_NANO_2GB,
        CONFIG_NANO_4GB,
        CONFIG_XAVIER_NX,
        CONFIG_ORIN_NANO,
        CONFIG_ORIN_NX,
        CONFIG_AGX_ORIN,
        CONFIG_TX2,
        CONFIG_AGX_ORIN_64GB,
    ]
    
    print("Available Jetson Configurations:")
    print("=" * 70)
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Enable Pruning: {config['enable_pruning']}")
        print(f"  Expected FPS: {config['expected_fps']}")
        print(f"  Memory: {config['max_memory_gb']}GB")
        print(f"  Notes: {config['notes']}")

# ========================================================================
# QUICK REFERENCE
# ========================================================================

"""
QUICK REFERENCE - Copy/Paste the BATCH_SIZE for your Jetson:

Jetson Nano 2GB:        BATCH_SIZE = 1
Jetson Nano 4GB:        BATCH_SIZE = 2
Jetson Xavier NX:       BATCH_SIZE = 4
Jetson Orin Nano (4GB): BATCH_SIZE = 2
Jetson Orin Nano (8GB): BATCH_SIZE = 4
Jetson Orin NX:         BATCH_SIZE = 6
Jetson AGX Orin (12GB): BATCH_SIZE = 8
Jetson AGX Orin (64GB): BATCH_SIZE = 16
Jetson TX2:             BATCH_SIZE = 2

Then paste into my_myevaluate_optimized.py line 398:
    BATCH_SIZE = <your_number>

And run:
    python my_myevaluate_optimized.py
"""

# ========================================================================
# HOW TO USE IN YOUR SCRIPT
# ========================================================================

"""
Method 1: Manual Configuration
------------------------------
In my_myevaluate_optimized.py, at the bottom:

    # Auto-detect or manually set
    config = CONFIG_XAVIER_NX  # Change this to your model
    
    BATCH_SIZE = config['batch_size']
    ENABLE_PRUNING = config['enable_pruning']
    
    print(f"Using config: {config['name']}")
    print(f"Expected FPS: {config['expected_fps']}")

Method 2: Auto-Detection
------------------------
    # Auto-detect your Jetson
    config = get_jetson_config()
    
    BATCH_SIZE = config['batch_size']
    ENABLE_PRUNING = config['enable_pruning']
    
    print(f"Detected: {config['name']}")
    print(f"Using batch_size: {BATCH_SIZE}")

Method 3: Command Line Override (Future Enhancement)
-----------------------------------------------------
    python my_myevaluate_optimized.py --batch_size 4 --enable_pruning false
"""

# ========================================================================
# TESTING YOUR CONFIG
# ========================================================================

"""
How to test if your batch_size is optimal:

1. Start with your recommended batch_size
2. Run a few iterations and watch GPU memory:
   
   watch -n 1 nvidia-smi

3. If you see "out of memory" errors, reduce batch_size by 1

4. If GPU is only 50% utilized, try increasing batch_size by 1

5. Once found, it should be stable for all 108 images

Expected memory usage by batch_size:
  batch_size=1: 1-2 GB
  batch_size=2: 2-3 GB
  batch_size=4: 3-4 GB
  batch_size=8: 4-6 GB
"""

if __name__ == "__main__":
    # Print available configs
    print_all_configs()
    
    # Try auto-detection
    print("\n" + "="*70)
    print("Auto-detection Result:")
    try:
        detected = get_jetson_config()
        print(f"Detected Model: {detected['name']}")
        print(f"Recommended Batch Size: {detected['batch_size']}")
        print(f"Expected FPS: {detected['expected_fps']}")
    except Exception as e:
        print(f"Could not auto-detect: {e}")
