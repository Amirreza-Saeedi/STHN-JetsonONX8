#!/usr/bin/env python3
"""
Test TensorRT engine for STHN (similar to my_myevaluate_onnx.py but using TensorRT).

Usage:
  # One-stage:
  python3 -m local_pipeline.my_myevaluate_trt2 \
      --eval_model trt/one_stage_v2/sthn_coarse.engine \
      --database_size 1536 --resize_width 256

  # Two-stage:
  python3 -m local_pipeline.my_myevaluate_trt2 \
      --eval_model trt/two_stages/sthn_coarse.engine \
      --eval_model_fine trt/two_stages/sthn_fine.engine \
      --two_stages --database_size 1536 --resize_width 256
"""

import numpy as np
import os
import sys
import argparse
import time
from PIL import Image
import pandas as pd

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# TensorRT imports
import tensorrt as trt

# For CUDA memory management without pycuda
import ctypes


# Image transforms (same as my_myevaluate.py)
base_transform = transforms.Compose([
    transforms.Resize([256, 256]),
])

query_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])


class TRTModel:
    """TensorRT engine wrapper for inference using torch CUDA tensors."""
    
    def __init__(self, engine_path: str):
        """
        Load TensorRT engine.
        
        Args:
            engine_path: Path to .engine file
        """
        print(f"Loading TensorRT engine: {engine_path}")
        
        # Create logger
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Get input/output info
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        self.output_shapes = {}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes[name] = shape
            else:
                self.output_names.append(name)
                self.output_shapes[name] = shape
        
        print(f"  Inputs: {self.input_names} -> {self.input_shapes}")
        print(f"  Outputs: {self.output_names} -> {self.output_shapes}")
        
        # Allocate output buffers
        self.output_buffers = {}
        for name in self.output_names:
            shape = self.output_shapes[name]
            # Handle dynamic shapes (replace -1 with 1)
            shape = tuple(s if s > 0 else 1 for s in shape)
            self.output_buffers[name] = torch.empty(shape, dtype=torch.float32, device='cuda')
    
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Run inference.
        
        Args:
            image1: [1, 3, 256, 256] numpy array (float32)
            image2: [1, 3, 256, 256] numpy array (float32)
        
        Returns:
            four_pred: [1, 2, 2, 2] numpy array
        """
        # Convert to torch CUDA tensors
        input1 = torch.from_numpy(image1).cuda().contiguous()
        input2 = torch.from_numpy(image2).cuda().contiguous()
        
        # Set input shapes for dynamic batch
        self.context.set_input_shape(self.input_names[0], input1.shape)
        self.context.set_input_shape(self.input_names[1], input2.shape)
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_names[0], input1.data_ptr())
        self.context.set_tensor_address(self.input_names[1], input2.data_ptr())
        
        for name in self.output_names:
            self.context.set_tensor_address(name, self.output_buffers[name].data_ptr())
        
        # Execute
        stream = torch.cuda.current_stream()
        success = self.context.execute_async_v3(stream.cuda_stream)
        
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        # Sync
        torch.cuda.synchronize()
        
        # Get output
        output = self.output_buffers[self.output_names[0]].cpu().numpy()
        return output


def parse_args():
    parser = argparse.ArgumentParser(description="Test TensorRT STHN model")
    parser.add_argument('--eval_model', type=str, required=True,
                        help='Path to coarse TensorRT engine')
    parser.add_argument('--eval_model_fine', type=str, default=None,
                        help='Path to fine TensorRT engine (for two-stage)')
    parser.add_argument('--two_stages', action='store_true',
                        help='Use two-stage inference')
    parser.add_argument('--resize_width', type=int, default=256,
                        help='Model input size')
    parser.add_argument('--database_size', type=int, default=1536,
                        help='Original satellite image size')
    parser.add_argument('--output_excel', type=str, default='js_excels/predicted-trt.xlsx',
                        help='Output Excel file path')
    return parser.parse_args()


def test(args):
    """Main test function - matches my_myevaluate.py logic."""
    
    # Load TensorRT engine
    model = TRTModel(args.eval_model)
    
    # Fine model for two-stage (if needed)
    fine_model = None
    if args.two_stages and args.eval_model_fine:
        fine_model = TRTModel(args.eval_model_fine)
    
    # Test dataset configuration (same as my_myevaluate.py)
    N = 108  # number of samples
    TH = 9
    
    all_corners = []
    times = []
    
    for i in range(N):
        try:
            # Image paths (same as my_myevaluate.py)
            img1_path = f"js_datasets/Dehat/satellite/{i // TH + 1}.tif"
            img2_path = f"js_datasets/Dehat/thermal/{i // TH + 1}_{i % TH + 1}.tif"
            
            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                print(f"⚠ Skipping {i}: files not found")
                continue
            
            # Load and preprocess images (exactly like my_myevaluate.py)
            img1 = TF.to_tensor(Image.open(img1_path).convert("RGB"))
            img1_resized = F.interpolate(img1.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True)
            
            img2 = base_transform(query_transform(Image.open(img2_path))).unsqueeze(0)
            
            start_time = time.time()
            
            # Run TensorRT inference
            four_pred = model(
                img1_resized.numpy().astype(np.float32),
                img2.numpy().astype(np.float32)
            )
            
            # Convert to corner points (exactly like my_myevaluate.py)
            four_point_org_single = np.zeros((1, 2, 2, 2), dtype=np.float32)
            four_point_org_single[:, :, 0, 0] = [0, 0]
            four_point_org_single[:, :, 0, 1] = [args.resize_width - 1, 0]
            four_point_org_single[:, :, 1, 0] = [0, args.resize_width - 1]
            four_point_org_single[:, :, 1, 1] = [args.resize_width - 1, args.resize_width - 1]
            
            four_point_1 = four_pred + four_point_org_single
            four_point_1 = four_point_1.reshape(1, 2, 4).transpose(0, 2, 1)  # [1, 4, 2]
            four_point_1_mul6 = four_point_1 * 6  # Scale to 1536 (database_size / resize_width)
            
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            
            # Extract points (same format as my_myevaluate.py)
            points = four_point_1_mul6.squeeze(0).tolist()  # 4 × 2 list
            flat_points = [coord for point in points for coord in point]
            
            all_corners.append([i] + flat_points + [img1_path, img2_path])
            
            print(f"✅ Done for image {i + 1} ({elapsed:.4f}s)")
            
        except Exception as e:
            print(f"❌ Error in image {i}: {e}")
            import traceback
            traceback.print_exc()
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n📊 Average processing time per image: {avg_time:.4f} sec")
        print(f"📊 FPS: {1/avg_time:.2f}")
    
    # Save to Excel (same format as my_myevaluate.py)
    if all_corners:
        os.makedirs(os.path.dirname(args.output_excel) or '.', exist_ok=True)
        columns = ["image_index", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "sat", "th"]
        df = pd.DataFrame(all_corners, columns=columns)
        df.to_excel(args.output_excel, index=False)
        print(f"📁 Saved results to {args.output_excel}")


if __name__ == '__main__':
    args = parse_args()
    test(args)
