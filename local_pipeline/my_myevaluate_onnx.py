#!/usr/bin/env python3
"""
Test ONNX model for STHN (similar to my_myevaluate.py but using ONNX Runtime).

Usage:
  # One-stage:
  python3 -m local_pipeline.my_myevaluate_onnx \
      --eval_model trt/one_stage_v2/sthn_coarse.onnx \
      --database_size 1536 --resize_width 256

  # Two-stage:
  python3 -m local_pipeline.my_myevaluate_onnx \
      --eval_model trt/two_stages/sthn_coarse.onnx \
      --eval_model_fine trt/two_stages/sthn_fine.onnx \
      --two_stages --database_size 1536 --resize_width 256
"""

import numpy as np
import os
import sys
import argparse
import time
from PIL import Image
import pandas as pd

import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# Image transforms (same as my_myevaluate.py)
base_transform = transforms.Compose([
    transforms.Resize([256, 256]),
])

query_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])


class ONNXModel:
    """Wrapper for ONNX model inference."""
    
    def __init__(self, onnx_path: str, device: str = 'cuda'):
        """
        Load ONNX model.
        
        Args:
            onnx_path: Path to .onnx file
            device: 'cuda' or 'cpu'
        """
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        
        print(f"Loading ONNX model: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"  Inputs: {self.input_names}")
        print(f"  Outputs: {self.output_names}")
    
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Run inference.
        
        Args:
            image1: [1, 3, H, W] numpy array (float32)
            image2: [1, 3, H, W] numpy array (float32)
        
        Returns:
            four_pred: [1, 2, 2, 2] numpy array
        """
        inputs = {
            self.input_names[0]: image1,
            self.input_names[1]: image2
        }
        outputs = self.session.run(self.output_names, inputs)
        return outputs[0]


class STHNOnnx:
    """STHN model wrapper for ONNX inference (one-stage or two-stage)."""
    
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.resize_width = args.resize_width
        self.database_size = args.database_size
        
        # Load coarse model
        self.coarse_model = ONNXModel(args.eval_model, device=self.device)
        
        # Load fine model if two-stage
        self.fine_model = None
        if args.two_stages and args.eval_model_fine:
            self.fine_model = ONNXModel(args.eval_model_fine, device=self.device)
        
        # Precompute corner template
        self.four_point_org_single = np.zeros((1, 2, 2, 2), dtype=np.float32)
        self.four_point_org_single[:, :, 0, 0] = [0, 0]
        self.four_point_org_single[:, :, 0, 1] = [self.resize_width - 1, 0]
        self.four_point_org_single[:, :, 1, 0] = [0, self.resize_width - 1]
        self.four_point_org_single[:, :, 1, 1] = [self.resize_width - 1, self.resize_width - 1]
    
    def preprocess_satellite(self, img_path: str) -> np.ndarray:
        """Load and preprocess satellite image."""
        img = Image.open(img_path).convert("RGB")
        img_tensor = TF.to_tensor(img)  # [3, H, W]
        # Resize to resize_width for coarse model
        img_resized = F.interpolate(
            img_tensor.unsqueeze(0), 
            size=(self.resize_width, self.resize_width), 
            mode='bilinear', 
            align_corners=True
        )
        return img_resized.numpy().astype(np.float32)
    
    def preprocess_thermal(self, img_path: str) -> np.ndarray:
        """Load and preprocess thermal/query image."""
        img = Image.open(img_path)
        img_tensor = base_transform(query_transform(img))  # [3, 256, 256]
        return img_tensor.unsqueeze(0).numpy().astype(np.float32)
    
    def get_cropped_satellite(self, img1_tensor: torch.Tensor, four_pred: np.ndarray, 
                               fine_padding: int = 0) -> np.ndarray:
        """
        Crop satellite image based on coarse prediction for fine stage.
        
        Args:
            img1_tensor: Original satellite image tensor [1, 3, H, W]
            four_pred: Coarse prediction [1, 2, 2, 2]
            fine_padding: Padding around crop region
        
        Returns:
            Cropped and resized image [1, 3, 256, 256] as numpy
        """
        four_pred_torch = torch.from_numpy(four_pred)
        four_point_org = torch.from_numpy(self.four_point_org_single)
        
        # Add predicted offset to get corners
        four_point = four_pred_torch + four_point_org
        
        # Scale to original image size
        alpha = self.database_size / self.resize_width
        x = four_point[:, 0] * alpha  # [1, 2, 2]
        y = four_point[:, 1] * alpha
        
        # Get bounding box
        x_flat = x.view(-1)
        y_flat = y.view(-1)
        left = x_flat.min().item()
        right = x_flat.max().item()
        top = y_flat.min().item()
        bottom = y_flat.max().item()
        
        # Make square and add padding
        w = max(right - left, bottom - top) + 2 * fine_padding
        cx = (left + right) / 2
        cy = (top + bottom) / 2
        
        # Crop coordinates
        x1 = int(max(0, cx - w / 2))
        y1 = int(max(0, cy - w / 2))
        x2 = int(min(self.database_size, cx + w / 2))
        y2 = int(min(self.database_size, cy + w / 2))
        
        # Crop and resize
        cropped = img1_tensor[:, :, y1:y2, x1:x2]
        cropped_resized = F.interpolate(
            cropped, 
            size=(self.resize_width, self.resize_width), 
            mode='bilinear', 
            align_corners=True
        )
        
        # Store crop info for combining results
        self.crop_info = {
            'x1': x1, 'y1': y1,
            'w': w,
            'alpha': alpha
        }
        
        return cropped_resized.numpy().astype(np.float32)
    
    def combine_coarse_fine(self, four_pred_coarse: np.ndarray, 
                            four_pred_fine: np.ndarray) -> np.ndarray:
        """Combine coarse and fine predictions."""
        alpha = self.database_size / self.resize_width
        w = self.crop_info['w']
        x1 = self.crop_info['x1']
        y1 = self.crop_info['y1']
        
        # Scale fine prediction to crop region, then to original
        kappa = w / self.resize_width / alpha
        
        # Fine prediction offset within crop
        fine_offset_x = four_pred_fine[:, 0] * kappa + x1 / alpha - self.four_point_org_single[:, 0]
        fine_offset_y = four_pred_fine[:, 1] * kappa + y1 / alpha - self.four_point_org_single[:, 1]
        
        four_pred_combined = np.stack([fine_offset_x, fine_offset_y], axis=1)
        return four_pred_combined
    
    def predict(self, img1_path: str, img2_path: str) -> np.ndarray:
        """
        Run full prediction pipeline.
        
        Args:
            img1_path: Path to satellite image
            img2_path: Path to thermal/query image
        
        Returns:
            four_pred: [1, 2, 2, 2] corner displacement prediction
        """
        # Load original satellite image (for two-stage cropping)
        img1_full = Image.open(img1_path).convert("RGB")
        img1_tensor = TF.to_tensor(img1_full).unsqueeze(0)  # [1, 3, H, W]
        
        # Preprocess
        img1 = self.preprocess_satellite(img1_path)
        img2 = self.preprocess_thermal(img2_path)
        
        # Coarse stage
        four_pred = self.coarse_model(img1, img2)
        
        # Fine stage (if two-stage)
        if self.fine_model is not None:
            img1_crop = self.get_cropped_satellite(img1_tensor, four_pred)
            four_pred_fine = self.fine_model(img1_crop, img2)
            four_pred = self.combine_coarse_fine(four_pred, four_pred_fine)
        
        return four_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Test ONNX STHN model")
    parser.add_argument('--eval_model', type=str, required=True,
                        help='Path to coarse ONNX model')
    parser.add_argument('--eval_model_fine', type=str, default=None,
                        help='Path to fine ONNX model (for two-stage)')
    parser.add_argument('--two_stages', action='store_true',
                        help='Use two-stage inference')
    parser.add_argument('--resize_width', type=int, default=256,
                        help='Model input size')
    parser.add_argument('--database_size', type=int, default=1536,
                        help='Original satellite image size')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device for inference')
    parser.add_argument('--output_excel', type=str, default='js_excels/predicted-onnx.xlsx',
                        help='Output Excel file path')
    return parser.parse_args()


def test(args):
    """Main test function - matches my_myevaluate.py logic."""
    
    # Load ONNX model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.device == 'cuda' else ['CPUExecutionProvider']
    print(f"Loading ONNX model: {args.eval_model}")
    session = ort.InferenceSession(args.eval_model, providers=providers)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"  Inputs: {input_names}")
    print(f"  Outputs: {output_names}")
    
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
            
            # Run ONNX inference
            outputs = session.run(output_names, {
                input_names[0]: img1_resized.numpy().astype(np.float32),
                input_names[1]: img2.numpy().astype(np.float32)
            })
            four_pred = outputs[0]
            
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


def test_single_image(args, img1_path: str, img2_path: str):
    """Test with a single image pair."""
    model = STHNOnnx(args)
    
    print(f"\nTesting single image pair:")
    print(f"  Satellite: {img1_path}")
    print(f"  Thermal:   {img2_path}")
    
    # Warmup
    for _ in range(3):
        _ = model.predict(img1_path, img2_path)
    
    # Timed run
    times = []
    for _ in range(10):
        start = time.time()
        four_pred = model.predict(img1_path, img2_path)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    
    # Convert to corner points
    four_point_org_single = np.zeros((1, 2, 2, 2), dtype=np.float32)
    four_point_org_single[:, :, 0, 0] = [0, 0]
    four_point_org_single[:, :, 0, 1] = [args.resize_width - 1, 0]
    four_point_org_single[:, :, 1, 0] = [0, args.resize_width - 1]
    four_point_org_single[:, :, 1, 1] = [args.resize_width - 1, args.resize_width - 1]
    
    four_point_1 = four_pred + four_point_org_single
    four_point_1 = four_point_1.reshape(1, 2, 4).transpose(0, 2, 1)
    four_point_1_mul6 = four_point_1 * (args.database_size / args.resize_width)
    
    print(f"\nResults:")
    print(f"  four_pred (displacement):")
    print(f"    {four_pred}")
    print(f"  four_point (absolute coords, scaled):")
    print(f"    {four_point_1_mul6.squeeze()}")
    print(f"  Center: {four_point_1_mul6.squeeze().mean(axis=0)}")
    print(f"\n  Avg inference time: {avg_time*1000:.2f} ms")
    print(f"  FPS: {1/avg_time:.1f}")


if __name__ == '__main__':
    args = parse_args()
    
    # Check if specific test images are provided
    if len(sys.argv) > 1 and '--test_single' in sys.argv:
        # Example single image test
        img1 = "js_datasets/Dehat/satellite/1.tif"
        img2 = "js_datasets/Dehat/thermal/1_1.tif"
        test_single_image(args, img1, img2)
    else:
        test(args)
