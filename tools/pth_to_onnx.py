#!/usr/bin/env python3
"""
Convert STHN .pth models to ONNX format for TensorRT deployment.

Supports:
- One-stage models (coarse only)
- Two-stage models (coarse + fine)

Uses ONNX-compatible perspective transform (Gaussian elimination instead of SVD).

Usage examples:
  # One-stage (coarse only):
  python3 -m tools.pth_to_onnx \
      --pth js_models/1536_one_stage/STHN.pth \
      --out_dir trt/one_stage \
      --database_size 1536 --resize_width 256 --corr_level 4 --iters 6

  # Two-stage (coarse + fine):
  python3 -m tools.pth_to_onnx \
      --pth js_models/1536_two_stages/STHN.pth \
      --out_dir trt/two_stages \
      --two_stages \
      --database_size 1536 --resize_width 256 --corr_level 4 --iters 6
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local_pipeline"))

from local_pipeline.extractor import BasicEncoderQuarter
from local_pipeline.corr import CorrBlock
from local_pipeline.update import GMA
from local_pipeline.utils import coords_grid


def solve_8x8_gauss_pivoting(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve 8x8 linear system Ax = b using Gaussian elimination with partial pivoting.
    ONNX-compatible version that handles zero pivots correctly.
    
    Args:
        A: [B, 8, 8] coefficient matrices
        b: [B, 8] right-hand side vectors
    
    Returns:
        x: [B, 8] solution vectors
    """
    B = A.shape[0]
    device = A.device
    dtype = A.dtype
    n = 8
    
    # Create augmented matrix [A | b]
    Ab = torch.cat([A, b.unsqueeze(-1)], dim=-1).clone()  # [B, 8, 9]
    
    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot (max absolute value in column col, from row col onwards)
        col_vals = Ab[:, col:, col].abs()  # [B, n-col]
        _, max_idx = col_vals.max(dim=1)  # [B]
        max_idx = max_idx + col  # Adjust to actual row index
        
        # Swap rows: row col <-> row max_idx
        # We need to do this in a way that works with ONNX tracing
        # Since batch size is typically 1, we can handle this
        for batch_idx in range(B):
            pivot_row = max_idx[batch_idx].item()
            if pivot_row != col:
                # Swap rows
                Ab[batch_idx, col, :], Ab[batch_idx, pivot_row, :] = \
                    Ab[batch_idx, pivot_row, :].clone(), Ab[batch_idx, col, :].clone()
        
        # Scale pivot row
        pivot = Ab[:, col, col:col+1].clone()
        pivot = torch.where(pivot.abs() < 1e-12, torch.ones_like(pivot) * 1e-12, pivot)
        Ab[:, col, :] = Ab[:, col, :] / pivot
        
        # Eliminate below pivot
        for row in range(col + 1, n):
            factor = Ab[:, row, col:col+1].clone()
            Ab[:, row, :] = Ab[:, row, :] - factor * Ab[:, col, :]
    
    # Back substitution
    x = torch.zeros(B, n, device=device, dtype=dtype)
    for i in range(n - 1, -1, -1):
        x[:, i] = Ab[:, i, n]
        for j in range(i + 1, n):
            x[:, i] = x[:, i] - Ab[:, i, j] * x[:, j]
    
    return x


def solve_8x8_preordered(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve 8x8 linear system Ax = b using Gaussian elimination.
    Assumes A is pre-ordered so diagonal elements are non-zero.
    ONNX-compatible (no data-dependent control flow).
    
    Args:
        A: [B, 8, 8] coefficient matrices (pre-ordered)
        b: [B, 8] right-hand side vectors (pre-ordered)
    
    Returns:
        x: [B, 8] solution vectors
    """
    B = A.shape[0]
    device = A.device
    dtype = A.dtype
    n = 8
    
    # Create augmented matrix [A | b]
    Ab = torch.cat([A, b.unsqueeze(-1)], dim=-1).clone()  # [B, 8, 9]
    
    # Forward elimination (no pivoting needed - matrix is pre-ordered)
    for col in range(n):
        # Scale pivot row
        pivot = Ab[:, col, col:col+1].clone()
        # Regularize pivot
        pivot = pivot + (pivot.abs() < 1e-10) * 1e-10 * (2 * (pivot >= 0).float() - 1)
        Ab[:, col, :] = Ab[:, col, :] / pivot
        
        # Eliminate below pivot
        for row in range(col + 1, n):
            factor = Ab[:, row, col:col+1].clone()
            Ab[:, row, :] = Ab[:, row, :] - factor * Ab[:, col, :]
    
    # Back substitution
    x = torch.zeros(B, n, device=device, dtype=dtype)
    for i in range(n - 1, -1, -1):
        x[:, i] = Ab[:, i, n]
        for j in range(i + 1, n):
            x[:, i] = x[:, i] - Ab[:, i, j] * x[:, j]
    
    return x


def get_perspective_transform_onnx(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Compute perspective transformation matrix from 4 point correspondences.
    ONNX-compatible version using Gaussian elimination with pre-ordered rows.
    
    Args:
        src: [B, 4, 2] source points  
        dst: [B, 4, 2] destination points
    
    Returns:
        H: [B, 3, 3] perspective transformation matrices
    """
    B = src.shape[0]
    device = src.device
    dtype = src.dtype
    
    # Build 8x8 system with H[2,2] = 1 constraint
    # For point (x,y) -> (u,v):
    # h0*x + h1*y + h2 - h6*x*u - h7*y*u = u
    # h3*x + h4*y + h5 - h6*x*v - h7*y*v = v
    #
    # The default row order with standard corners creates bad pivots.
    # We reorder to: [row2, row6, row4, row0, row3, row7, row5, row1]
    # This ensures non-zero diagonal elements.
    
    A = torch.zeros(B, 8, 8, device=device, dtype=dtype)
    b_vec = torch.zeros(B, 8, device=device, dtype=dtype)
    
    # Original row order
    A_orig = torch.zeros(B, 8, 8, device=device, dtype=dtype)
    b_orig = torch.zeros(B, 8, device=device, dtype=dtype)
    
    for i in range(4):
        x = src[:, i, 0]
        y = src[:, i, 1]
        u = dst[:, i, 0]
        v = dst[:, i, 1]
        
        # Row 2*i: h0*x + h1*y + h2 + 0 + 0 + 0 - h6*xu - h7*yu = u
        A_orig[:, 2*i, 0] = x
        A_orig[:, 2*i, 1] = y
        A_orig[:, 2*i, 2] = 1.0
        A_orig[:, 2*i, 6] = -x * u
        A_orig[:, 2*i, 7] = -y * u
        b_orig[:, 2*i] = u
        
        # Row 2*i+1: 0 + 0 + 0 + h3*x + h4*y + h5 - h6*xv - h7*yv = v
        A_orig[:, 2*i+1, 3] = x
        A_orig[:, 2*i+1, 4] = y
        A_orig[:, 2*i+1, 5] = 1.0
        A_orig[:, 2*i+1, 6] = -x * v
        A_orig[:, 2*i+1, 7] = -y * v
        b_orig[:, 2*i+1] = v
    
    # Reorder rows for better pivoting.
    # For corners [TL(0,0), TR(W,0), BL(0,H), BR(W,H)]:
    # Row 0: A[0,0]=0 (TL x=0)
    # Row 1: A[1,3]=0 (TL x=0) 
    # Row 2: A[2,0]=W (TR x=W) - good pivot for col 0
    # Row 3: A[3,3]=W (TR x=W) - good pivot for col 3
    # Row 4: A[4,1]=H (BL y=H) - good pivot for col 1
    # Row 5: A[5,4]=H (BL y=H) - good pivot for col 4  
    # Row 6: A[6,0]=W, A[6,1]=H (BR) - good
    # Row 7: A[7,3]=W, A[7,4]=H (BR) - good
    #
    # Good order: [2, 4, 0, 6, 3, 5, 1, 7]
    # col 0: row 2 has A[2,0]=W
    # col 1: row 4 has A[4,1]=H  
    # col 2: row 0 has A[0,2]=1
    # col 3: row 6 has A[6,3]=W
    # col 4: row 3 has A[3,4]=0? No...
    # Let me think again...
    
    # Actually easier: just use the pivoting solver and trace with batch=1
    # The pivoting is fixed at trace time for batch=1
    h = solve_8x8_gauss_pivoting(A_orig, b_orig)
    
    # Build 3x3 homography matrix
    H = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    H[:, 0, 0] = h[:, 0]
    H[:, 0, 1] = h[:, 1]
    H[:, 0, 2] = h[:, 2]
    H[:, 1, 0] = h[:, 3]
    H[:, 1, 1] = h[:, 4]
    H[:, 1, 2] = h[:, 5]
    H[:, 2, 0] = h[:, 6]
    H[:, 2, 1] = h[:, 7]
    H[:, 2, 2] = 1.0
    
    return H


def get_flow_from_homography(H: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Generate dense flow field by applying homography to grid of points.
    
    Args:
        H: [B, 3, 3] homography matrices
        height: output height
        width: output width
    
    Returns:
        flow: [B, 2, H, W] dense flow field (absolute coordinates)
    """
    B = H.shape[0]
    device = H.device
    dtype = H.dtype
    
    # Create grid of points
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, height - 1, steps=height, device=device, dtype=dtype),
        torch.linspace(0, width - 1, steps=width, device=device, dtype=dtype),
        indexing='ij'
    )
    
    # Flatten and create homogeneous coords [B, 3, H*W]
    ones = torch.ones(B, 1, height * width, device=device, dtype=dtype)
    grid_x_flat = grid_x.flatten().unsqueeze(0).expand(B, 1, -1)
    grid_y_flat = grid_y.flatten().unsqueeze(0).expand(B, 1, -1)
    points = torch.cat([grid_x_flat, grid_y_flat, ones], dim=1)  # [B, 3, H*W]
    
    # Transform: H @ points
    points_new = torch.bmm(H, points)  # [B, 3, H*W]
    
    # Dehomogenize
    w = points_new[:, 2:3, :] + 1e-8
    points_new = points_new / w
    
    # Reshape to flow [B, 2, H, W]
    flow_x = points_new[:, 0, :].reshape(B, 1, height, width)
    flow_y = points_new[:, 1, :].reshape(B, 1, height, width)
    flow = torch.cat([flow_x, flow_y], dim=1)
    
    return flow


def get_flow_from_four_point_onnx(four_point: torch.Tensor, resize_width: int, 
                                   out_h: int, out_w: int) -> torch.Tensor:
    """
    Convert 4-corner displacements to dense flow using perspective transform.
    ONNX-compatible replacement for get_flow_now_4.
    
    Args:
        four_point: [B, 2, 2, 2] corner displacements at full resolution
        resize_width: original image size (256)
        out_h, out_w: output flow dimensions (usually resize_width // 4)
    
    Returns:
        flow: [B, 2, out_h, out_w] dense flow field
    """
    B = four_point.shape[0]
    device = four_point.device
    dtype = four_point.dtype
    
    # Scale displacement to output resolution
    scale = resize_width / out_w  # typically 4
    four_point_scaled = four_point / scale
    
    # Build source corners (original positions at output resolution)
    src = torch.zeros(B, 4, 2, device=device, dtype=dtype)
    src[:, 0, :] = torch.tensor([0, 0], device=device, dtype=dtype)                      # top-left
    src[:, 1, :] = torch.tensor([out_w - 1, 0], device=device, dtype=dtype)              # top-right  
    src[:, 2, :] = torch.tensor([0, out_h - 1], device=device, dtype=dtype)              # bottom-left
    src[:, 3, :] = torch.tensor([out_w - 1, out_h - 1], device=device, dtype=dtype)      # bottom-right
    
    # Destination corners = source + displacement
    # four_point layout: [B, 2, 2, 2] where dim1=xy, dim2=row, dim3=col
    dst = src.clone()
    dst[:, 0, 0] += four_point_scaled[:, 0, 0, 0]  # top-left x
    dst[:, 0, 1] += four_point_scaled[:, 1, 0, 0]  # top-left y
    dst[:, 1, 0] += four_point_scaled[:, 0, 0, 1]  # top-right x
    dst[:, 1, 1] += four_point_scaled[:, 1, 0, 1]  # top-right y
    dst[:, 2, 0] += four_point_scaled[:, 0, 1, 0]  # bottom-left x
    dst[:, 2, 1] += four_point_scaled[:, 1, 1, 0]  # bottom-left y
    dst[:, 3, 0] += four_point_scaled[:, 0, 1, 1]  # bottom-right x
    dst[:, 3, 1] += four_point_scaled[:, 1, 1, 1]  # bottom-right y
    
    # Compute homography
    H = get_perspective_transform_onnx(src, dst)
    
    # Generate flow field
    flow = get_flow_from_homography(H, out_h, out_w)
    
    return flow


def bilinear_flow_from_corners(four_point: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Compute dense flow field from 4-corner displacements using bilinear interpolation.
    This is fully ONNX-compatible (no matrix inversion or SVD).
    
    Args:
        four_point: [B, 2, 2, 2] corner displacements
                    Layout: [batch, xy, row, col] where
                    (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right
        H: output height
        W: output width
    
    Returns:
        flow: [B, 2, H, W] dense flow field (absolute coordinates)
    """
    B = four_point.shape[0]
    device = four_point.device
    dtype = four_point.dtype
    
    # four_point layout: [B, 2, 2, 2]
    # dim1: x, y coordinates
    # dim2, dim3: corner grid (2x2)
    
    # Extract corner displacements
    # tl = (0,0), tr = (0,1), bl = (1,0), br = (1,1)
    tl = four_point[:, :, 0, 0]  # [B, 2]
    tr = four_point[:, :, 0, 1]  # [B, 2]
    bl = four_point[:, :, 1, 0]  # [B, 2]
    br = four_point[:, :, 1, 1]  # [B, 2]
    
    # Corner positions (original + displacement)
    # Original corners at quarter resolution
    tl_pos = tl + torch.tensor([0.0, 0.0], device=device, dtype=dtype)
    tr_pos = tr + torch.tensor([W - 1.0, 0.0], device=device, dtype=dtype)
    bl_pos = bl + torch.tensor([0.0, H - 1.0], device=device, dtype=dtype)
    br_pos = br + torch.tensor([W - 1.0, H - 1.0], device=device, dtype=dtype)
    
    # Create normalized grid for interpolation [-1, 1]
    # grid_y, grid_x normalized
    grid_y = torch.linspace(0, 1, H, device=device, dtype=dtype)
    grid_x = torch.linspace(0, 1, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing='ij')
    
    # Bilinear interpolation weights
    # For each pixel (x, y) in normalized [0,1]:
    # pos = (1-y) * ((1-x)*tl + x*tr) + y * ((1-x)*bl + x*br)
    
    xx = xx.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    yy = yy.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    
    # Expand corner positions for broadcasting
    tl_pos = tl_pos.view(B, 2, 1, 1)  # [B, 2, 1, 1]
    tr_pos = tr_pos.view(B, 2, 1, 1)
    bl_pos = bl_pos.view(B, 2, 1, 1)
    br_pos = br_pos.view(B, 2, 1, 1)
    
    xx = xx.unsqueeze(1)  # [B, 1, H, W]
    yy = yy.unsqueeze(1)  # [B, 1, H, W]
    
    # Bilinear interpolation
    top = (1 - xx) * tl_pos + xx * tr_pos  # [B, 2, H, W]
    bottom = (1 - xx) * bl_pos + xx * br_pos  # [B, 2, H, W]
    flow = (1 - yy) * top + yy * bottom  # [B, 2, H, W]
    
    return flow


class IHNCoarse(nn.Module):
    """
    Coarse-level IHN network for ONNX export.
    Takes two 256x256 images and outputs 4-corner displacement.
    """
    def __init__(self, resize_width: int = 256, corr_level: int = 4, 
                 corr_radius: int = 4, iters: int = 6, mixed_precision: bool = False):
        super().__init__()
        self.resize_width = resize_width
        self.corr_level = corr_level
        self.corr_radius = corr_radius
        self.iters = iters
        self.mixed_precision = mixed_precision
        self.H_out = resize_width // 4
        self.W_out = resize_width // 4
        
        # Feature extractor
        self.fnet = BasicEncoderQuarter(output_dim=256, norm_fn='instance')
        
        # Update block
        sz = resize_width // 4
        
        # Create args-like object for GMA
        class Args:
            def __init__(self, corr_level):
                self.weight = False
                self.mixed_precision = False
                self.corr_level = corr_level
        
        self.args = Args(corr_level)
        self.args.mixed_precision = mixed_precision
        self.update_block = GMA(self.args, sz)
        
        # ImageNet normalization
        self.register_buffer('imagenet_mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', 
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def get_flow_from_four_point(self, four_point: torch.Tensor) -> torch.Tensor:
        """Convert 4-point displacement to dense flow field using PROPER perspective transform."""
        return get_flow_from_four_point_onnx(four_point, self.resize_width, self.H_out, self.W_out)
    
    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image1: [1, 3, 256, 256] satellite image (already resized)
            image2: [1, 3, 256, 256] query image
        
        Returns:
            four_pred: [1, 2, 2, 2] 4-corner displacement prediction
        """
        # Normalize
        image1 = (image1 - self.imagenet_mean) / self.imagenet_std
        image2 = (image2 - self.imagenet_mean) / self.imagenet_std
        
        # Extract features
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)
        
        sz = fmap1.shape
        B = sz[0]
        
        # Initialize correlation
        corr_fn = CorrBlock(fmap1.float(), fmap2.float(), 
                           num_levels=self.corr_level, radius=self.corr_radius)
        
        # Initialize coordinates
        coords0 = coords_grid(B, self.H_out, self.W_out).to(image1.device)
        coords1 = coords_grid(B, self.H_out, self.W_out).to(image1.device)
        
        # Initialize displacement
        four_point_disp = torch.zeros((B, 2, 2, 2), device=image1.device, dtype=image1.dtype)
        
        # Iterative refinement
        for _ in range(self.iters):
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            
            delta_four_point = self.update_block(corr, flow)
            four_point_disp = four_point_disp + delta_four_point
            
            coords1 = self.get_flow_from_four_point(four_point_disp)
        
        return four_point_disp


class IHNFine(nn.Module):
    """
    Fine-level IHN network for ONNX export (two-stage refinement).
    Takes cropped satellite and query images, outputs refined displacement.
    """
    def __init__(self, resize_width: int = 256, corr_level: int = 2,
                 corr_radius: int = 4, iters: int = 6, mixed_precision: bool = False):
        super().__init__()
        self.resize_width = resize_width
        self.corr_level = corr_level
        self.corr_radius = corr_radius
        self.iters = iters
        self.mixed_precision = mixed_precision
        self.H_out = resize_width // 4
        self.W_out = resize_width // 4
        
        # Feature extractor
        self.fnet = BasicEncoderQuarter(output_dim=256, norm_fn='instance')
        
        # Update block
        sz = resize_width // 4
        
        class Args:
            def __init__(self, corr_level):
                self.weight = False
                self.mixed_precision = False
                self.corr_level = corr_level
        
        self.args = Args(corr_level)
        self.args.mixed_precision = mixed_precision
        self.update_block = GMA(self.args, sz)
        
        # ImageNet normalization
        self.register_buffer('imagenet_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def get_flow_from_four_point(self, four_point: torch.Tensor) -> torch.Tensor:
        """Convert 4-point displacement to dense flow field using PROPER perspective transform."""
        return get_flow_from_four_point_onnx(four_point, self.resize_width, self.H_out, self.W_out)
    
    def forward(self, image1_crop: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for fine stage.
        
        Args:
            image1_crop: [1, 3, 256, 256] cropped satellite image
            image2: [1, 3, 256, 256] query image
        
        Returns:
            four_pred: [1, 2, 2, 2] fine 4-corner displacement
        """
        image1_crop = (image1_crop - self.imagenet_mean) / self.imagenet_std
        image2 = (image2 - self.imagenet_mean) / self.imagenet_std
        
        fmap1 = self.fnet(image1_crop)
        fmap2 = self.fnet(image2)
        
        sz = fmap1.shape
        B = sz[0]
        
        corr_fn = CorrBlock(fmap1.float(), fmap2.float(),
                           num_levels=self.corr_level, radius=self.corr_radius)
        
        coords0 = coords_grid(B, self.H_out, self.W_out).to(image1_crop.device)
        coords1 = coords_grid(B, self.H_out, self.W_out).to(image1_crop.device)
        
        four_point_disp = torch.zeros((B, 2, 2, 2), device=image1_crop.device, dtype=image1_crop.dtype)
        
        for _ in range(self.iters):
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            
            delta_four_point = self.update_block(corr, flow)
            four_point_disp = four_point_disp + delta_four_point
            
            coords1 = self.get_flow_from_four_point(four_point_disp)
        
        return four_point_disp


def load_weights_from_pth(model: nn.Module, checkpoint: dict, key: str = 'netG'):
    """Load weights from checkpoint, handling 'module.' prefix."""
    state_dict = checkpoint[key]
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    
    # Map old keys to new model keys
    mapped_state = {}
    
    for k, v in new_state_dict.items():
        # Map fnet1 -> fnet
        if k.startswith('fnet1.'):
            new_k = k.replace('fnet1.', 'fnet.')
            mapped_state[new_k] = v
        # Map update_block_4 -> update_block
        elif k.startswith('update_block_4.'):
            new_k = k.replace('update_block_4.', 'update_block.')
            mapped_state[new_k] = v
        else:
            mapped_state[k] = v
    
    # Load only matching keys
    missing, unexpected = model.load_state_dict(mapped_state, strict=False)
    if missing:
        # Filter out expected missing keys (buffers we create ourselves)
        expected_missing = {'imagenet_mean', 'imagenet_std'}
        actual_missing = [k for k in missing if k not in expected_missing]
        if actual_missing:
            print(f"  Missing keys: {actual_missing[:5]}..." if len(actual_missing) > 5 else f"  Missing keys: {actual_missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Unexpected keys: {unexpected}")
    
    return model


def export_onnx(model: nn.Module, output_path: str, input_names: list,
                output_names: list, resize_width: int = 256):
    """Export model to ONNX format."""
    model.eval()
    
    # Use CPU for export to avoid Jetson CUDA issues
    model = model.cpu()
    
    # Create dummy inputs on CPU
    dummy_img1 = torch.randn(1, 3, resize_width, resize_width)
    dummy_img2 = torch.randn(1, 3, resize_width, resize_width)
    
    print(f"Exporting to {output_path}...")
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_img1, dummy_img2),
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={
                input_names[0]: {0: 'batch'},
                input_names[1]: {0: 'batch'},
                output_names[0]: {0: 'batch'}
            }
        )
    
    print(f"✓ Exported: {output_path}")
    
    # Verify with onnx
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verified successfully")
    except ImportError:
        print("⚠ onnx package not installed, skipping verification")
    except Exception as e:
        print(f"⚠ ONNX verification warning: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert STHN .pth to ONNX")
    parser.add_argument('--pth', type=str, required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for ONNX files')
    parser.add_argument('--two_stages', action='store_true',
                        help='Export both coarse and fine models (two-stage)')
    parser.add_argument('--resize_width', type=int, default=256,
                        help='Input image size (default: 256)')
    parser.add_argument('--corr_level', type=int, default=4,
                        help='Correlation pyramid levels (default: 4)')
    parser.add_argument('--iters', type=int, default=6,
                        help='Number of refinement iterations (default: 6)')
    parser.add_argument('--database_size', type=int, default=1536,
                        help='Original database/satellite image size (default: 1536)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for loading weights (default: cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load checkpoint on CPU to avoid issues
    print(f"Loading checkpoint: {args.pth}")
    checkpoint = torch.load(args.pth, map_location='cpu')
    
    print(f"Using device: cpu (for stable ONNX export)")
    
    # Export coarse model
    print("\n=== Exporting Coarse Model ===")
    coarse_model = IHNCoarse(
        resize_width=args.resize_width,
        corr_level=args.corr_level,
        iters=args.iters
    )
    
    coarse_model = load_weights_from_pth(coarse_model, checkpoint, 'netG')
    coarse_model.eval()
    
    coarse_onnx_path = os.path.join(args.out_dir, 'sthn_coarse.onnx')
    export_onnx(
        coarse_model, 
        coarse_onnx_path,
        input_names=['image1', 'image2'],
        output_names=['four_pred'],
        resize_width=args.resize_width
    )
    
    # Export fine model if two-stage
    if args.two_stages:
        print("\n=== Exporting Fine Model ===")
        fine_model = IHNFine(
            resize_width=args.resize_width,
            corr_level=2,  # Fine stage uses corr_level=2
            iters=args.iters
        )
        
        # Try to load fine weights
        if 'netG_fine' in checkpoint:
            fine_model = load_weights_from_pth(fine_model, checkpoint, 'netG_fine')
        else:
            print("⚠ No 'netG_fine' in checkpoint, using 'netG' weights for fine model")
            fine_model = load_weights_from_pth(fine_model, checkpoint, 'netG')
        
        fine_model.eval()
        
        fine_onnx_path = os.path.join(args.out_dir, 'sthn_fine.onnx')
        export_onnx(
            fine_model,
            fine_onnx_path,
            input_names=['image1_crop', 'image2'],
            output_names=['four_pred_fine'],
            resize_width=args.resize_width
        )
    
    print("\n=== Export Complete ===")
    print(f"Output directory: {args.out_dir}")
    print("\nNext steps:")
    print("1. Build TensorRT engine with trtexec:")
    print(f"   trtexec --onnx={coarse_onnx_path} --saveEngine={args.out_dir}/sthn_coarse_fp16.engine --fp16")
    if args.two_stages:
        print(f"   trtexec --onnx={fine_onnx_path} --saveEngine={args.out_dir}/sthn_fine_fp16.engine --fp16")


if __name__ == '__main__':
    main()
