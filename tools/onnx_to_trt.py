#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT engine.

Usage:
  # One-stage:
  python3 -m tools.onnx_to_trt \
      --onnx trt/one_stage_v2/sthn_coarse.onnx \
      --engine trt/one_stage_v2/sthn_coarse.engine \
      --fp16

  # Two-stage:
  python3 -m tools.onnx_to_trt \
      --onnx trt/two_stages/sthn_coarse.onnx \
      --engine trt/two_stages/sthn_coarse.engine \
      --fp16
  python3 -m tools.onnx_to_trt \
      --onnx trt/two_stages/sthn_fine.onnx \
      --engine trt/two_stages/sthn_fine.engine \
      --fp16
"""

import argparse
import os
import subprocess
import sys


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True, 
                 trtexec_path: str = "/usr/src/tensorrt/bin/trtexec"):
    """
    Build TensorRT engine from ONNX model using trtexec.
    
    Args:
        onnx_path: Path to input ONNX file
        engine_path: Path to output .engine file
        fp16: Use FP16 precision
        trtexec_path: Path to trtexec binary
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
    
    # Build command
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--minShapes=image1:1x3x256x256,image2:1x3x256x256",
        "--optShapes=image1:1x3x256x256,image2:1x3x256x256",
        "--maxShapes=image1:1x3x256x256,image2:1x3x256x256",
    ]
    
    if fp16:
        cmd.append("--fp16")
    
    print("=" * 60)
    print("Building TensorRT Engine")
    print("=" * 60)
    print(f"ONNX:   {onnx_path}")
    print(f"Engine: {engine_path}")
    print(f"FP16:   {fp16}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run trtexec
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"trtexec failed with return code {result.returncode}")
    
    if os.path.exists(engine_path):
        size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"\n✅ Engine built successfully: {engine_path} ({size_mb:.1f} MB)")
    else:
        raise RuntimeError("Engine file was not created")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")
    parser.add_argument('--onnx', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--engine', type=str, required=True,
                        help='Path to output TensorRT engine')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 precision')
    parser.add_argument('--trtexec', type=str, default='/usr/src/tensorrt/bin/trtexec',
                        help='Path to trtexec binary')
    return parser.parse_args()


def main():
    args = parse_args()
    build_engine(args.onnx, args.engine, args.fp16, args.trtexec)


if __name__ == '__main__':
    main()
