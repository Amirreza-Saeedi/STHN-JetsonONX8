"""
Quick test to isolate the get_perspective_transform issue
Run this on both devices and compare outputs
"""

import torch
import sys

# Import your replacement function
sys.path.append('model')  # adjust path as needed
from local_pipeline.model.js_kornia_replacement import get_perspective_transform_torch

def test_with_real_data():
    """Test with data similar to what your model produces"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("QUICK FOCUSED TEST - get_perspective_transform_torch")
    print("="*80)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    # Set deterministic behavior
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Test Case 1: Small displacement (typical early iteration)
    print("\n--- Test 1: Small displacement ---")
    four_point_disp = torch.tensor([[[[0.5, -1.0],
                                       [0.3, -0.8]],
                                      [[0.7, 0.9],
                                       [-0.4, -0.6]]]], device=device, dtype=torch.float32)
    
    # Simulate your model's behavior
    sz = (1, 256, 64, 64)  # typical feature map size
    
    # Build four_point_org as your model does
    four_point = four_point_disp / 4
    four_point_org = torch.zeros((2, 2, 2), device=device)
    four_point_org[:, 0, 0] = torch.tensor([0, 0], device=device)
    four_point_org[:, 0, 1] = torch.tensor([sz[3]-1, 0], device=device)
    four_point_org[:, 1, 0] = torch.tensor([0, sz[2]-1], device=device)
    four_point_org[:, 1, 1] = torch.tensor([sz[3]-1, sz[2]-1], device=device)
    
    four_point_org = four_point_org.unsqueeze(0).repeat(sz[0], 1, 1, 1)
    four_point_new = four_point_org + four_point
    
    four_point_org_flat = four_point_org.flatten(2).permute(0, 2, 1).contiguous()
    four_point_new_flat = four_point_new.flatten(2).permute(0, 2, 1).contiguous()
    
    print(f"four_point_org_flat:\n{four_point_org_flat}")
    print(f"four_point_new_flat:\n{four_point_new_flat}")
    
    H = get_perspective_transform_torch(four_point_org_flat, four_point_new_flat)
    
    print(f"\nH matrix:\n{H}")
    print(f"\nH statistics:")
    print(f"  Min: {H.min().item():.10f}")
    print(f"  Max: {H.max().item():.10f}")
    print(f"  Mean: {H.mean().item():.10f}")
    print(f"  Std: {H.std().item():.10f}")
    print(f"  H[0,0,0]: {H[0,0,0].item():.10f}")
    print(f"  H[0,1,1]: {H[0,1,1].item():.10f}")
    print(f"  H[0,2,2]: {H[0,2,2].item():.10f}")
    
    # Verify transformation
    src_test = four_point_org_flat[0].T  # [2, 4]
    src_homo = torch.cat([src_test, torch.ones((1, 4), device=device)], dim=0)  # [3, 4]
    dst_computed = H[0] @ src_homo
    dst_computed = dst_computed[:2] / dst_computed[2:3]
    error = torch.abs(dst_computed.T - four_point_new_flat[0]).max().item()
    
    print(f"\nTransformation error: {error:.10e}")
    print(f"✓ PASS" if error < 1e-3 else f"✗ FAIL")
    
    # Test Case 2: Larger displacement (typical later iteration)
    print("\n--- Test 2: Larger displacement ---")
    four_point_disp = torch.tensor([[[[3.6615, -167.2107],
                                       [3.6554, -167.1031]],
                                      [[107.6565, 107.8281],
                                       [-62.8925, -63.0094]]]], device=device, dtype=torch.float32)
    
    four_point = four_point_disp / 4
    four_point_new = four_point_org + four_point
    four_point_new_flat = four_point_new.flatten(2).permute(0, 2, 1).contiguous()
    
    print(f"four_point_disp (ASUS expected output):\n{four_point_disp}")
    print(f"four_point_new_flat:\n{four_point_new_flat}")
    
    H = get_perspective_transform_torch(four_point_org_flat, four_point_new_flat)
    
    print(f"\nH matrix:\n{H}")
    print(f"\nH statistics:")
    print(f"  Min: {H.min().item():.10f}")
    print(f"  Max: {H.max().item():.10f}")
    print(f"  Mean: {H.mean().item():.10f}")
    print(f"  H[0,0,0]: {H[0,0,0].item():.10f}")
    print(f"  H[0,2,2]: {H[0,2,2].item():.10f}")
    
    # Verify transformation
    dst_computed = H[0] @ src_homo
    dst_computed = dst_computed[:2] / dst_computed[2:3]
    error = torch.abs(dst_computed.T - four_point_new_flat[0]).max().item()
    
    print(f"\nTransformation error: {error:.10e}")
    print(f"✓ PASS" if error < 1e-3 else f"✗ FAIL")
    
    # Test Case 3: Determinism check
    print("\n--- Test 3: Determinism check (run 10 times) ---")
    H_list = []
    for i in range(10):
        H_test = get_perspective_transform_torch(four_point_org_flat, four_point_new_flat)
        H_list.append(H_test)
    
    max_diff = 0
    for i in range(1, 10):
        diff = torch.abs(H_list[0] - H_list[i]).max().item()
        max_diff = max(max_diff, diff)
    
    print(f"Max difference across 10 runs: {max_diff:.10e}")
    print(f"✓ PASS - Deterministic" if max_diff < 1e-6 else f"✗ FAIL - Non-deterministic!")
    
    # Save results
    results = {
        'device': str(device),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'H_test1': H_list[0][0].cpu().numpy().tolist(),
        'max_diff': max_diff,
        'transformation_error': error,
    }
    
    import json
    with open('quick_test_results-norm.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to quick_test_results-norm.json")
    print("\nRun this on both devices and compare the JSON files!")


if __name__ == "__main__":
    test_with_real_data()
