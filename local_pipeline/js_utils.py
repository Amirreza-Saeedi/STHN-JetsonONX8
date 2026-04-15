import torch


def get_module_stats(module, name=""):
    """
    برمی‌گرداند:
      - تعداد پارامترها
      - حجم پارامترها به بایت
      - تعداد پارامترهای dtype متفاوت
      - حجم کل buffers (BatchNorm و غیره)
    """
    # وزن‌ها
    params = list(module.parameters())
    num_params = sum(p.numel() for p in params if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in params if p.requires_grad)

    # dtype ها
    dtype_counts = {}
    for p in params:
        dtype_counts[p.dtype] = dtype_counts.get(p.dtype, 0) + 1

    # buffers
    buffers = list(module.buffers())
    buf_bytes = sum(b.numel() * b.element_size() for b in buffers)

    print(f"vvv Module Weights Stats: {name}")
    print(f"تعداد وزن‌ها: {num_params:,}")
    print(f"حجم وزن‌ها (بایت): {param_bytes:,}")
    print(f"نوع‌های dtype: {dtype_counts}")
    print(f"حجم buffers (بایت): {buf_bytes:,}")
    print(f"حجم کل (بایت): {(param_bytes + buf_bytes):,}")
    print()

  
def print_gpu_mem(msg: str, device=None):
      print('GPU Memory Stats')
      print('vvv mem_allocated', msg + ':', torch.cuda.memory_allocated(device)/1e6, 'MB')
      print('vvv mem_reserved ', msg + ':', torch.cuda.memory_reserved(device)/1e6, 'MB')
      print
      # print('vvv mem_summary  ', msg + ':', torch.cuda.memory_summary(device))
      # print('vvv mem_usage    ', msg + ':', torch.cuda.memory_usage(device))