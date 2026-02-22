import numpy as np
try:
    import torch
except ImportError as e:
    raise ImportError(
        "Failed to import PyTorch. This script uses torch for image preprocessing and CUDA buffers. "
        "On Jetson, a common fix is: sudo apt-get update && sudo apt-get install -y libcufile-12-6 && sudo ldconfig"
    ) from e
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
import time

from PIL import Image

import parser
from model.network_trt import STHNTRT


base_transform = transforms.Compose(
    [
        transforms.Resize([256, 256]),
    ]
)

query_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)


def test(args):
    # Reuse existing argument names: for TRT we interpret eval_model as the coarse .engine path
    # and eval_model_fine (if provided) as the fine .engine path.
    if args.eval_model is None:
        raise ValueError("For TensorRT inference, pass --eval_model pointing to the coarse .engine")
    if args.two_stages and args.eval_model_fine is None:
        raise ValueError("two_stages requires --eval_model_fine pointing to the fine .engine")

    model = STHNTRT(
        args,
        engine_coarse=args.eval_model,
        engine_fine=args.eval_model_fine,
    )

    model.setup()

    folder_name = "maps_results/farm"
    all_corners = []
    times = []

    N = 108 # number of samples
    T = 31 # tiles in each x dir
    TH = 9
    SAT = 12
    for i in range(N):
        try:
            # مسیر تصاویر با شماره i
            
            # img1_path = f"js_datasets/qomFly2-400m/satellite/tile_{i+1011}.png"
            # img2_path = f"js_datasets/qomFly2-400m/thermal/frame_{i*3 +3096}.png"

            # --- XYZ TILES ---
            # z = 19 
            # x = 348420 + (i // T)
            # y = 204759 + (i % T)

            # img1_path = fr"D:\RPL\Tiles\Mashhad\satellite\{z}\{x}\{y}.png"
            # img2_path = fr"D:\RPL\Tiles\Mashhad\thermal\{z}_{x}_{y}.png"

            # --- GRID CROP ---

            img1_path = fr"js_datasets/Dehat/satellite/{i // TH + 1}.tif"
            img2_path = fr"js_datasets/Dehat/thermal/{i // TH + 1}_{i % TH + 1}.tif"

            # خواندن تصاویر
            img1 = F.to_tensor(Image.open(img1_path).convert("RGB")).unsqueeze(0)
            img2 = (base_transform(query_transform(Image.open(img2_path)))).unsqueeze(0)

            start_time = time.time()
            # اعمال مدل
            with torch.no_grad():
                model.set_input(img1, img2)
                model.forward()
                four_pred = model.four_pred

            # print('!!!four_pred')
            # print(four_pred)

    
            # آماده‌سازی نقاط مرجع
            four_point_org_single = torch.zeros((1, 2, 2, 2))
            four_point_org_single[:, :, 0, 0] = torch.Tensor([0, 0])
            four_point_org_single[:, :, 0, 1] = torch.Tensor([args.resize_width - 1, 0])
            four_point_org_single[:, :, 1, 0] = torch.Tensor([0, args.resize_width - 1])
            four_point_org_single[:, :, 1, 1] = torch.Tensor([args.resize_width - 1, args.resize_width - 1])
            
            # پردازش خروجی
            four_point_1 = four_pred.cpu().detach() + four_point_org_single
            four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous()
            four_point_1_mul6 = four_point_1 * 6
            center = four_point_1_mul6.mean(dim=1)  # شکل (1,2)
            center = tuple(center[0].tolist())
            # print(center)
            # print(four_point_1_mul6)
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
        
            # استخراج نقاط پیش‌بینی‌شده (4 گوشه)
            points = four_point_1_mul6.squeeze(0).tolist()  # 4 × 2 لیست
            flat_points = [coord for point in points for coord in point]  # تبدیل به لیست 8 تایی
    
            all_corners.append([i] + flat_points + [img1_path, img2_path])  # اضافه کردن شماره عکس + نقاط
    
            print(f"✅ Done for image {i + 1}")
    
        except Exception as e:
            print(f"❌ Error in image {i}: {e}")
            
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n📊 Average processing time per image: {avg_time:.4f} sec")

    # ذخیره در فایل Excel
    columns = ["image_index", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "sat", "th"]
    df = pd.DataFrame(all_corners, columns=columns)
    df.to_excel(f"js_excels/predicted-dehat-noSVD.xlsx", index=False)
    print("📁 Saved all corner points to four_point_1_mul6.xlsx")


if __name__ == "__main__":
    args = parser.parse_arguments()
    test(args)
