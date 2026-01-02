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

    for i in range(50):
        try:
            img1_path = f"js_datasets/qomFly2/satellite/tile_{i+165}.png"
            img2_path = f"js_datasets/qomFly2/thermal/frame_{i}.png"

            img1 = F.to_tensor(Image.open(img1_path).convert("RGB")).unsqueeze(0)
            img2 = (base_transform(query_transform(Image.open(img2_path)))).unsqueeze(0)

            start_time = time.time()
            with torch.no_grad():
                model.set_input(img1, img2)
                model.forward()
                four_pred = model.four_pred

            four_point_org_single = torch.zeros((1, 2, 2, 2))
            four_point_org_single[:, :, 0, 0] = torch.Tensor([0, 0])
            four_point_org_single[:, :, 0, 1] = torch.Tensor([args.resize_width - 1, 0])
            four_point_org_single[:, :, 1, 0] = torch.Tensor([0, args.resize_width - 1])
            four_point_org_single[:, :, 1, 1] = torch.Tensor([args.resize_width - 1, args.resize_width - 1])

            four_point_1 = four_pred.detach().float().cpu() + four_point_org_single
            four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous()
            four_point_1_mul6 = four_point_1 * 6

            center = four_point_1_mul6.mean(dim=1)
            center = tuple(center[0].tolist())
            # print(center)
            # print(four_point_1_mul6)

            elapsed = time.time() - start_time
            times.append(elapsed)

            points = four_point_1_mul6.squeeze(0).tolist()
            flat_points = [coord for point in points for coord in point]
            all_corners.append([i] + flat_points)

            print(f"✅ Done for image {i}")

        except Exception as e:
            print(f"❌ Error in image {i}: {e}")

    if times:
        avg_time = sum(times) / len(times)
        print(f"\n📊 Average processing time per image: {avg_time:.4f} sec")

    columns = ["image_index", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    df = pd.DataFrame(all_corners, columns=columns)
    df.to_excel("js_excels/predicted_trt.xlsx", index=False)
    print("📁 Saved all corner points to js_excels/predicted_trt.xlsx")


if __name__ == "__main__":
    args = parser.parse_arguments()
    test(args)
