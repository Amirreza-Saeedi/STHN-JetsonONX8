import numpy as np
import os
import torch
import argparse
import sys
from model.js_network_noKornia_const import STHN
from utils import save_overlap_img, save_img, setup_seed, save_overlap_bbox_img
import datasets_4cor_img as datasets
import scipy.io as io
import torchvision
import numpy as np
import time
from tqdm import tqdm
import cv2
# import kornia.geometry.transform as tgm
import matplotlib.pyplot as plt
from plot_hist import plot_hist_helper
# import torch.nn.functional as F
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
import time
import parser
from datetime import datetime
from os.path import join
import commons
import logging
import wandb
from PIL import Image 
base_transform = transforms.Compose(
            [
                transforms.Resize([256,256]),
            ]
        )
query_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ]
        )

def load_sthn(args):
    if not args.identity:
        model = STHN(args)
        if not args.train_ue_method == "train_only_ue_raw_input":
            model_med = torch.load(args.eval_model, map_location=args.device)
            for key in list(model_med['netG'].keys()):
                model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
            for key in list(model_med['netG'].keys()):
                if key.startswith('module'):
                    del model_med['netG'][key]
            model.netG.load_state_dict(model_med['netG'], strict=False)

        print('args.eval_model_ue', args.eval_model_ue)
        
        if args.eval_model_fine is None:
            model_med = torch.load(args.eval_model, map_location=args.device)
            for key in list(model_med['netG_fine'].keys()):
                model_med['netG_fine'][key.replace('module.','')] = model_med['netG_fine'][key]
            for key in list(model_med['netG_fine'].keys()):
                if key.startswith('module'):
                    del model_med['netG_fine'][key]
            model.netG_fine.load_state_dict(model_med['netG_fine'])
        else:
            model_med = torch.load(args.eval_model_fine, map_location=args.device)
            for key in list(model_med['netG'].keys()):
                model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
            for key in list(model_med['netG'].keys()):
                if key.startswith('module'):
                    del model_med['netG'][key]
            model.netG_fine.load_state_dict(model_med['netG'], strict=False)
        
        model.setup() 
        model.netG.eval()
        model.netG_fine.eval()

        return model
    

import torch.onnx

# CRITICAL: Force legacy exporter
os.environ['TORCH_ONNX_USE_NEW_EXPORTER'] = '0'

def export_to_onnx(model, args, onnx_path="sthn.onnx"):
    # Force CPU
    model = model.cpu()
    model.eval()
    
    # Disable CUDA optimizations
    torch.backends.cudnn.enabled = False
    # torch.cuda.set_device('cpu')  # This doesn't work, but try:
    
    # Actually move all tensors to CPU
    dummy1 = torch.randn(1, 3, 256, 256)
    dummy2 = torch.randn(1, 3, 256, 256)
    
    # Use older API
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy1, dummy2),
            "model.onnx",
            opset_version=11,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        )



def export_via_jit(model, args, onnx_path="sthn.onnx"):
    # Move to CPU
    model = model.cpu()
    model.eval()
    
    # Create dummy inputs
    dummy_img1 = torch.randn(1, 3, args.resize_width, args.resize_width)
    dummy_img2 = torch.randn(1, 3, args.resize_width, args.resize_width)
    
    # Script the model
    with torch.no_grad():
        try:
            # Try tracing first (more compatible)
            traced_model = torch.jit.trace(model, (dummy_img1, dummy_img2))
        except:
            # Fall back to scripting
            traced_model = torch.jit.script(model)
        
        # Save TorchScript
        traced_model.save("sthn_scripted.pt")
        
        # Convert to ONNX from TorchScript
        torch.onnx.export(
            traced_model,
            (dummy_img1, dummy_img2),
            onnx_path,
            input_names=['image1', 'image2'],
            output_names=['output'],
            opset_version=11,
            export_params=True
        )
    
    print(f"✅ Exported to {onnx_path}")


def test(args, wandb_log):

    model = load_sthn(args).cpu()
    # model = load_sthn(args)

    print(10 * '=')
    print()

    # export_to_onnx(model, args)
    # export_via_jit(model, args)

    folder_name = "maps_results/farm"
    all_corners = []
    times = []

    N = 108 # number of samples
    N = 10 # number of samples
    T = 31 # tiles in each x dir
    TH = 9
    SAT = 12
    for i in range(N):
        try:
            # --- GRID CROP ---
            img1_path = fr"D:/RPL/Tiles/Dehat/satellite/{i // TH + 1}.tif"
            img2_path = fr"D:/RPL/Tiles/Dehat/thermal/{i // TH + 1}_{i % TH + 1}.tif"

            # خواندن تصاویر
            img1 = F.to_tensor(Image.open(img1_path).convert("RGB")).unsqueeze(0)
            img2 = (base_transform(query_transform(Image.open(img2_path)))).unsqueeze(0)
            start_time = time.time()
            # اعمال مدل
            with torch.no_grad():
                model.set_input(img1, img2)
                model.forward(img1, img2)
                four_pred = model.four_pred
    
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
            print(flat_points)
            all_corners.append([i] + flat_points + [img1_path, img2_path])  # اضافه کردن شماره عکس + نقاط
    
            print(f"✅ Done for image {i + 1}   {elapsed:.3f} sec")
    
        except Exception as e:
            print(f"❌ Error in image {i}: {e}")
            
    if times:
        avg_time = sum(times[1:]) / len(times[1:])
        print(f"\n📊 Average processing time per image: {avg_time:.3f} sec, {1/avg_time:.2f} fps")

    # ذخیره در فایل Excel
    columns = ["image_index", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "sat", "th"]
    df = pd.DataFrame(all_corners, columns=columns)
    df.to_excel(f"js_excels/new.xlsx", index=False)
    print("📁 Saved all corner points to four_point_1_mul6.xlsx")

class Args:
    def __init__(self):
        self.dataset_name = 'NoName'
        self.resize_width = 256
        self.database_size = 1536
        self.lev0 = True
        self.mixed_precision = False
        self.arch = "IHN"
        self.iters_lev0 = 6
        self.iters_lev1 = 6
        self.corr_level = 4
        self.fine_padding = 32
        self.detach = False
        self.augment_two_stages = 0
        self.augment_type = 'center'
        self.identity = False
        self.device = torch.device('cuda:0')
        self.device = torch.device('cpu')
        self.two_stages = True
        self.use_ue = False
        self.train_ue_method = 'train_end_to_end'
        self.eval_model_fine = None
        self.eval_model_ue = None
        self.val_positive_dist_threshold = 512
        self.test = True
        self.eval_model = 'js_models/1536_two_stages/STHN.pth'

if __name__ == '__main__':
    # Use command args or in-code args
    args = Args()
    # args = parser.parse_arguments()

    # start_time = datetime.now()
    # if args.identity:
    #     pass
    # else:
    #     args.save_dir = join(
    #     "test",
    #     args.save_dir,
    #     args.eval_model.split("/")[-2] if args.eval_model is not None else args.eval_model_ue.split("/")[-2],
    #     f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
    #     )
    #     commons.setup_logging(args.save_dir, console='info')
    # setup_seed(0)
    # logging.debug(args)
    wandb_log = True
    # if wandb_log:
    #     wandb.init(project="STHN-eval", entity="xjh19971", config=vars(args))
    test(args, wandb_log)
