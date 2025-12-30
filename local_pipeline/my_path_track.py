import numpy as np
import os
import torch
import argparse
from model.network import STHN
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

def crop_center(img, center, w, h):
    """
    img : تصویر OpenCV (BGR) قبلا خوانده شده
    center : (x, y) مرکز برش
    w, h : ابعاد برش
    خروجی: Tensor به شکل (1, 3, h, w) با مقادیر بین 0 و 1
    """
    H, W = img.shape[:2]
    x, y = center

    # مختصات گوشه‌ها
    x1 = max(x - w // 2, 0)
    y1 = max(y - h // 2, 0)
    x2 = min(x + w // 2, W)
    y2 = min(y + h // 2, H)

    # برش
    crop = img[y1:y2, x1:x2]

    # پدینگ در صورت نیاز
    if crop.shape[0] != h or crop.shape[1] != w:
        crop = cv2.copyMakeBorder(
            crop,
            top=0, bottom=h - crop.shape[0],
            left=0, right=w - crop.shape[1],
            borderType=cv2.BORDER_CONSTANT,
            value=(0,0,0)
        )

    # BGR → RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # NumPy → PIL → Tensor
    crop_pil = Image.fromarray(crop_rgb)
    crop_tensor = F.to_tensor(crop_pil).unsqueeze(0)

    return crop_tensor

def test(args, wandb_log):
    if not args.identity:
        model = STHN(args)
        if not args.train_ue_method == "train_only_ue_raw_input":
            model_med = torch.load(args.eval_model, map_location='cuda:0')
            for key in list(model_med['netG'].keys()):
                model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
            for key in list(model_med['netG'].keys()):
                if key.startswith('module'):
                    del model_med['netG'][key]
            model.netG.load_state_dict(model_med['netG'], strict=False)
        if args.use_ue:
            if args.eval_model_ue is not None:
                model_med = torch.load(args.eval_model_ue, map_location='cuda:0')
            for key in list(model_med['netD'].keys()):
                model_med['netD'][key.replace('module.','')] = model_med['netD'][key]
            for key in list(model_med['netD'].keys()):
                if key.startswith('module'):
                    del model_med['netD'][key]
            model.netD.load_state_dict(model_med['netD'])
        if args.two_stages:
            if args.eval_model_fine is None:
                model_med = torch.load(args.eval_model, map_location='cuda:0')
                for key in list(model_med['netG_fine'].keys()):
                    model_med['netG_fine'][key.replace('module.','')] = model_med['netG_fine'][key]
                for key in list(model_med['netG_fine'].keys()):
                    if key.startswith('module'):
                        del model_med['netG_fine'][key]
                model.netG_fine.load_state_dict(model_med['netG_fine'])
            else:
                model_med = torch.load(args.eval_model_fine, map_location='cuda:0')
                for key in list(model_med['netG'].keys()):
                    model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
                for key in list(model_med['netG'].keys()):
                    if key.startswith('module'):
                        del model_med['netG'][key]
                model.netG_fine.load_state_dict(model_med['netG'], strict=False)
        
        model.setup() 
        model.netG.eval()
        if args.use_ue:
            model.netD.eval()
        if args.two_stages:
            model.netG_fine.eval()
  
    all_corners = []
    times = []
    
    map_path = "/content/drive/MyDrive/6_20cpp.tif"
    map = cv2.imread(map_path, cv2.IMREAD_COLOR)
    last_location =(26360,13824)

    
    predict_locations = []
    for i in range(97):
        try:
        
            # مسیر تصاویر با شماره i
            # img1_path = f"/content/drive/MyDrive/{folder_name}/big_patches/big_picture{i}.jpg"
            img1=crop_center(map,last_location, 600*5, 600*5)
            img2_path = f"/content/drive/MyDrive/infinity_patches_thermal/patch_{i:04d}.png"
    
            # خواندن تصاویر
            img2 = (base_transform(query_transform(Image.open(img2_path)))).unsqueeze(0)
            
            # اعمال مدل
            with torch.no_grad():
                model.set_input(img1, img2)
                model.forward()
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
            four_point_1_mul6 = four_point_1 * (600/256)
            
            center = four_point_1_mul6.mean(dim=1)  # شکل (1,2)
            center = tuple(center[0].tolist())
            x_center ,y_center = center
            last_location = (
                int(last_location[0] + x_center*5 - 1500),
                int(last_location[1] + y_center*5 - 1500)
            )
           
        
            # استخراج نقاط پیش‌بینی‌شده (4 گوشه)
            points = four_point_1_mul6.squeeze(0).tolist()  # 4 × 2 لیست
            flat_points = [coord for point in points for coord in point]  # تبدیل به لیست 8 تایی
    
            all_corners.append([i] + flat_points)  # اضافه کردن شماره عکس + نقاط
    
            print(f"✅ Done for image {i}")
            print(last_location)
        except Exception as e:
            print(f"❌ Error in image {i}: {e}")
            
        predict_locations.append(last_location)

    # ذخیره در فایل Excel
    # columns = ["image_index", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    # df = pd.DataFrame(all_corners, columns=columns)
    # df.to_excel(f"/content/drive/MyDrive/{folder_name}/predicted.xlsx", index=False)
    # print("📁 Saved all corner points to four_point_1_mul6.xlsx")
    #برای ترک مسیر
    df = pd.DataFrame(predict_locations, columns=["X", "Y"])
    
    # ذخیره به فایل اکسل
    df.to_excel("/content/drive/MyDrive/predict_locations.xlsx", index=False)

if __name__ == '__main__':
    args = parser.parse_arguments()
    # args.resize_width = 512 # just for test
    start_time = datetime.now()
    if args.identity:
        pass
    else:
        args.save_dir = join(
        "test",
        args.save_dir,
        args.eval_model.split("/")[-2] if args.eval_model is not None else args.eval_model_ue.split("/")[-2],
        f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
        )
        commons.setup_logging(args.save_dir, console='info')
    setup_seed(0)
    logging.debug(args)
    wandb_log = True
    # if wandb_log:
    #     wandb.init(project="STHN-eval", entity="xjh19971", config=vars(args))
    test(args, wandb_log)