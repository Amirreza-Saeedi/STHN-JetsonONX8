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
    # else:
    #     model = None
    #  if args.test:
    #      val_dataset = datasets.fetch_dataloader(args, split='test')
    #  else:
    #      val_dataset = datasets.fetch_dataloader(args, split='val')
    # evaluate_SNet(model, val_dataset, batch_size=args.batch_size, args=args, wandb_log=wandb_log)
    
    # img2 = (base_transform(query_transform(Image.open("/content/drive/MyDrive/image_2701.png")))).unsqueeze(0)
    # img1 = F.to_tensor(Image.open("/content/drive/MyDrive/1536-1536.png").convert("RGB")).unsqueeze(0)
    
    # # img2 = F.to_tensor((Image.open("/content/drive/MyDrive/image_3275.png"))).unsqueeze(0)
    # # img1 = base_transform(query_transform(Image.open("/content/drive/MyDrive/1536-1536.png").convert("RGB"))).unsqueeze(0)
    # # print("img1 shape:", img1.shape)
    # # print("img2 shape:", img2.shape)
    # model.set_input(img1, img2)
    # model.forward()
    # four_pred = model.four_pred
    # four_point_org_single = torch.zeros((1, 2, 2, 2))
    # four_point_org_single[:, :, 0, 0] = torch.Tensor([0, 0])
    # four_point_org_single[:, :, 0, 1] = torch.Tensor([args.resize_width - 1, 0])
    # four_point_org_single[:, :, 1, 0] = torch.Tensor([0, args.resize_width - 1])
    # four_point_org_single[:, :, 1, 1] = torch.Tensor([args.resize_width - 1, args.resize_width - 1])
    # four_point_1 = four_pred.cpu().detach() + four_point_org_single
    # four_point_org = four_point_org_single.repeat(four_point_1.shape[0],1,1,1).flatten(2).permute(0, 2, 1).contiguous() 
    # four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous()
    # four_point_1_mul6 = four_point_1 * 6
    # print(f"four_point_1:{four_point_1}\nfour_point_1_mul6:{four_point_1_mul6}")
    folder_name = "maps_results/farm"
    all_corners = []
    times = []
    for i in range(6):
        try:
            # مسیر تصاویر با شماره i
            
            img1_path = f"js_datasets/qomFly2-400m/satellite/tile_{i+1011}.png"
            img2_path = f"js_datasets/qomFly2-400m/thermal/frame_{i*3 +3096}.png"
    
            # خواندن تصاویر
            img1 = F.to_tensor(Image.open(img1_path).convert("RGB")).unsqueeze(0)
            img2 = (base_transform(query_transform(Image.open(img2_path)))).unsqueeze(0)
            start_time = time.time()
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
            four_point_1_mul6 = four_point_1 * 6
            center = four_point_1_mul6.mean(dim=1)  # شکل (1,2)
            center = tuple(center[0].tolist())
            print(center)
            print(four_point_1_mul6)
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
        
            # استخراج نقاط پیش‌بینی‌شده (4 گوشه)
            points = four_point_1_mul6.squeeze(0).tolist()  # 4 × 2 لیست
            flat_points = [coord for point in points for coord in point]  # تبدیل به لیست 8 تایی
    
            all_corners.append([i] + flat_points)  # اضافه کردن شماره عکس + نقاط
    
            print(f"✅ Done for image {i}")
    
        except Exception as e:
            print(f"❌ Error in image {i}: {e}")
            
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n📊 Average processing time per image: {avg_time:.4f} sec")

    # ذخیره در فایل Excel
    columns = ["image_index", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    df = pd.DataFrame(all_corners, columns=columns)
    df.to_excel(f"js_excels-gpu/predicted.xlsx", index=False)
    print("📁 Saved all corner points to four_point_1_mul6.xlsx")


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
