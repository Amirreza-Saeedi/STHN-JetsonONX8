import numpy as np
import os
import torch
import argparse
import sys
print(sys.path)
from model.js_network_noKornia import STHN
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
import js_utils
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
    print("--1--")
    if not args.identity:
        print("--2--")
        model = STHN(args)
        if not args.train_ue_method == "train_only_ue_raw_input":
            model_med = torch.load(args.eval_model, map_location='cuda:0')
            print("--3--")
            for key in list(model_med['netG'].keys()):
                model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
            for key in list(model_med['netG'].keys()):
                if key.startswith('module'):
                    del model_med['netG'][key]
            model.netG.load_state_dict(model_med['netG'], strict=False)
        # if args.use_ue:
        #     if args.eval_model_ue is not None:
        #         model_med = torch.load(args.eval_model_ue, map_location='cuda:0')
        #     for key in list(model_med['netD'].keys()):
        #         model_med['netD'][key.replace('module.','')] = model_med['netD'][key]
        #     for key in list(model_med['netD'].keys()):
        #         if key.startswith('module'):
        #             del model_med['netD'][key]
        #     model.netD.load_state_dict(model_med['netD'])
        if args.two_stages:
            print("--6--")
            if args.eval_model_fine is None:
                print("--7--")
                model_med = torch.load(args.eval_model, map_location='cuda:0')
                for key in list(model_med['netG_fine'].keys()):
                    model_med['netG_fine'][key.replace('module.','')] = model_med['netG_fine'][key]
                for key in list(model_med['netG_fine'].keys()):
                    if key.startswith('module'):
                        del model_med['netG_fine'][key]
                model.netG_fine.load_state_dict(model_med['netG_fine'])
            else:
                print("--8--")
                model_med = torch.load(args.eval_model_fine, map_location='cuda:0')
                for key in list(model_med['netG'].keys()):
                    model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
                for key in list(model_med['netG'].keys()):
                    if key.startswith('module'):
                        del model_med['netG'][key]
                model.netG_fine.load_state_dict(model_med['netG'], strict=False)
        # js_utils.print_gpu_mem('Before Eval')

        import torch.nn.utils.prune as prune
        import torch.nn as nn
        def structured_prune_model(model, amount=0.3):
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # prune output channels (dim=0)
                    prune.ln_structured(
                        module,
                        name="weight",
                        amount=amount,
                        n=2,
                        dim=0
                    )
                    prune.remove(module, "weight")  # make permanent
            return model
        
        def get_alive_channels(weight):
            # weight: [out_channels, in_channels, k, k]
            alive = torch.norm(weight.view(weight.size(0), -1), dim=1) > 0
            return alive
        
        def prune_conv_layer(conv, in_mask=None):    
            W = conv.weight.data
            out_mask = get_alive_channels(W)
            if in_mask is not None:        W = W[:, in_mask, :, :]
            W = W[out_mask, :, :, :]
            new_conv = nn.Conv2d(        in_channels=W.shape[1],        out_channels=W.shape[0],        kernel_size=conv.kernel_size,        stride=conv.stride,        padding=conv.padding,        bias=(conv.bias is not None)    )
            new_conv.weight.data = W.clone()
            if conv.bias is not None:        new_conv.bias.data = conv.bias.data[out_mask].clone()
            return new_conv, out_mask

        def find_valid_groups(num_channels, max_groups=32):
            # find largest divisor ≤ max_groups
            for g in reversed(range(1, max_groups + 1)):
                if num_channels % g == 0:
                    return g
            return 1  # fallback

        def prune_groupnorm(gn, mask):
            new_channels = int(mask.sum().item())

            new_groups = find_valid_groups(new_channels)

            new_gn = nn.GroupNorm(new_groups, new_channels)

            new_gn.weight.data = gn.weight.data[mask].clone()
            new_gn.bias.data = gn.bias.data[mask].clone()

            return new_gn

        def surgery_layer(seq, in_mask=None):
            conv = seq[0]
            gn = seq[1]
            relu = seq[2]
            pool = seq[3]

            new_conv, out_mask = prune_conv_layer(conv, in_mask)
            new_gn = prune_groupnorm(gn, out_mask)

            new_seq = nn.Sequential(new_conv, new_gn, relu, pool)

            return new_seq, out_mask

        def surgery_cnn64(model):
            mask = None

            model.layer1, mask = surgery_layer(model.layer1, mask)
            model.layer2, mask = surgery_layer(model.layer2, mask)
            model.layer3, mask = surgery_layer(model.layer3, mask)
            model.layer4, mask = surgery_layer(model.layer4, mask)
            model.layer5, mask = surgery_layer(model.layer5, mask)

            # final layer (no pooling)
            conv1 = model.layer10[0]
            gn = model.layer10[1]
            relu = model.layer10[2]
            conv2 = model.layer10[3]

            conv1, mask = prune_conv_layer(conv1, mask)
            gn = prune_groupnorm(gn, mask)

            conv2, _ = prune_conv_layer(conv2, mask)

            model.layer10 = nn.Sequential(conv1, gn, relu, conv2)

            return model

        # model = CNN_64(128, init_dim=164)

        # # load weights first
        # model.load_state_dict(...)

        parameters = sum(p.numel() for p in model.netG.update_block_4.cnn.parameters())
        print(parameters)
        # prune
        model.netG.update_block_4.cnn = structured_prune_model(model.netG.update_block_4.cnn, amount=0.6)

        model.netG.update_block_4.cnn = surgery_cnn64(model.netG.update_block_4.cnn)

        # prune
        model.netG_fine.update_block_4.cnn = structured_prune_model(model.netG_fine.update_block_4.cnn, amount=0.3)

        model.netG_fine.update_block_4.cnn = surgery_cnn64(model.netG_fine.update_block_4.cnn)

        parameters = sum(p.numel() for p in model.netG.update_block_4.cnn.parameters())
        print(parameters)

        model.setup() 
        model.netG.eval()
        if args.use_ue:
            model.netD.eval()
        if args.two_stages:
            model.netG_fine.eval()
        # js_utils.print_gpu_mem('After Eval')
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

    # model = torch.quantization.quantize_dynamic(
    #     model, {torch.nn.Linear}, dtype=torch.qint8
    # )

    # js_utils.get_module_stats(model.netG, 'IHN1')
    # js_utils.get_module_stats(model.netG.fnet1, 'extractor1')
    # js_utils.get_module_stats(model.netG.update_block_4, 'update1')
    # js_utils.get_module_stats(model.netG_fine, 'IHN2')
    # js_utils.get_module_stats(model.netG_fine.fnet1, 'extractor2')
    # js_utils.get_module_stats(model.netG_fine.update_block_4, 'update2')
    # print(10 * '=')



    folder_name = "maps_results/farm"
    all_corners = []
    times = []
    time_round1_ihn1 = 0
    time_round1_ihn2 = 0

    N = 108 # number of samples
    # N = 40 # number of samples
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
        
            # استخراج نقاط پیش‌بینی‌شده (4 گوشه)
            points = four_point_1_mul6.squeeze(0).tolist()  # 4 × 2 لیست
            flat_points = [coord for point in points for coord in point]  # تبدیل به لیست 8 تایی
    
            all_corners.append([i] + flat_points + [img1_path, img2_path])  # اضافه کردن شماره عکس + نقاط
    
            # Time Logs
            elapsed = end_time - start_time
            times.append(elapsed)
            print(f"✅ Done for image {i + 1}, {elapsed:.3}")
            if i == 0:
                time_round1_ihn1 = model.netG.times.copy()
                time_round1_ihn2 = model.netG_fine.times.copy()


        except Exception as e:
            print(f"❌ Error in image {i}: {e}")
            
    if times:
        rounds = len(times) - 1
        print(f'\nNOTICE: First data takes much more time, so it is discarded in the following report. Log is calculated for "{rounds}" images instead of "{len(times)}".')
        avg_time = sum(times[1:]) / rounds
        model.netG.times[0] -= time_round1_ihn1[0]
        model.netG.times[1] -= time_round1_ihn1[1]
        model.netG.times[2] -= time_round1_ihn1[2]
        model.netG_fine.times[0] -= time_round1_ihn2[0]
        model.netG_fine.times[1] -= time_round1_ihn2[1]
        model.netG_fine.times[2] -= time_round1_ihn2[2]
        
        print(f"📊 Average per image: {avg_time:.3f} sec, {1 / avg_time:.2f} fps")
        t0 = model.netG.times[0] / rounds
        t1 = model.netG.times[1] / rounds
        t2 = model.netG.times[2] / rounds
        t_sum = t0 + t1 + t2
        print('ttt avg extract', model.netG.ihn_str, f'{t0:.3f}, {t0 / t_sum:.2f}%')
        print('ttt avg corr   ', model.netG.ihn_str, f'{t1:.3f}, {t1 / t_sum:.2f}%')
        print('ttt avg update ', model.netG.ihn_str, f'{t2:.3f}, {t2 / t_sum:.2f}%')
        print('ttt avg sum    ', model.netG.ihn_str, f'{t_sum:.3f}')
        t0 = model.netG_fine.times[0] / rounds
        t1 = model.netG_fine.times[1] / rounds
        t2 = model.netG_fine.times[2] / rounds
        t_sum = t0 + t1 + t2
        print('ttt avg extract', model.netG_fine.ihn_str, f'{t0:.3f}, {t0 / t_sum:.2f}%')
        print('ttt avg corr   ', model.netG_fine.ihn_str, f'{t1:.3f}, {t1 / t_sum:.2f}%')
        print('ttt avg update ', model.netG_fine.ihn_str, f'{t2:.3f}, {t2 / t_sum:.2f}%')
        print('ttt avg sum    ', model.netG_fine.ihn_str, f'{t_sum:.3f}')

    # ذخیره در فایل Excel
    columns = ["image_index", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "sat", "th"]
    df = pd.DataFrame(all_corners, columns=columns)
    # df.to_excel(f"js_excels/predicted-dehat-int8.xlsx", index=False)
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
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    test(args, wandb_log)
