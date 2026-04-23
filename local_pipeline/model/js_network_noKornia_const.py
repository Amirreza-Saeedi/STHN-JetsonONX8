import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as tgm
import kornia.geometry.bbox as bbox
from update_const import GMA
from extractor_const import BasicEncoderQuarter
from corr_const import CorrBlock
from utils import coords_grid, sequence_loss, single_loss, fetch_optimizer, warp
import os
import sys
from model.sync_batchnorm import convert_model
import wandb
import torchvision
import random
import time
import logging
import datasets_4cor_img as datasets
import numpy as np
import kornia.geometry.transform as tgm  # Only for warp_perspective

# autocast = torch.cuda.amp.autocast
autocast = torch.amp.autocast

import sys
from model.js_kornia_replacement import (
    get_perspective_transform_torch, 
    crop_and_resize_torch,
    bbox_generator_torch,
    warp_perspective_torch
)


class IHN(nn.Module):
    def __init__(self, args, first_stage):
        super().__init__()
        self.device = args.device # TODO
        self.args = args
        self.hidden_dim = 128
        self.context_dim = 128
        self.first_stage = first_stage
        self.fnet1 = BasicEncoderQuarter(output_dim=256, norm_fn='instance')

        sz = 64
        self.update_block_4 = GMA(self.args, sz)

        self.imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(self.device)
        self.imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(self.device)
        
    def get_flow_now_4(self, four_point):
        four_point = four_point / 4
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3]-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2]-1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3]-1, self.sz[2]-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)
        four_point_new = four_point_org + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1).contiguous()
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1).contiguous()
        
        # REPLACED: Use custom implementation instead of kornia
        H = get_perspective_transform_torch(four_point_org, four_point_new)
        
        gridy, gridx = torch.meshgrid(
            torch.linspace(0, 64-1, steps=64), 
            torch.linspace(0, 64-1, steps=64),
            indexing='ij'
        )
        points = torch.cat(
            (gridx.flatten().unsqueeze(0), 
             gridy.flatten().unsqueeze(0), 
             torch.ones((1, 64 * 64))),
            dim=0
        ).unsqueeze(0).repeat(H.shape[0], 1, 1).to(four_point.device)
        
        points_new = H.bmm(points)
        
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat(
            (points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
             points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), 
            dim=1
        )
        return flow


    def initialize_flow_4(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//4, W//4).to(img.device)
        coords1 = coords_grid(N, H//4, W//4).to(img.device)
        return coords0, coords1

    def forward(self, image1, image2, iters_lev0 = 6, iters_lev1=6, corr_level=2, corr_radius=4):
        # Scaling to Mean 0, STD 1
        image1 = (image1.contiguous() - self.imagenet_mean) / self.imagenet_std
        image2 = (image2.contiguous() - self.imagenet_mean) / self.imagenet_std

        # Extract
        with autocast(device_type='cuda', enabled=self.args.mixed_precision):  # TODO
            fmap1_64 = self.fnet1(image1)
            fmap2_64 = self.fnet1(image2)
        fmap1 = fmap1_64.float()
        fmap2 = fmap2_64.float()
        # fmap1 = self.fnet1(image1)
        # fmap2 = self.fnet1(image2)

        # Corr
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=corr_level, radius=corr_radius)

        # Update
        coords0, coords1 = self.initialize_flow_4(image1)
        sz = fmap1.shape
        self.sz = sz
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)
        four_point_predictions = []
        for itr in range(iters_lev0):
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(device_type='cuda', enabled=self.args.mixed_precision):  # TODO
                delta_four_point = self.update_block_4(corr, flow)
            # delta_four_point = self.update_block_4(corr, flow)
                    
            last_four_point_disp = four_point_disp
            four_point_disp =  four_point_disp + delta_four_point
            coords1 = self.get_flow_now_4(four_point_disp) # Possible error: Unsolvable H
            four_point_predictions.append(four_point_disp)
            
        # time2 = time.time()
        # print("Time for iterative: " + str(time2 - time1) + " seconds") # 0.12
        return four_point_predictions, four_point_disp


class STHN(nn.Module):
    def __init__(self, args, for_training=False):
        super().__init__()
        self.args = args
        self.device = args.device
        
        four_point_org_single = torch.zeros((1, 2, 2, 2), device=self.device)
        four_point_org_single[:, :, 0, 0] = torch.tensor([0, 0], device=self.device)
        four_point_org_single[:, :, 0, 1] = torch.tensor([self.args.resize_width - 1, 0], device=self.device)
        four_point_org_single[:, :, 1, 0] = torch.tensor([0, self.args.resize_width - 1], device=self.device)
        four_point_org_single[:, :, 1, 1] = torch.tensor([self.args.resize_width - 1, self.args.resize_width - 1], device=self.device)
        self.register_buffer('four_point_org_single', four_point_org_single)
        
        four_point_org_large_single = torch.zeros((1, 2, 2, 2), device=self.device)
        four_point_org_large_single[:, :, 0, 0] = torch.tensor([0, 0], device=self.device)
        four_point_org_large_single[:, :, 0, 1] = torch.tensor([self.args.database_size - 1, 0], device=self.device)
        four_point_org_large_single[:, :, 1, 0] = torch.tensor([0, self.args.database_size - 1], device=self.device)
        four_point_org_large_single[:, :, 1, 1] = torch.tensor([self.args.database_size - 1, self.args.database_size - 1], device=self.device)
        self.register_buffer('four_point_org_large_single', four_point_org_large_single)

        # Sub Modules
        self.netG = IHN(args, True)
        self.shift_flow_bbox = None
        corr_level = args.corr_level
        args.corr_level = 2
        self.netG_fine = IHN(args, False)
        args.corr_level = corr_level
        
        # self.set_requires_grad(self.netG, False) # TODO

        self.criterionAUX = sequence_loss 
        
            
    def setup(self):
        self.netG = self.init_net(self.netG)
        self.netG_fine = self.init_net(self.netG_fine)

    def init_net(self, model):
        model = model.to(self.device)
        return model
    
    # def set_input(self, A, B, flow_gt=None):
    #     self.image_1_ori = A.to(self.device, non_blocking=True)
    #     self.image_2 = B.to(self.device, non_blocking=True)

    #     self.real_warped_image_2 = None
    #     self.image_1 = F.interpolate(self.image_1_ori, size=self.args.resize_width, mode='bilinear', align_corners=True, antialias=True)
        
    def forward(self, image1, image2):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        # FIXED: Move input tensors to the model's device
        device = next(self.parameters()).device
        image1 = image1.to(device)
        image2 = image2.to(device)

        # Interpolate input
        image1_resized = F.interpolate(
            image1, size=self.args.resize_width, 
            mode='bilinear', align_corners=True, antialias=True
        )
        # Coarse prediction
        four_preds_list, four_pred = self.netG(
            image1=image1_resized, 
            image2=image2, 
            iters_lev0=self.args.iters_lev0, 
            corr_level=self.args.corr_level
        )
        
        # Crop for fine stage
        image1_crop, delta, flow_bbox = self.get_cropped_st_images(
            image1, four_pred, self.args.fine_padding, 
            self.args.detach, self.args.augment_two_stages
        )
        
        # Fine prediction
        four_preds_list_fine, four_pred_fine = self.netG_fine(
            image1=image1_crop, 
            image2=image2, 
            iters_lev0=self.args.iters_lev1
        )
        
        # Combine results
        _, four_pred_combined = self.combine_coarse_fine(
            four_preds_list, four_pred, 
            four_preds_list_fine, four_pred_fine, 
            delta, flow_bbox, for_training=False
        )

        self.four_pred = four_pred_combined
        
        return four_pred_combined

    def get_cropped_st_images(self, image_1_ori, four_pred, fine_padding, detach=True, augment_two_stages=0):
        # From four_pred to bbox coordinates
        four_point = four_pred + self.four_point_org_single
        x = four_point[:, 0]
        y = four_point[:, 1]
        # Make it same scale as image_1_ori
        alpha = self.args.database_size / self.args.resize_width
        x[:, :, 0] = x[:, :, 0] * alpha
        x[:, :, 1] = (x[:, :, 1] + 1) * alpha
        y[:, 0, :] = y[:, 0, :] * alpha
        y[:, 1, :] = (y[:, 1, :] + 1) * alpha
        # Crop
        left = torch.min(x.view(x.shape[0], -1), dim=1)[0]
        right = torch.max(x.view(x.shape[0], -1), dim=1)[0]
        top = torch.min(y.view(y.shape[0], -1), dim=1)[0]
        bottom = torch.max(y.view(y.shape[0], -1), dim=1)[0]
        
        w = torch.max(torch.stack([right-left, bottom-top], dim=1), dim=1)[0]
        c = torch.stack([(left + right)/2, (bottom + top)/2], dim=1)
        
        w_padded = w + 2 * fine_padding
        crop_top_left = c + torch.stack([-w_padded / 2, -w_padded / 2], dim=1)
        x_start = crop_top_left[:, 0]
        y_start = crop_top_left[:, 1]
        
        # REPLACED: Use custom bbox_generator instead of kornia
        bbox_s = bbox_generator_torch(x_start, y_start, w_padded, w_padded)
        
        delta = (w_padded / self.args.resize_width).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        # REPLACED: Use custom crop_and_resize instead of kornia
        image_1_crop = crop_and_resize_torch(image_1_ori, bbox_s, 
                                            (self.args.resize_width, self.args.resize_width))
        
        # swap bbox_s for flow_bbox calculation
        bbox_s_swap = torch.stack([bbox_s[:, 0], bbox_s[:, 1], bbox_s[:, 3], bbox_s[:, 2]], dim=1)
        four_cor_bbox = bbox_s_swap.permute(0, 2, 1).view(-1, 2, 2, 2)
        flow_bbox = four_cor_bbox - self.four_point_org_large_single
        
        if detach:
            image_1_crop = image_1_crop.detach()
            delta = delta.detach()
            flow_bbox = flow_bbox.detach()
        
        return image_1_crop, delta, flow_bbox
    
    def combine_coarse_fine(self, four_preds_list, four_pred, four_preds_list_fine, four_pred_fine, delta, flow_bbox, for_training=False):
        alpha = self.args.database_size / self.args.resize_width
        kappa = delta / alpha
        four_preds_list_fine = [four_preds_list_fine_single * kappa + flow_bbox / alpha for four_preds_list_fine_single in four_preds_list_fine]
        four_pred_fine = four_pred_fine * kappa + flow_bbox / alpha
        four_preds_list = four_preds_list + four_preds_list_fine
        return four_preds_list, four_pred_fine

    def backward_G(self):  # XXX not used
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        self.loss_G_Homo, self.metrics = self.criterionAUX(self.four_preds_list, self.four_pred, self.flow_gt, self.args.gamma, self.args, self.metrics) 
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_Homo * self.G_loss_lambda
        self.metrics["G_loss"] = self.loss_G.cpu().item()
        self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):  # XXX not used
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):  # XXX not used
        self.forward(for_training=True) # Calculate Fake A
        self.metrics = dict()
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        if self.args.restore_ckpt is None or self.args.finetune:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.args.clip)
        if self.args.two_stages:
            nn.utils.clip_grad_norm_(self.netG_fine.parameters(), self.args.clip)
        self.optimizer_G.step()             # update G's weights
        return self.metrics

    def update_learning_rate(self):  # XXX not used
        """Update learning rates for all the networks; called at the end of every epoch"""
        self.scheduler_G.step()

def mywarp(x, flow_pred, four_point_org_single, ue_std=None):  # XXX not used
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    """
    if not torch.isnan(flow_pred).any():
        if flow_pred.shape[-1] != 2:
            flow_4cor = torch.zeros((flow_pred.shape[0], 2, 2, 2)).to(flow_pred.device)
            flow_4cor[:, :, 0, 0] = flow_pred[:, :, 0, 0]
            flow_4cor[:, :, 0, 1] = flow_pred[:, :, 0, -1]
            flow_4cor[:, :, 1, 0] = flow_pred[:, :, -1, 0]
            flow_4cor[:, :, 1, 1] = flow_pred[:, :, -1, -1]
        else:
            flow_4cor = flow_pred

        four_point_1 = flow_4cor + four_point_org_single
        
        four_point_org = four_point_org_single.repeat(flow_pred.shape[0],1,1,1).flatten(2).permute(0, 2, 1).contiguous() 
        four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous() 
        
        try:
            # REPLACED: Use custom implementation instead of kornia
            H = get_perspective_transform_torch(four_point_org, four_point_1)
        except Exception:
            logging.debug("No solution")
            H = torch.eye(3).to(four_point_org.device).repeat(four_point_1.shape[0],1,1)
        
        # This one still uses kornia since you said it works
        warped_image = warp_perspective_torch(x, H, (x.shape[2], x.shape[3]))
    else:
        logging.debug("Output NaN by model error.")
        warped_image = x
    
    return warped_image