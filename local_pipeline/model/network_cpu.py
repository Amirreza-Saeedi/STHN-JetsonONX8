import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as tgm
import kornia.geometry.bbox as bbox
from update import GMA
from extractor import BasicEncoderQuarter
from corr import CorrBlock
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


def _get_perspective_transform_onnx_safe(points_src: torch.Tensor, points_dst: torch.Tensor) -> torch.Tensor:
    """Compute homography H mapping points_src -> points_dst.

    ONNX-friendly alternative to kornia.geometry.transform.get_perspective_transform,
    which may dispatch to torch.linalg.solve (aten::linalg_solve) that is not exported
    by PyTorch's ONNX exporter.

    Args:
        points_src: (B, 4, 2)
        points_dst: (B, 4, 2)

    Returns:
        H: (B, 3, 3)
    """

    # Closed-form homography from unit square to quadrilateral, then scale to pixels.
    # This avoids torch.linalg.solve / torch.inverse which the ONNX exporter cannot lower.
    # Point order is expected as: TL, TR, BL, BR.

    eps = 1e-6

    # Infer width/height from src rectangle (TL assumed to be 0,0)
    w = points_src[:, 1, 0].clamp_min(eps)  # (B,)
    h = points_src[:, 2, 1].clamp_min(eps)  # (B,)
    w_ = w.unsqueeze(-1)
    h_ = h.unsqueeze(-1)

    # Normalize destination corners to unit square coordinates
    xd = points_dst[..., 0] / w_
    yd = points_dst[..., 1] / h_

    x0, y0 = xd[:, 0], yd[:, 0]
    x1, y1 = xd[:, 1], yd[:, 1]
    x2, y2 = xd[:, 2], yd[:, 2]
    x3, y3 = xd[:, 3], yd[:, 3]

    dx1 = x1 - x2
    dx2 = x3 - x2
    dx3 = x0 - x1 + x2 - x3
    dy1 = y1 - y2
    dy2 = y3 - y2
    dy3 = y0 - y1 + y2 - y3

    den = (dx1 * dy2 - dx2 * dy1).clamp_min(eps)
    g = (dx3 * dy2 - dx2 * dy3) / den
    hh = (dx1 * dy3 - dx3 * dy1) / den

    a = x1 - x0 + g * x1
    b = x3 - x0 + hh * x3
    c = x0
    d = y1 - y0 + g * y1
    e = y3 - y0 + hh * y3
    f = y0

    H_norm = torch.stack(
        [
            torch.stack([a, b, c], dim=-1),
            torch.stack([d, e, f], dim=-1),
            torch.stack([g, hh, torch.ones_like(g)], dim=-1),
        ],
        dim=-2,
    )  # (B, 3, 3)

    # Convert unit-square homography to pixel-space homography.
    # H_pixel = S_inv * H_norm * S, where S = diag([1/w, 1/h, 1]) and S_inv = diag([w, h, 1])
    zeros = torch.zeros_like(w)
    ones = torch.ones_like(w)
    S = torch.stack(
        [
            torch.stack([1.0 / w, zeros, zeros], dim=-1),
            torch.stack([zeros, 1.0 / h, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    S_inv = torch.stack(
        [
            torch.stack([w, zeros, zeros], dim=-1),
            torch.stack([zeros, h, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    return torch.bmm(torch.bmm(S_inv, H_norm), S)


def _autocast(device_type, enabled):
    return torch.autocast(device_type=device_type, enabled=enabled)


class IHN(nn.Module):
    def __init__(self, args, first_stage):
        super().__init__()
        self.device = torch.device(args.device)
        self.args = args
        self.hidden_dim = 128
        self.context_dim = 128
        self.first_stage = first_stage
        self.fnet1 = BasicEncoderQuarter(output_dim=256, norm_fn='instance')
        if self.args.lev0:
            sz = self.args.resize_width // 4
            self.update_block_4 = GMA(self.args, sz)

        self.imagenet_mean = None
        self.imagenet_std = None

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
        H = _get_perspective_transform_onnx_safe(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(
            torch.linspace(0, self.args.resize_width // 4 - 1, steps=self.args.resize_width // 4),
            torch.linspace(0, self.args.resize_width // 4 - 1, steps=self.args.resize_width // 4),
            indexing='ij',
        )
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, self.args.resize_width//4 * self.args.resize_width//4))),
                           dim=0).unsqueeze(0).repeat(H.shape[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)
        if torch.isnan(points_new).any():
            raise KeyError("Some of transformed coords are NaN!")
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                          points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        return flow

    def get_flow_now_2(self, four_point):
        four_point = four_point / 2
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
        H = _get_perspective_transform_onnx_safe(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(
            torch.linspace(0, self.sz[3] - 1, steps=self.sz[3]),
            torch.linspace(0, self.sz[2] - 1, steps=self.sz[2]),
            indexing='ij',
        )
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, self.sz[3] * self.sz[2]))),
                           dim=0).unsqueeze(0).repeat(self.sz[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                          points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        return flow

    def initialize_flow_4(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//4, W//4).to(img.device)
        coords1 = coords_grid(N, H//4, W//4).to(img.device)

        return coords0, coords1

    def initialize_flow_2(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//2, W//2).to(img.device)
        coords1 = coords_grid(N, H//2, W//2).to(img.device)

        return coords0, coords1

    def forward(self, image1, image2, iters_lev0=6, iters_lev1=6, corr_level=2, corr_radius=4):
        if self.imagenet_mean is None:
            self.imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(image1.device)
            self.imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(image1.device)
        image1 = (image1.contiguous() - self.imagenet_mean) / self.imagenet_std
        image2 = (image2.contiguous() - self.imagenet_mean) / self.imagenet_std

        with _autocast(self.device.type, enabled=self.args.mixed_precision and self.device.type != "cpu"):
            if not self.args.fnet_cat:
                fmap1_64 = self.fnet1(image1)
                fmap2_64 = self.fnet1(image2)
            else:
                fmap_64 = self.fnet1(torch.cat([image1, image2], dim=0))
                fmap1_64 = fmap_64[:image1.shape[0]]
                fmap2_64 = fmap_64[image1.shape[0]:]

        fmap1 = fmap1_64.float()
        fmap2 = fmap2_64.float()

        corr_fn = CorrBlock(fmap1, fmap2, num_levels=corr_level, radius=corr_radius)
        coords0, coords1 = self.initialize_flow_4(image1)
        sz = fmap1_64.shape
        self.sz = sz
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)
        four_point_predictions = []
        for itr in range(iters_lev0):
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with _autocast(self.device.type, enabled=self.args.mixed_precision and self.device.type != "cpu"):
                if self.args.weight:
                    delta_four_point, weight = self.update_block_4(corr, flow)
                else:
                    delta_four_point = self.update_block_4(corr, flow)

            try:
                last_four_point_disp = four_point_disp
                four_point_disp = four_point_disp + delta_four_point
                coords1 = self.get_flow_now_4(four_point_disp)
                four_point_predictions.append(four_point_disp)
            except Exception as e:
                logging.debug(e)
                logging.debug("Ignore this delta. Use last disp.")
                four_point_disp = last_four_point_disp
                coords1 = self.get_flow_now_4(four_point_disp)
                four_point_predictions.append(four_point_disp)
        return four_point_predictions, four_point_disp


arch_list = {"IHN": IHN}


class STHN():
    def __init__(self, args, for_training=False):
        super().__init__()
        args.device = "cpu"
        self.args = args
        self.device = args.device
        self.four_point_org_single = torch.zeros((1, 2, 2, 2)).to(self.device)
        self.four_point_org_single[:, :, 0, 0] = torch.Tensor([0, 0]).to(self.device)
        self.four_point_org_single[:, :, 0, 1] = torch.Tensor([self.args.resize_width - 1, 0]).to(self.device)
        self.four_point_org_single[:, :, 1, 0] = torch.Tensor([0, self.args.resize_width - 1]).to(self.device)
        self.four_point_org_single[:, :, 1, 1] = torch.Tensor([self.args.resize_width - 1, self.args.resize_width - 1]).to(self.device)
        self.four_point_org_large_single = torch.zeros((1, 2, 2, 2)).to(self.device)
        self.four_point_org_large_single[:, :, 0, 0] = torch.Tensor([0, 0]).to(self.device)
        self.four_point_org_large_single[:, :, 0, 1] = torch.Tensor([self.args.database_size - 1, 0]).to(self.device)
        self.four_point_org_large_single[:, :, 1, 0] = torch.Tensor([0, self.args.database_size - 1]).to(self.device)
        self.four_point_org_large_single[:, :, 1, 1] = torch.Tensor([self.args.database_size - 1, self.args.database_size - 1]).to(self.device)
        self.netG = arch_list[args.arch](args, True)
        self.shift_flow_bbox = None
        if args.two_stages:
            corr_level = args.corr_level
            args.corr_level = 2
            self.netG_fine = IHN(args, False)
            args.corr_level = corr_level
            if args.restore_ckpt is not None and not args.finetune:
                self.set_requires_grad(self.netG, False)
        self.criterionAUX = sequence_loss if self.args.arch == "IHN" else single_loss
        if for_training:
            if args.two_stages:
                if args.restore_ckpt is None or args.finetune:
                    self.optimizer_G, self.scheduler_G = fetch_optimizer(args, list(self.netG.parameters()) + list(self.netG_fine.parameters()))
                else:
                    self.optimizer_G, self.scheduler_G = fetch_optimizer(args, list(self.netG_fine.parameters()))
            else:
                self.optimizer_G, self.scheduler_G = fetch_optimizer(args, list(self.netG.parameters()))
            self.G_loss_lambda = args.G_loss_lambda

    def setup(self):
        self.netG = self.init_net(self.netG)
        if hasattr(self, 'netG_fine'):
            self.netG_fine = self.init_net(self.netG_fine)

    def init_net(self, model):
        model = model.to(self.device)
        return model

    def set_input(self, A, B, flow_gt=None):
        self.image_1_ori = A.to(self.device)
        self.image_2 = B.to(self.device)
        self.real_warped_image_2 = None
        self.image_1 = F.interpolate(self.image_1_ori, size=self.args.resize_width, mode='bilinear', align_corners=True, antialias=True)

    def forward(self, for_training=False):
        self.four_preds_list, self.four_pred = self.netG(image1=self.image_1, image2=self.image_2, iters_lev0=self.args.iters_lev0, corr_level=self.args.corr_level)
        if self.args.two_stages:
            self.image_1_crop, delta, self.flow_bbox = self.get_cropped_st_images(self.image_1_ori, self.four_pred, self.args.fine_padding, self.args.detach, self.args.augment_two_stages)
            self.image_2_crop = self.image_2
            self.four_preds_list_fine, self.four_pred_fine = self.netG_fine(image1=self.image_1_crop, image2=self.image_2_crop, iters_lev0=self.args.iters_lev1)
            self.four_preds_list, self.four_pred = self.combine_coarse_fine(self.four_preds_list, self.four_pred, self.four_preds_list_fine, self.four_pred_fine, delta, self.flow_bbox, for_training)
        if self.args.vis_all:
            self.fake_warped_image_2 = mywarp(self.image_2, self.four_pred, self.four_point_org_single)
        return self.four_pred

    def get_cropped_st_images(self, image_1_ori, four_pred, fine_padding, detach=True, augment_two_stages=0):
        four_point = four_pred + self.four_point_org_single
        x = four_point[:, 0]
        y = four_point[:, 1]
        alpha = self.args.database_size / self.args.resize_width
        x[:, :, 0] = x[:, :, 0] * alpha
        x[:, :, 1] = (x[:, :, 1] + 1) * alpha
        y[:, 0, :] = y[:, 0, :] * alpha
        y[:, 1, :] = (y[:, 1, :] + 1) * alpha
        left = torch.min(x.view(x.shape[0], -1), dim=1)[0]
        right = torch.max(x.view(x.shape[0], -1), dim=1)[0]
        top = torch.min(y.view(y.shape[0], -1), dim=1)[0]
        bottom = torch.max(y.view(y.shape[0], -1), dim=1)[0]
        if augment_two_stages != 0:
            if self.args.augment_type == "bbox":
                left += (torch.rand(left.shape).to(left.device) * 2 - 1) * augment_two_stages
                right += (torch.rand(right.shape).to(right.device) * 2 - 1) * augment_two_stages
                top += (torch.rand(top.shape).to(top.device) * 2 - 1) * augment_two_stages
                bottom += (torch.rand(bottom.shape).to(bottom.device) * 2 - 1) * augment_two_stages
            w = torch.max(torch.stack([right-left, bottom-top], dim=1), dim=1)[0]
            c = torch.stack([(left + right)/2, (bottom + top)/2], dim=1)
            if self.args.augment_type == "center":
                w += torch.rand(w.shape).to(w.device) * augment_two_stages
                c += (torch.rand(c.shape).to(c.device) * 2 - 1) * augment_two_stages
        else:
            w = torch.max(torch.stack([right-left, bottom-top], dim=1), dim=1)[0]
            c = torch.stack([(left + right)/2, (bottom + top)/2], dim=1)
        w_padded = w + 2 * fine_padding
        crop_top_left = c + torch.stack([-w_padded / 2, -w_padded / 2], dim=1)
        x_start = crop_top_left[:, 0]
        y_start = crop_top_left[:, 1]
        bbox_s = bbox.bbox_generator(x_start, y_start, w_padded, w_padded)
        delta = (w_padded / self.args.resize_width).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        image_1_crop = tgm.crop_and_resize(image_1_ori, bbox_s, (self.args.resize_width, self.args.resize_width))
        bbox_s_swap = torch.stack([bbox_s[:, 0], bbox_s[:, 1], bbox_s[:, 3], bbox_s[:, 2]], dim=1)
        four_cor_bbox = bbox_s_swap.permute(0, 2, 1). view(-1, 2, 2, 2)
        flow_bbox = four_cor_bbox - self.four_point_org_large_single
        if detach:
            image_1_crop = image_1_crop.detach()
            delta = delta.detach()
            flow_bbox = flow_bbox.detach()
        return image_1_crop, delta, flow_bbox

    def combine_coarse_fine(self, four_preds_list, four_pred, four_preds_list_fine, four_pred_fine, delta, flow_bbox, for_training):
        alpha = self.args.database_size / self.args.resize_width
        kappa = delta / alpha
        four_preds_list_fine = [four_preds_list_fine_single * kappa + flow_bbox / alpha for four_preds_list_fine_single in four_preds_list_fine]
        four_pred_fine = four_pred_fine * kappa + flow_bbox / alpha
        four_preds_list = four_preds_list + four_preds_list_fine
        return four_preds_list, four_pred_fine

    def backward_G(self):
        self.loss_G_Homo, self.metrics = self.criterionAUX(self.four_preds_list, self.four_pred, self.flow_gt, self.args.gamma, self.args, self.metrics)
        self.loss_G = self.loss_G_Homo * self.G_loss_lambda
        self.metrics["G_loss"] = self.loss_G.cpu().item()
        self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        self.forward(for_training=True)
        self.metrics = dict()
        self.optimizer_G.zero_grad()
        self.backward_G()
        if self.args.restore_ckpt is None or self.args.finetune:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.args.clip)
        if self.args.two_stages:
            nn.utils.clip_grad_norm_(self.netG_fine.parameters(), self.args.clip)
        self.optimizer_G.step()
        return self.metrics

    def update_learning_rate(self):
        self.scheduler_G.step()


def mywarp(x, flow_pred, four_point_org_single, ue_std=None):
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

        four_point_org = four_point_org_single.repeat(flow_pred.shape[0], 1, 1, 1).flatten(2).permute(0, 2, 1).contiguous()
        four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous()
        try:
            H = tgm.get_perspective_transform(four_point_org, four_point_1)
        except Exception:
            logging.debug("No solution")
            H = torch.eye(3).to(four_point_org.device).repeat(four_point_1.shape[0], 1, 1)
        warped_image = tgm.warp_perspective(x, H, (x.shape[2], x.shape[3]))
    else:
        logging.debug("Output NaN by model error.")
        warped_image = x
    return warped_image
