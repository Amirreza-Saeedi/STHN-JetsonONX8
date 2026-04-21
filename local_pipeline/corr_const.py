import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import *
import time

### ORIGINAL CODE
# class CorrBlock:
#     def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
#         self.num_levels = num_levels
#         self.radius = radius
#         self.corr_pyramid = []

#         corr = CorrBlock.corr(fmap1, fmap2)
#         batch, h1, w1, dim, h2, w2 = corr.shape
#         corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

#         self.corr_pyramid.append(corr)
#         for i in range(self.num_levels - 1):
#             corr = F.avg_pool2d(corr, 2, stride=2)
#             self.corr_pyramid.append(corr)

#         r = radius
#         dx = torch.linspace(-r, r, 2 * r + 1)
#         dy = torch.linspace(-r, r, 2 * r + 1)
#         self.delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(fmap1.device)

#     def __call__(self, coords):
#         r = self.radius
#         coords = coords.permute(0, 2, 3, 1)
#         batch, h1, w1, _ = coords.shape

#         out_pyramid = []
#         for i in range(self.num_levels):
#             corr = self.corr_pyramid[i]
#             delta = self.delta
            
#             centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
#             delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
#             coords_lvl = centroid_lvl + delta_lvl

#             corr = bilinear_sampler(corr, coords_lvl)
#             corr = corr.view(batch, h1, w1, -1)
#             out_pyramid.append(corr)

#         out = torch.cat(out_pyramid, dim=-1)
#         return out.permute(0, 3, 1, 2).contiguous().float()

#     @staticmethod
#     def corr(fmap1, fmap2):
#         batch, dim, ht, wd = fmap1.shape
#         fmap1 = fmap1.view(batch, dim, ht * wd)
#         fmap2 = fmap2.view(batch, dim, ht * wd)

#         corr = torch.relu(torch.matmul(fmap1.transpose(1, 2), fmap2))
#         corr = corr.view(batch, ht, wd, 1, ht, wd)

#         return corr


### SPEEDUP 1
class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        batch, dim, ht, wd = fmap1.shape

        # ---- Correlation (unchanged math, but efficient) ----
        fmap1_flat = fmap1.view(batch, dim, ht * wd)
        fmap2_flat = fmap2.view(batch, dim, ht * wd)

        corr = torch.bmm(fmap1_flat.transpose(1, 2), fmap2_flat)
        corr = torch.relu(corr)
        corr = corr.view(batch, ht, wd, 1, ht, wd)

        # reshape once
        corr = corr.reshape(batch * ht * wd, 1, ht, wd)

        # ---- Build pyramid ----
        self.corr_pyramid = [corr]
        for _ in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

        # ---- Precompute delta grid (keep SAME behavior as original) ----
        r = radius
        dx = torch.linspace(-r, r, 2 * r + 1, device=fmap1.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=fmap1.device)

        dy, dx = torch.meshgrid(dy, dx, indexing="ij")
        self.delta = torch.stack([dy, dx], dim=-1)  # [K, K, 2]

        # reshape once (avoid doing it every call)
        self.delta = self.delta.view(1, 2 * r + 1, 2 * r + 1, 2)

    def __call__(self, coords):
        """
        coords: [B, 2, H, W]
        """
        r = self.radius
        batch, _, h1, w1 = coords.shape

        # ---- Flatten coords once ----
        coords = coords.permute(0, 2, 3, 1)  # [B, H, W, 2]
        coords_flat = coords.reshape(batch * h1 * w1, 1, 1, 2)

        # ---- Precompute pyramid coordinates (avoids repeated division) ----
        coords_pyramid = [coords_flat]
        for _ in range(1, self.num_levels):
            coords_pyramid.append(coords_pyramid[-1] / 2.0)

        out_pyramid = []

        # ---- Sampling ----
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]

            coords_lvl = coords_pyramid[i] + self.delta

            # IMPORTANT: keep original sampler (accuracy-critical)
            corr_sampled = bilinear_sampler(corr, coords_lvl)

            corr_sampled = corr_sampled.view(batch, h1, w1, -1)
            out_pyramid.append(corr_sampled)

        # ---- Concatenate once ----
        out = torch.cat(out_pyramid, dim=-1)

        return out.permute(0, 3, 1, 2).contiguous().float()


### SPEEDUP 1
# class CorrBlock:
#     def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
#         self.num_levels = num_levels
#         self.radius = radius

#         corr = self.corr(fmap1, fmap2)
#         B, H1, W1, _, H2, W2 = corr.shape
#         corr = corr.view(B * H1 * W1, 1, H2, W2)

#         self.corr_pyramid = [corr]
#         for _ in range(1, num_levels):
#             corr = F.avg_pool2d(corr, 2, stride=2, count_include_pad=False)
#             self.corr_pyramid.append(corr)

#         r = radius
#         dy, dx = torch.meshgrid(
#             torch.arange(-r, r+1, device=fmap1.device),
#             torch.arange(-r, r+1, device=fmap1.device),
#             indexing="ij"
#         )
#         self.delta = torch.stack((dy, dx), dim=-1).float()

#     def __call__(self, coords):
#         r = self.radius
#         B, _, H, W = coords.shape

#         coords = coords.permute(0, 2, 3, 1).contiguous()
#         coords_flat = coords.view(-1, 2)

#         out = []

#         for i, corr in enumerate(self.corr_pyramid):
#             centroid = coords_flat[:, None, None, :] / (2 ** i)
#             coords_lvl = centroid + self.delta[None]

#             sampled = bilinear_sampler(corr, coords_lvl)
#             sampled = sampled.view(B, H, W, -1)

#             out.append(sampled)

#         return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous()

#     @staticmethod
#     def corr(fmap1, fmap2):
#         B, C, H, W = fmap1.shape
#         fmap1 = fmap1.view(B, C, -1)
#         fmap2 = fmap2.view(B, C, -1)

#         corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
#         corr = F.relu(corr)

#         return corr.view(B, H, W, 1, H, W)