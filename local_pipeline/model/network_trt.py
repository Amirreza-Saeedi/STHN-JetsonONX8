from __future__ import annotations

from typing import List, Optional, Tuple

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Failed to import PyTorch. This TensorRT inference path uses torch CUDA tensors as IO buffers (no pycuda). "
        "If you see missing 'libcufile.so.0', install it: sudo apt-get install -y libcufile-12-6 && sudo ldconfig"
    ) from e
import torch.nn as nn
import torch.nn.functional as F

from model.trt_engine import TrtEngine


class IHNTRT(nn.Module):
    def __init__(
        self,
        engine_path: str,
        *,
        image1_tensor_name: str,
        image2_tensor_name: str = "image2",
        output_tensor_name: str = "four_point_disp",
    ):
        super().__init__()
        self.engine_path = engine_path
        self.image1_tensor_name = image1_tensor_name
        self.image2_tensor_name = image2_tensor_name
        self.output_tensor_name = output_tensor_name
        self.engine = TrtEngine(engine_path)

    def forward(
        self,
        *,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters_lev0: int = 1,
        **_kwargs,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        outs = self.engine.infer(
            {
                self.image1_tensor_name: image1,
                self.image2_tensor_name: image2,
            }
        )
        disp = outs[self.output_tensor_name]
        # Mimic original IHN return type: (predictions list, final prediction)
        preds = [disp] * max(int(iters_lev0), 1)
        return preds, disp


class STHNTRT:
    def __init__(
        self,
        args,
        *,
        engine_coarse: str,
        engine_fine: Optional[str] = None,
    ):
        super().__init__()
        self.args = args
        self.device = args.device

        self.four_point_org_single = torch.zeros((1, 2, 2, 2)).to(self.device)
        self.four_point_org_single[:, :, 0, 0] = torch.tensor([0, 0], device=self.device)
        self.four_point_org_single[:, :, 0, 1] = torch.tensor([self.args.resize_width - 1, 0], device=self.device)
        self.four_point_org_single[:, :, 1, 0] = torch.tensor([0, self.args.resize_width - 1], device=self.device)
        self.four_point_org_single[:, :, 1, 1] = torch.tensor([
            self.args.resize_width - 1,
            self.args.resize_width - 1,
        ], device=self.device)

        self.four_point_org_large_single = torch.zeros((1, 2, 2, 2)).to(self.device)
        self.four_point_org_large_single[:, :, 0, 0] = torch.tensor([0, 0], device=self.device)
        self.four_point_org_large_single[:, :, 0, 1] = torch.tensor([self.args.database_size - 1, 0], device=self.device)
        self.four_point_org_large_single[:, :, 1, 0] = torch.tensor([0, self.args.database_size - 1], device=self.device)
        self.four_point_org_large_single[:, :, 1, 1] = torch.tensor([
            self.args.database_size - 1,
            self.args.database_size - 1,
        ], device=self.device)

        self.netG = IHNTRT(engine_coarse, image1_tensor_name="image1")

        if self.args.two_stages:
            if not engine_fine:
                raise ValueError("two_stages is set but engine_fine was not provided")
            self.netG_fine = IHNTRT(engine_fine, image1_tensor_name="image1_crop")

    def setup(self):
        # Keep API parity with the PyTorch model class.
        return

    def set_input(self, A: torch.Tensor, B: torch.Tensor, flow_gt=None):
        self.image_1_ori = A.to(self.device, non_blocking=True)
        self.image_2 = B.to(self.device, non_blocking=True)
        self.real_warped_image_2 = None
        self.image_1 = F.interpolate(
            self.image_1_ori,
            size=self.args.resize_width,
            mode="bilinear",
            align_corners=True,
            antialias=True,
        )

    def forward(self, for_training: bool = False):
        self.four_preds_list, self.four_pred = self.netG(
            image1=self.image_1,
            image2=self.image_2,
            iters_lev0=self.args.iters_lev0,
            corr_level=self.args.corr_level,
        )

        if self.args.two_stages:
            self.image_1_crop, delta, flow_bbox = self.get_cropped_st_images(
                self.image_1_ori,
                self.four_pred,
                self.args.fine_padding,
                self.args.detach,
                self.args.augment_two_stages,
            )
            self.image_2_crop = self.image_2
            self.four_preds_list_fine, self.four_pred_fine = self.netG_fine(
                image1=self.image_1_crop,
                image2=self.image_2_crop,
                iters_lev0=self.args.iters_lev1,
                corr_level=2,
            )
            self.four_preds_list, self.four_pred = self.combine_coarse_fine(
                self.four_preds_list,
                self.four_pred,
                self.four_preds_list_fine,
                self.four_pred_fine,
                delta,
                flow_bbox,
                for_training,
            )

    def get_cropped_st_images(self, image_1_ori, four_pred, fine_padding, detach=True, augment_two_stages=0):
        # Adapted from local_pipeline/model/network.py.
        # IMPORTANT: implemented without kornia/torch.linalg to avoid loading libtorch_cuda_linalg
        # (some JetPack/PyTorch combinations fail to resolve cuSOLVER symbols).
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
            w = torch.max(torch.stack([right - left, bottom - top], dim=1), dim=1)[0]
            c = torch.stack([(left + right) / 2, (bottom + top) / 2], dim=1)
            if self.args.augment_type == "center":
                w += torch.rand(w.shape).to(w.device) * augment_two_stages
                c += (torch.rand(c.shape).to(c.device) * 2 - 1) * augment_two_stages
        else:
            w = torch.max(torch.stack([right - left, bottom - top], dim=1), dim=1)[0]
            c = torch.stack([(left + right) / 2, (bottom + top) / 2], dim=1)

        w_padded = w + 2 * fine_padding
        crop_top_left = c + torch.stack([-w_padded / 2, -w_padded / 2], dim=1)
        x_start = crop_top_left[:, 0]
        y_start = crop_top_left[:, 1]

        delta = (w_padded / self.args.resize_width).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # Build a sampling grid for grid_sample.
        # image_1_ori: (B, C, H, W), crop is (resize_width, resize_width)
        B, C, H, W = image_1_ori.shape
        out_hw = int(self.args.resize_width)

        # Generate normalized [0, 1] coordinates and scale to absolute pixel coords.
        lin = torch.linspace(0.0, 1.0, out_hw, device=image_1_ori.device, dtype=image_1_ori.dtype)
        gx = lin.view(1, 1, out_hw).expand(B, out_hw, out_hw)  # (B, out, out)
        gy = lin.view(1, out_hw, 1).expand(B, out_hw, out_hw)

        x0 = x_start.view(B, 1, 1)
        y0 = y_start.view(B, 1, 1)
        ww = w_padded.view(B, 1, 1)

        x_abs = x0 + gx * ww
        y_abs = y0 + gy * ww

        # Convert to grid_sample normalized coords in [-1, 1]
        x_norm = (x_abs / (W - 1)) * 2 - 1
        y_norm = (y_abs / (H - 1)) * 2 - 1
        grid = torch.stack([x_norm, y_norm], dim=-1)  # (B, out, out, 2)

        image_1_crop = F.grid_sample(
            image_1_ori,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # Construct bbox corners in the same 2x2 layout as four_point_org_large_single.
        # (B, 2, 2, 2): [x/y, row, col]
        four_cor_bbox = torch.zeros((B, 2, 2, 2), device=image_1_ori.device, dtype=image_1_ori.dtype)
        four_cor_bbox[:, 0, 0, 0] = x_start
        four_cor_bbox[:, 1, 0, 0] = y_start
        four_cor_bbox[:, 0, 0, 1] = x_start + w_padded
        four_cor_bbox[:, 1, 0, 1] = y_start
        four_cor_bbox[:, 0, 1, 0] = x_start
        four_cor_bbox[:, 1, 1, 0] = y_start + w_padded
        four_cor_bbox[:, 0, 1, 1] = x_start + w_padded
        four_cor_bbox[:, 1, 1, 1] = y_start + w_padded

        flow_bbox = four_cor_bbox - self.four_point_org_large_single

        if detach:
            image_1_crop = image_1_crop.detach()
            delta = delta.detach()
            flow_bbox = flow_bbox.detach()

        return image_1_crop, delta, flow_bbox

    def combine_coarse_fine(self, four_preds_list, four_pred, four_preds_list_fine, four_pred_fine, delta, flow_bbox, for_training):
        alpha = self.args.database_size / self.args.resize_width
        kappa = delta / alpha
        four_preds_list_fine = [fp * kappa + flow_bbox / alpha for fp in four_preds_list_fine]
        four_pred_fine = four_pred_fine * kappa + flow_bbox / alpha
        four_preds_list = four_preds_list + four_preds_list_fine
        return four_preds_list, four_pred_fine
