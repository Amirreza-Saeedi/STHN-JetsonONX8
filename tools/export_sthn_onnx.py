import argparse
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import sys
sys.path[0] = 'd:\\Robotic Perceptoin Lab\\map-matching\\STHN-JetsonONX8\\local_pipeline'

def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Matches the cleanup logic used in local_pipeline/my_myevaluate.py
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


@dataclass
class ExportConfig:
    resize_width: int = 256
    corr_level: int = 4
    iters: int = 6
    mixed_precision: bool = False
    fnet_cat: bool = False
    lev0: bool = True
    weight: bool = False


class _Args:
    """Minimal args object expected by local_pipeline.model.network_cpu.IHN."""

    def __init__(
        self,
        *,
        resize_width: int,
        corr_level: int,
        iters_lev0: int,
        iters_lev1: int,
        mixed_precision: bool,
        fnet_cat: bool,
        lev0: bool,
        weight: bool,
        device: str,
        database_size: int,
        two_stages: bool,
        fine_padding: int = 0,
        detach: bool = False,
        augment_two_stages: float = 0.0,
        arch: str = "IHN",
        gpuid=None,
        restore_ckpt=None,
        finetune: bool = False,
    ):
        self.resize_width = resize_width
        self.corr_level = corr_level
        self.iters_lev0 = iters_lev0
        self.iters_lev1 = iters_lev1
        self.mixed_precision = mixed_precision
        self.fnet_cat = fnet_cat
        self.lev0 = lev0
        self.weight = weight
        self.device = device
        self.database_size = database_size
        self.two_stages = two_stages
        self.fine_padding = fine_padding
        self.detach = detach
        self.augment_two_stages = augment_two_stages
        self.arch = arch
        self.gpuid = gpuid or [0]
        self.restore_ckpt = restore_ckpt
        self.finetune = finetune
        # Fields referenced elsewhere in the codebase, but not needed for export
        self.G_loss_lambda = 1.0
        self.gamma = 0.85


class STHNCoarseONNX(nn.Module):
    def __init__(self, net: nn.Module, *, iters: int, corr_level: int):
        super().__init__()
        self.net = net
        self.iters = iters
        self.corr_level = corr_level

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        _preds, four_point_disp = self.net(
            image1=image1,
            image2=image2,
            iters_lev0=self.iters,
            corr_level=self.corr_level,
        )
        return four_point_disp


class STHNFineONNX(nn.Module):
    def __init__(self, net: nn.Module, *, iters: int):
        super().__init__()
        self.net = net
        self.iters = iters

    def forward(self, image1_crop: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        _preds, four_point_disp = self.net(
            image1=image1_crop,
            image2=image2,
            iters_lev0=self.iters,
            corr_level=2,
        )
        return four_point_disp


def _load_checkpoint(pth_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(pth_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")
    return checkpoint


def _build_nets(config: ExportConfig, *, two_stages: bool, database_size: int) -> Tuple[nn.Module, Optional[nn.Module], _Args]:
    # Use the CPU version to avoid CUDA dependency during export.
    # from local_pipeline.model.network_cpu import IHN
    from model.network_cpu import IHN

    args = _Args(
        resize_width=config.resize_width,
        corr_level=config.corr_level,
        iters_lev0=config.iters,
        iters_lev1=config.iters,
        mixed_precision=config.mixed_precision,
        fnet_cat=config.fnet_cat,
        lev0=config.lev0,
        weight=config.weight,
        device="cpu",
        database_size=database_size,
        two_stages=two_stages,
    )

    netG = IHN(args, True)
    netG_fine = None
    if two_stages:
        # Fine stage uses corr_level=2 in training/eval code.
        # This must be set at construction time because it affects the update block
        # channel dimensions (state_dict shapes).
        args_for_fine = _Args(
            resize_width=config.resize_width,
            corr_level=2,
            iters_lev0=config.iters,
            iters_lev1=config.iters,
            mixed_precision=config.mixed_precision,
            fnet_cat=config.fnet_cat,
            lev0=config.lev0,
            weight=config.weight,
            device="cpu",
            database_size=database_size,
            two_stages=True,
        )
        netG_fine = IHN(args_for_fine, False)
    return netG, netG_fine, args


def _export_onnx(
    model: nn.Module,
    onnx_out: str,
    *,
    input_shape: Tuple[int, int, int, int],
    input_names: Tuple[str, str],
    output_name: str,
    dynamic_batch: bool,
    dynamo: bool,
    opset: int,
):
    model.eval()

    b, c, h, w = input_shape
    dummy1 = torch.randn(b, c, h, w, dtype=torch.float32)
    dummy2 = torch.randn(b, c, h, w, dtype=torch.float32)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            input_names[0]: {0: "batch"},
            input_names[1]: {0: "batch"},
            output_name: {0: "batch"},
        }

    os.makedirs(os.path.dirname(onnx_out) or ".", exist_ok=True)

    # PyTorch 2.9+ defaults to dynamo=True, which routes through torch.export.
    # This model contains data-dependent control flow (e.g., isnan checks), which
    # torch.export cannot trace. For ONNX export we default to dynamo=False.
    torch.onnx.export(
        model,
        (dummy1, dummy2),
        onnx_out,
        input_names=list(input_names),
        output_names=[output_name],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=dynamo,
    )


def main():
    parser = argparse.ArgumentParser(description="Export STHN .pth to ONNX (coarse and/or fine).")
    parser.add_argument("--pth", required=True, help="Path to STHN .pth checkpoint")
    parser.add_argument("--out_dir", required=True, help="Directory to write ONNX files")
    parser.add_argument(
        "--stage",
        choices=["coarse", "fine", "both", "auto"],
        default="auto",
        help="Which stage to export. 'auto' exports coarse, and fine if checkpoint has netG_fine.",
    )
    parser.add_argument("--resize_width", type=int, default=256)
    parser.add_argument("--corr_level", type=int, default=4)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--database_size", type=int, default=1536)
    parser.add_argument("--dynamic_batch", action="store_true", help="Export with dynamic batch axis")
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use PyTorch dynamo/torch.export based ONNX exporter (can fail for tensor-dependent control flow).",
    )
    parser.add_argument("--opset", type=int, default=18)

    args_cli = parser.parse_args()

    ckpt = _load_checkpoint(args_cli.pth)
    has_fine = "netG_fine" in ckpt

    stage = args_cli.stage
    if stage == "auto":
        stage = "both" if has_fine else "coarse"

    config = ExportConfig(
        resize_width=args_cli.resize_width,
        corr_level=args_cli.corr_level,
        iters=args_cli.iters,
        mixed_precision=False,
        fnet_cat=False,
        lev0=True,
        weight=False,
    )

    two_stages = stage in ("fine", "both")
    netG, netG_fine, _model_args = _build_nets(config, two_stages=two_stages, database_size=args_cli.database_size)

    if "netG" not in ckpt:
        raise KeyError("Checkpoint does not contain key 'netG'")

    netG.load_state_dict(_clean_state_dict(ckpt["netG"]), strict=False)

    if stage in ("coarse", "both"):
        coarse = STHNCoarseONNX(netG, iters=config.iters, corr_level=config.corr_level)
        coarse_onnx = os.path.join(args_cli.out_dir, "sthn_coarse.onnx")
        _export_onnx(
            coarse,
            coarse_onnx,
            input_shape=(1, 3, config.resize_width, config.resize_width),
            input_names=("image1", "image2"),
            output_name="four_point_disp",
            dynamic_batch=args_cli.dynamic_batch,
            dynamo=args_cli.dynamo,
            opset=args_cli.opset,
        )
        print(f"Wrote: {coarse_onnx}")

    if stage in ("fine", "both"):
        if not has_fine:
            raise KeyError("Requested fine export, but checkpoint has no 'netG_fine'.")
        assert netG_fine is not None
        netG_fine.load_state_dict(_clean_state_dict(ckpt["netG_fine"]), strict=False)
        fine = STHNFineONNX(netG_fine, iters=config.iters)
        fine_onnx = os.path.join(args_cli.out_dir, "sthn_fine.onnx")
        _export_onnx(
            fine,
            fine_onnx,
            input_shape=(1, 3, config.resize_width, config.resize_width),
            input_names=("image1_crop", "image2"),
            output_name="four_point_disp",
            dynamic_batch=args_cli.dynamic_batch,
            dynamo=args_cli.dynamo,
            opset=args_cli.opset,
        )
        print(f"Wrote: {fine_onnx}")


if __name__ == "__main__":
    main()
