import torch
import parser
from model.network import STHN

def export_onnx(args):
    sthn = STHN(args)

    ckpt = torch.load(
        "js_models/1536_one_stage/STHN.pth",
        map_location="cuda"
    )

    state = ckpt["netG"] if "netG" in ckpt else ckpt
    sthn.netG.load_state_dict(state, strict=False)

    net = sthn.netG
    net.eval().cuda()

    # TWO inputs!
    img1 = torch.randn(1, 3, 256, 256, device="cuda")
    img2 = torch.randn(1, 3, 256, 256, device="cuda")

    torch.onnx.export(
        net,
        (img1, img2),
        "sthn_netG.onnx",
        opset_version=17,
        input_names=["image1", "image2"],
        output_names=["four_preds", "four_pred"],
        dynamic_axes={
            "image1": {0: "batch"},
            "image2": {0: "batch"},
            "four_pred": {0: "batch"}
        }
    )

if __name__ == "__main__":
    args = parser.parse_arguments()
    export_onnx(args)
