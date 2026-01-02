import argparse
import os
import shutil
import subprocess
from typing import Optional


def _find_trtexec(explicit_path: Optional[str]) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"trtexec not found at: {explicit_path}")
        return explicit_path

    which = shutil.which("trtexec")
    if which:
        return which

    # Common Jetson / Linux locations (Windows users will likely need to set --trtexec)
    candidates = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/src/tensorrt/bin/trtexec.exe",
        "/usr/bin/trtexec",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find 'trtexec' on PATH. Provide --trtexec /path/to/trtexec."
    )


def main():
    ap = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX model using trtexec (no pycuda).")
    ap.add_argument("--onnx", required=True, help="Path to ONNX model")
    ap.add_argument("--engine", required=True, help="Output .engine path")
    ap.add_argument("--trtexec", default=None, help="Path to trtexec binary (optional if on PATH)")
    ap.add_argument("--fp16", action="store_true", help="Enable FP16")
    ap.add_argument("--workspace", type=int, default=2048, help="Workspace size (MiB)")
    ap.add_argument(
        "--shapes",
        default=None,
        help=(
            "Shapes spec for trtexec, e.g. "
            "min=image1:1x3x256x256,image2:1x3x256x256;"
            "opt=image1:1x3x256x256,image2:1x3x256x256;"
            "max=image1:4x3x256x256,image2:4x3x256x256"
        ),
    )
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    trtexec = _find_trtexec(args.trtexec)
    os.makedirs(os.path.dirname(args.engine) or ".", exist_ok=True)

    cmd = [
        trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        "--explicitBatch",
        f"--workspace={args.workspace}",
    ]

    if args.fp16:
        cmd.append("--fp16")

    if args.verbose:
        cmd.append("--verbose")

    # Parse optional min/opt/max shapes
    if args.shapes:
        parts = [p.strip() for p in args.shapes.split(";") if p.strip()]
        for part in parts:
            if part.startswith("min="):
                cmd.append(f"--minShapes={part[len('min='):]}")
            elif part.startswith("opt="):
                cmd.append(f"--optShapes={part[len('opt='):]}")
            elif part.startswith("max="):
                cmd.append(f"--maxShapes={part[len('max='):]}")
            else:
                raise ValueError("--shapes must be in 'min=...;opt=...;max=...' format")

    print("Running:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)
    print(f"Wrote: {args.engine}")


if __name__ == "__main__":
    main()
