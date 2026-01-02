import argparse
import os
import shutil
import subprocess
from typing import Optional


def _normalize_path(path: str) -> str:
    # The notebook historically used Windows-style paths (e.g., trt\one_stage\x.onnx).
    # On Linux, backslashes are literal characters and will break file resolution.
    if os.sep == "/":
        path = path.replace("\\", "/")
    return path


def _trtexec_help(trtexec_path: str) -> str:
    # trtexec flag names differ across TensorRT versions.
    # We probe --help once and select compatible flags.
    try:
        result = subprocess.run(
            [trtexec_path, "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout or ""
    except Exception:
        return ""


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

    # Pre-flight: TensorRT 10.x plugin library on Jetson can depend on cuDLA.
    # If it's missing, trtexec fails before parsing the ONNX.
    if os.name == "posix" and os.uname().machine in {"aarch64", "arm64"}:
        cudla_candidates = [
            "/lib/aarch64-linux-gnu/libcudla.so.1",
            "/usr/lib/aarch64-linux-gnu/libcudla.so.1",
            "/usr/lib/aarch64-linux-gnu/tegra/libcudla.so.1",
        ]
        if not any(os.path.exists(p) for p in cudla_candidates):
            print(
                "WARNING: libcudla.so.1 not found. TensorRT may fail to load libnvinfer_plugin.\n"
                "On Jetson/Ubuntu, install the cuDLA runtime, e.g.:\n"
                "  sudo apt-get update && sudo apt-get install -y libcudla-12-6\n"
                "Then run: sudo ldconfig\n"
            )

    onnx_path = _normalize_path(args.onnx)
    engine_path = _normalize_path(args.engine)
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)

    help_text = _trtexec_help(trtexec)
    supports_explicit_batch = "--explicitBatch" in help_text
    supports_workspace = "--workspace" in help_text
    supports_mem_pool = "--memPoolSize" in help_text

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]

    # TensorRT 10+ removed some legacy flags.
    if supports_explicit_batch:
        cmd.append("--explicitBatch")

    if supports_mem_pool:
        # Default unit is MiB per trtexec help.
        cmd.append(f"--memPoolSize=workspace:{args.workspace}")
    elif supports_workspace:
        cmd.append(f"--workspace={args.workspace}")
    else:
        # No known workspace flag; proceed without setting it.
        pass

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
    print(f"Wrote: {engine_path}")


if __name__ == "__main__":
    main()
