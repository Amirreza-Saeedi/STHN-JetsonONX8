from __future__ import annotations

import argparse
import os
from pathlib import Path


def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def ensure_local_libcudla(out_dir: Path) -> dict[str, str]:
    """Ensure a local libcudla.so.1 exists for Jetson TensorRT Python imports.

    Some Jetson images provide /lib/aarch64-linux-gnu/libnvcudla.so but not libcudla.so.1.
    Certain TensorRT Python wheels/extensions may be linked against libcudla.so.1.

    We avoid sudo by creating a local symlink and relying on LD_LIBRARY_PATH.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # If the system already has libcudla, nothing to do.
    system_libcudla = _first_existing(
        [
            "/lib/aarch64-linux-gnu/libcudla.so.1",
            "/usr/lib/aarch64-linux-gnu/libcudla.so.1",
            "/usr/local/cuda/lib64/libcudla.so.1",
            "/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudla.so.1",
            "/usr/local/cuda-12/targets/aarch64-linux/lib/libcudla.so.1",
        ]
    )
    if system_libcudla:
        return {
            "status": "ok",
            "message": f"System libcudla found: {system_libcudla}",
            "out_dir": str(out_dir),
        }

    # If we get here, libcudla.so.1 is genuinely missing. Do NOT try to fake it by symlinking
    # libnvcudla -> libcudla; that commonly causes symbol-version errors in libnvinfer.
    return {
        "status": "error",
        "message": (
            "libcudla.so.1 is missing. On Jetson this usually means the CUDLA runtime package is not installed "
            "or is broken. Try: sudo apt-get update && sudo apt-get install -y --reinstall libcudla-12-6 && sudo ldconfig"
        ),
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="libfix",
        help="Directory to write a local libcudla.so.1 symlink (default: libfix)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    result = ensure_local_libcudla(out_dir)
    print(result["message"])

    if result.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
