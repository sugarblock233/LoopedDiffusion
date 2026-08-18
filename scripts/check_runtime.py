#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-h200", action="store_true")
    parser.add_argument("--visible-gpus", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch")
    count = torch.cuda.device_count()
    if count != args.visible_gpus:
        raise SystemExit(f"expected {args.visible_gpus} visible GPU, found {count}")

    properties = torch.cuda.get_device_properties(0)
    name = properties.name
    if args.require_h200 and "H200" not in name.upper():
        raise SystemExit(f"expected an H200, found {name}")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit(f"BF16 is not supported by {name}")

    left = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    right = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    result = left @ right
    torch.cuda.synchronize()
    if not torch.isfinite(result.float()).all():
        raise SystemExit("BF16 CUDA matmul produced non-finite output")

    print(
        json.dumps(
            {
                "bf16": True,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "cuda": torch.version.cuda,
                "device": name,
                "memory_gib": round(properties.total_memory / 1024**3, 2),
                "torch": torch.__version__,
                "visible_gpus": count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
