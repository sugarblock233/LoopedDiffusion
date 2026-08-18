#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.diffusion_loop.imagenet.elt_fixed.model import FixedLoopDiT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--minimum-step", type=int, default=200_000)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = {"config", "ema", "model", "optimizer", "step"}
    if set(payload) != required:
        raise SystemExit(f"unexpected checkpoint keys: {sorted(payload)}")
    step = int(payload["step"])
    if step < args.minimum_step:
        raise SystemExit(f"expected step >= {args.minimum_step}, found {step}")

    config = payload["config"]
    if config.get("architecture") != "fixed_loop_dit_2n12l_d1024":
        raise SystemExit(f"unexpected architecture: {config.get('architecture')}")
    model = FixedLoopDiT(
        latent_size=int(config["latent_size"]),
        patch_size=int(config["patch_size"]),
        channels=int(config["channels"]),
        hidden_size=int(config["hidden_size"]),
        heads=int(config["heads"]),
        mlp_ratio=float(config["mlp_ratio"]),
        unique_blocks=int(config["unique_blocks"]),
        loops=int(config["loops"]),
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 42_250_256:
        raise SystemExit(f"unexpected parameter count: {parameters}")
    model.load_state_dict(payload["model"], strict=True)
    model.load_state_dict(payload["ema"], strict=True)
    optimizer_states = len(payload["optimizer"].get("state", {}))
    if optimizer_states == 0:
        raise SystemExit("optimizer state is empty")

    print(
        json.dumps(
            {
                "architecture": config["architecture"],
                "checkpoint": str(args.checkpoint.resolve()),
                "ema_tensors": len(payload["ema"]),
                "model_tensors": len(payload["model"]),
                "optimizer_states": optimizer_states,
                "parameters": parameters,
                "step": step,
            },
            sort_keys=True,
        )
    )
    del model, payload
    gc.collect()


if __name__ == "__main__":
    main()
