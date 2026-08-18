from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import AutoencoderKL
from torchvision.io import decode_jpeg

from .data import iter_parquet_encoded_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vae", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    device = torch.device("cuda", local_rank)
    args.output.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.input.glob("data/train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no ImageNet parquet shards under {args.input}")

    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae").to(device, torch.bfloat16).eval()
    vae.requires_grad_(False)
    completed = 0
    for shard_index, shard in enumerate(shards):
        if shard_index % world != rank:
            continue
        output = args.output / f"train-{shard_index:04d}.pt"
        if output.exists():
            completed += 1
            continue
        moments = []
        labels = []
        with torch.inference_mode():
            for encoded_images, batch_labels in iter_parquet_encoded_batches(
                shard, args.batch_size
            ):
                byte_tensors = [
                    torch.frombuffer(bytearray(image), dtype=torch.uint8) for image in encoded_images
                ]
                decoded = decode_jpeg(byte_tensors, mode="RGB", device=device)
                images = torch.stack(decoded).to(torch.bfloat16).div_(127.5).sub_(1.0)
                posterior = vae.encode(images).latent_dist
                moments.append(torch.cat((posterior.mean, posterior.logvar), dim=1).half().cpu())
                labels.append(batch_labels)
        temporary = output.with_suffix(".pt.tmp")
        shard_labels = torch.cat(labels)
        torch.save(
            {"moments": torch.cat(moments), "labels": shard_labels},
            temporary,
        )
        temporary.replace(output)
        completed += 1
        print(
            f"rank={rank} cached={completed} shard={shard.name} samples={len(shard_labels)}",
            flush=True,
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
