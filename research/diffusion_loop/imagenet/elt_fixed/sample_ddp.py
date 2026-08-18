from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import AutoencoderKL
from PIL import Image
from tqdm import tqdm

from .diffusion import alpha_sigma, ddpm_posterior_step
from .model import FixedLoopDiT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=512)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vae", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--grid-size", type=int, default=64)
    return parser.parse_args()


def model_from_checkpoint(payload: dict, device: torch.device) -> FixedLoopDiT:
    config = payload["config"]
    architecture = str(config.get("architecture", "fixed_loop_dit"))
    model_class: type[FixedLoopDiT]
    if architecture.startswith("elastic_loop_dit_"):
        from research.diffusion_loop.imagenet.elt_rmdm.model import ElasticLoopDiT

        model_class = ElasticLoopDiT
    elif architecture.startswith("fixed_loop_dit_"):
        model_class = FixedLoopDiT
    else:
        raise ValueError(f"unsupported checkpoint architecture: {architecture}")
    model = model_class(
        latent_size=int(config["latent_size"]),
        patch_size=int(config["patch_size"]),
        channels=int(config["channels"]),
        hidden_size=int(config["hidden_size"]),
        heads=int(config.get("heads", 16)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        unique_blocks=int(config["unique_blocks"]),
        loops=int(config["loops"]),
    ).to(device)
    model.load_state_dict(payload["ema"])
    model.eval().requires_grad_(False)
    return model


def predict_v(
    model: FixedLoopDiT,
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    labels: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    if cfg_scale == 1.0:
        return model(noisy, timestep, labels)
    model_input = torch.cat((noisy, noisy), dim=0)
    timestep_input = torch.cat((timestep, timestep), dim=0)
    label_input = torch.cat((labels, torch.full_like(labels, model.null_class)), dim=0)
    conditional, unconditional = model(model_input, timestep_input, label_input).chunk(2)
    return unconditional + cfg_scale * (conditional - unconditional)


@torch.no_grad()
def sample_latents(
    model: FixedLoopDiT,
    initial_noise: torch.Tensor,
    labels: torch.Tensor,
    *,
    sampling_steps: int,
    cfg_scale: float,
    image_resolution: int,
    noise_resolution: int,
    logsnr_min: float,
    logsnr_max: float,
    generator: torch.Generator,
) -> torch.Tensor:
    state = initial_noise.float()
    batch = state.shape[0]
    schedule = {
        "image_resolution": image_resolution,
        "noise_resolution": noise_resolution,
        "logsnr_min": logsnr_min,
        "logsnr_max": logsnr_max,
    }
    for index in range(sampling_steps - 1, -1, -1):
        timestep = torch.full(
            (batch,), (index + 1) / sampling_steps, device=state.device, dtype=torch.float32
        )
        next_timestep = torch.full(
            (batch,), index / sampling_steps, device=state.device, dtype=torch.float32
        )
        alpha_t, sigma_t = alpha_sigma(timestep, **schedule)
        alpha_s, sigma_s = alpha_sigma(next_timestep, **schedule)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = predict_v(model, state, timestep, labels, cfg_scale)
        step_noise = None
        if index > 0:
            step_noise = torch.randn(
                state.shape, device=state.device, dtype=state.dtype, generator=generator
            )
        state = ddpm_posterior_step(
            state,
            prediction,
            alpha_t,
            sigma_t,
            alpha_s,
            sigma_s,
            noise=step_noise,
        )
    return state


def save_grid(sample_dir: Path, output: Path, count: int) -> None:
    paths = [sample_dir / f"{index:06d}.png" for index in range(count)]
    paths = [path for path in paths if path.exists()]
    if not paths:
        return
    side = math.ceil(math.sqrt(len(paths)))
    with Image.open(paths[0]) as first:
        width, height = first.size
    grid = Image.new("RGB", (side * width, side * height))
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            grid.paste(image.convert("RGB"), ((index % side) * width, (index // side) * height))
    grid.save(output, compress_level=1)


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.batch_size < 1 or args.sampling_steps < 1:
        raise ValueError("sample count, batch size, and sampling steps must be positive")
    if args.cfg_scale < 1.0:
        raise ValueError("CFG scale must be at least one")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    checkpoint_step = int(payload["step"])
    model = model_from_checkpoint(payload, device)
    del payload
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae").to(
        device=device, dtype=torch.bfloat16
    )
    vae.eval().requires_grad_(False)

    args.output.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output / "images"
    sample_dir.mkdir(exist_ok=True)
    global_batch = args.batch_size * world
    batches = math.ceil(args.num_samples / global_batch)
    if rank == 0:
        metadata = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_step": checkpoint_step,
            "architecture": str(config.get("architecture", "unknown")),
            "inference_loops": int(config["loops"]),
            "weights": "ema",
            "num_samples": args.num_samples,
            "sampling_steps": args.sampling_steps,
            "sampler": "ancestral_ddpm",
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "class_distribution": "index_mod_1000",
            "vae": args.vae,
            "world_size": world,
            "batch_size_per_gpu": args.batch_size,
        }
        (args.output / "sampling_config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    iterator = tqdm(range(batches), desc="DDPM sampling") if rank == 0 else range(batches)
    started = time.perf_counter()
    for batch_index in iterator:
        first_index = batch_index * global_batch + rank * args.batch_size
        indices = list(range(first_index, min(first_index + args.batch_size, args.num_samples)))
        if not indices:
            continue
        paths = [sample_dir / f"{index:06d}.png" for index in indices]
        if all(path.exists() for path in paths):
            continue

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + batch_index * world + rank)
        noise = torch.randn(
            (len(indices), int(config["channels"]), int(config["latent_size"]), int(config["latent_size"])),
            device=device,
            generator=generator,
        )
        labels = torch.tensor([index % 1000 for index in indices], device=device)
        latents = sample_latents(
            model,
            noise,
            labels,
            sampling_steps=args.sampling_steps,
            cfg_scale=args.cfg_scale,
            image_resolution=int(config["image_resolution"]),
            noise_resolution=int(config["noise_resolution"]),
            logsnr_min=float(config["logsnr_min"]),
            logsnr_max=float(config["logsnr_max"]),
            generator=generator,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            decoded = vae.decode(latents.to(torch.bfloat16) / float(config["vae_scale"])).sample
        images = (
            (127.5 * decoded.float() + 128.0)
            .clamp(0, 255)
            .permute(0, 2, 3, 1)
            .to(device="cpu", dtype=torch.uint8)
            .numpy()
        )
        for image, path in zip(images, paths, strict=True):
            if not path.exists():
                Image.fromarray(image).save(path, compress_level=1)

        if rank == 0:
            completed = min((batch_index + 1) * global_batch, args.num_samples)
            elapsed = time.perf_counter() - started
            iterator.set_postfix(images=completed, images_per_second=completed / elapsed)

    dist.barrier()
    if rank == 0:
        save_grid(sample_dir, args.output / "grid.png", min(args.grid_size, args.num_samples))
        elapsed = time.perf_counter() - started
        result = {"elapsed_seconds": elapsed, "images_per_second": args.num_samples / elapsed}
        (args.output / "sampling_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
