from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .data import LatentShardDataset
from .diffusion import shifted_cosine_logsnr
from .model import FixedLoopDiT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--global-batch", type=int, default=512)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-drop", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--vae-scale", type=float, default=0.18215)
    parser.add_argument("--image-resolution", type=int, default=256)
    parser.add_argument("--noise-resolution", type=int, default=64)
    parser.add_argument("--logsnr-min", type=float, default=-15.0)
    parser.add_argument("--logsnr-max", type=float, default=15.0)
    parser.add_argument("--sigmoid-bias", type=float, default=-1.0)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--unique-blocks", type=int, default=2)
    parser.add_argument("--loops", type=int, default=12)
    parser.add_argument("--activation-checkpoint-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument(
        "--save-final", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def update_latest(checkpoint: Path, latest: Path) -> None:
    temporary = latest.with_name(latest.name + ".tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(checkpoint.name)
    temporary.replace(latest)


def sigmoid_elbo_from_v(
    prediction: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    logsnr: torch.Tensor,
    dlogsnr_dt: torch.Tensor,
    *,
    bias: float,
) -> torch.Tensor:
    """SiD2 sigmoid-weighted x0 loss while retaining ELT's v parameterization."""
    predicted_clean = alpha * noisy - sigma * prediction.float()
    squared_error = (predicted_clean - clean).square().flatten(1).mean(1)
    weight = -0.5 * dlogsnr_dt * math.exp(bias) * torch.sigmoid(logsnr - bias)
    return (weight * squared_error).mean()


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    source = model.module if isinstance(model, DDP) else model
    ema_parameters = tuple(ema.parameters())
    source_parameters = tuple(source.parameters())
    if len(ema_parameters) != len(source_parameters):
        raise ValueError("EMA and source parameter counts do not match")
    for target, current in zip(ema_parameters, source_parameters):
        target.mul_(decay).add_(current.detach(), alpha=1.0 - decay)
    ema_buffers = tuple(ema.buffers())
    source_buffers = tuple(source.buffers())
    if len(ema_buffers) != len(source_buffers):
        raise ValueError("EMA and source buffer counts do not match")
    for target, current in zip(ema_buffers, source_buffers):
        target.copy_(current)


def learning_rate(step: int, *, base: float, warmup: int) -> float:
    return base * min(1.0, (step + 1) / max(warmup, 1))


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.global_batch % (args.micro_batch * world):
        raise ValueError("global batch must be divisible by micro batch times world size")
    accumulation = args.global_batch // (args.micro_batch * world)
    args.output.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    model = FixedLoopDiT(
        latent_size=args.latent_size,
        patch_size=args.patch_size,
        channels=args.channels,
        hidden_size=args.hidden_size,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        unique_blocks=args.unique_blocks,
        loops=args.loops,
        activation_checkpoint_every=args.activation_checkpoint_every,
    ).to(device)
    ema = copy.deepcopy(model).to(device).eval()
    ema.requires_grad_(False)
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
        fused=True,
    )
    start_step = 0
    data_seed = args.seed
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.module.load_state_dict(payload["model"])
        ema.load_state_dict(payload["ema"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        data_seed = args.seed + start_step
        torch.manual_seed(data_seed + rank)
        torch.cuda.manual_seed_all(data_seed + rank)

    dataset = LatentShardDataset(args.data, seed=data_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    iterator = iter(loader)
    config = vars(args).copy()
    config.update(model.module.metadata())
    config.update(
        {
            "gpus_per_node": int(os.environ.get("LOCAL_WORLD_SIZE", world)),
            "nnodes": int(os.environ.get("NNODES", "1")),
            "world_size": world,
            "gradient_accumulation": accumulation,
            "data_seed": data_seed,
            "start_step": start_step,
        }
    )
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    if rank == 0:
        (args.output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        print(json.dumps(config, sort_keys=True), flush=True)

    running_loss = 0.0
    running_start = time.perf_counter()
    for step in range(start_step, args.steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device)
        for micro_step in range(accumulation):
            moments, labels = next(iterator)
            moments = moments.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mean, logvar = moments.chunk(2, dim=1)
            clean = (mean + torch.exp(0.5 * logvar.clamp(-30.0, 20.0)) * torch.randn_like(mean))
            clean.mul_(args.vae_scale)
            drop = torch.rand(labels.shape, device=device) < args.label_drop
            labels = torch.where(drop, model.module.null_class, labels)
            timestep = torch.rand(labels.shape[0], device=device)
            logsnr, dlogsnr_dt = shifted_cosine_logsnr(
                timestep,
                image_resolution=args.image_resolution,
                noise_resolution=args.noise_resolution,
                logsnr_min=args.logsnr_min,
                logsnr_max=args.logsnr_max,
            )
            alpha = torch.sigmoid(logsnr).sqrt().reshape(-1, 1, 1, 1)
            sigma = torch.sigmoid(-logsnr).sqrt().reshape(-1, 1, 1, 1)
            noise = torch.randn_like(clean)
            noisy = alpha * clean + sigma * noise
            sync = micro_step == accumulation - 1
            context = nullcontext() if sync else model.no_sync()
            with context, torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(noisy, timestep, labels)
                loss = sigmoid_elbo_from_v(
                    prediction,
                    clean,
                    noisy,
                    alpha,
                    sigma,
                    logsnr,
                    dlogsnr_dt,
                    bias=args.sigmoid_bias,
                ) / accumulation
            loss.backward()
            step_loss += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = learning_rate(step, base=args.learning_rate, warmup=args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        update_ema(ema, model, args.ema_decay)
        dist.all_reduce(step_loss, op=dist.ReduceOp.SUM)
        step_loss /= world
        running_loss += float(step_loss)

        completed = step + 1
        if rank == 0 and completed % args.log_every == 0:
            elapsed = time.perf_counter() - running_start
            average_loss = running_loss / args.log_every
            step_seconds = elapsed / args.log_every
            record = {
                "step": completed,
                "loss": average_loss,
                "lr": lr,
                "step_seconds": step_seconds,
                "images_per_second": args.global_batch / step_seconds,
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            }
            with (args.output / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            running_loss = 0.0
            running_start = time.perf_counter()

        should_save = completed % args.checkpoint_every == 0 or (
            completed == args.steps and args.save_final
        )
        if should_save:
            dist.barrier()
            if rank == 0:
                payload = {
                    "step": completed,
                    "config": config,
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                checkpoint = args.output / f"checkpoint-{completed:07d}.pt"
                atomic_save(payload, checkpoint)
                update_latest(checkpoint, args.output / "latest.pt")
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
