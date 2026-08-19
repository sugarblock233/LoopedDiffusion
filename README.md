# LoopedDiffusion fixed baseline

This repository contains the fixed-loop ImageNet-256 latent-diffusion baseline
and a full 200K-step training checkpoint. The checkpoint is distributed as a
GitHub Release asset because its 647 MiB size exceeds GitHub's normal file
limit.

## Model and result

- architecture: two unique DiT blocks repeated for 12 loops;
- effective depth: 24 blocks;
- width / heads / MLP ratio: 1024 / 16 / 4;
- stored denoiser parameters: 42,250,256;
- objective: final-output v-prediction with sigmoid-ELBO bias `-1`;
- global batch: 512, AdamW, BF16, EMA 0.9999;
- VAE: frozen SD 1.4 VAE; the VAE is not trained with the denoiser.

The completed ADM FID-50K measurements use EMA weights, DDPM-512, CFG 3.0,
and ImageNet-256 reference statistics:

| Training step | FID-50K |
| ---: | ---: |
| 50K | 54.1643 |
| 100K | 26.5246 |
| 200K | 15.8243 |

The Release checkpoint contains `model`, `ema`, `optimizer`, `config`, and
`step=200000`, so it can continue training rather than only run evaluation.

## 1. Environment

Clone the repository on the cluster and load its CUDA/PyTorch module first.
The PyTorch build must support the cluster's H200 driver. Do not copy CUDA
libraries from another server.

```bash
git clone https://github.com/sugarblock233/LoopedDiffusion.git
cd LoopedDiffusion

# Cluster-specific examples; use the modules provided by the cluster.
module load cuda
module load python

bash scripts/setup_env.sh
python scripts/check_runtime.py --require-h200
```

`setup_env.sh` creates `.venv` with access to system site packages. If the
cluster module does not provide PyTorch and torchvision, provide the cluster's
matching wheel index:

```bash
TORCH_INDEX_URL='<cluster-compatible PyTorch CUDA wheel index>' \
  bash scripts/setup_env.sh
```

## 2. Download the 200K checkpoint

```bash
bash scripts/download_checkpoint.sh
python scripts/validate_checkpoint.py checkpoints/checkpoint-0200000.pt
```

The expected SHA-256 is recorded in `checkpoints/SHA256SUMS`. The download is
resumable.

The 210K, 215K, and 220K continuation checkpoints are available from the same
release. Select one by step, then pass it as the initial checkpoint when
starting a new run directory:

```bash
CHECKPOINT_STEP=220000 bash scripts/download_checkpoint.sh
python scripts/validate_checkpoint.py checkpoints/checkpoint-0220000.pt

INITIAL_CHECKPOINT="$PWD/checkpoints/checkpoint-0220000.pt" \
DATA_ROOT="$SCRATCH/looped-diffusion/imagenet_sd14_latents" \
RUN_ROOT="$SCRATCH/looped-diffusion/runs/fixed_2n12l_d1024_seed0" \
  sbatch --partition=<h200-partition> scripts/train_h200_single.sbatch
```

Use `CHECKPOINT_STEP=210000` or `CHECKPOINT_STEP=215000` to select an earlier
continuation checkpoint. If `RUN_ROOT/latest.pt` already exists, the launcher
intentionally prefers that newer run-local checkpoint over
`INITIAL_CHECKPOINT`.

## 3. Prepare ImageNet latent shards

Training reads cached SD 1.4 posterior moments rather than running the VAE in
the training loop. The cache consists of 40 shards and occupies about 20 GiB.
The source 256px ImageNet parquet files occupy about 17 GiB. Put both on local
scratch or fast parallel storage, not a slow home-directory NFS mount.

```bash
RAW_ROOT="$SCRATCH/looped-diffusion/imagenet_256_parquet" \
LATENT_ROOT="$SCRATCH/looped-diffusion/imagenet_sd14_latents" \
  bash scripts/prepare_imagenet_latents.sh
```

The downloader uses the public
`benjamin-paine/imagenet-1k-256x256` Hugging Face dataset. Set `HF_TOKEN` first
if the destination environment requires authentication. Existing parquet and
latent shards are reused.

## 4. Submit one H200

Create the Slurm output directory before submission and pass the cluster's
partition, account, or H200 constraint on the `sbatch` command line:

```bash
mkdir -p slurm-logs

DATA_ROOT="$SCRATCH/looped-diffusion/imagenet_sd14_latents" \
RUN_ROOT="$SCRATCH/looped-diffusion/runs/fixed_2n12l_d1024_seed0" \
  sbatch --partition=<h200-partition> scripts/train_h200_single.sbatch
```

Some clusters select the GPU type with `--constraint=h200` or
`--gres=gpu:h200:1` instead. Check `sinfo` or ask the administrator rather than
adding incompatible directives to the generic batch file.

Defaults:

- resume from `checkpoints/checkpoint-0200000.pt`;
- train to 400K steps;
- one visible H200;
- global batch 512;
- micro-batch 512 and one optimizer pass;
- selective activation checkpointing every four recurrent block calls;
- save every 5K steps for Slurm preemption recovery.

The run directory's `latest.pt` is preferred automatically after the first new
checkpoint. Re-submitting the same command therefore resumes the newest saved
step. A single H200 keeps the same global batch and optimizer state, but a
different GPU world size is not bitwise identical to the original four-A100
trajectory.

For a short operational test:

```bash
DATA_ROOT="$SCRATCH/looped-diffusion/imagenet_sd14_latents" \
RUN_ROOT="$SCRATCH/looped-diffusion/runs/fixed_smoke" \
TARGET_STEPS=200100 CHECKPOINT_EVERY=100 \
  sbatch --partition=<h200-partition> scripts/train_h200_single.sbatch
```

After the smoke test, use a fresh official run directory starting from the
unaltered 200K checkpoint. The high-occupancy default uses real activations,
not an unused reservation tensor, and is expected to peak around 105--125 GiB
on an H200. The exact value depends on the PyTorch/CUDA build and is recorded
as `peak_memory_gib` in `train.jsonl` after the first 50 steps.

`ACTIVATION_CHECKPOINT_EVERY` controls the memory/compute tradeoff. A larger
value checkpoints fewer of the 24 recurrent block calls and therefore uses
more memory. If the default value `4` is too close to OOM, use `3`; if the
measured allocation is below the cluster requirement and headroom remains,
try `6`. The conservative fallback is
`MICRO_BATCH=256 ACTIVATION_CHECKPOINT_EVERY=0`. Preserve
`GLOBAL_BATCH=512` for comparison with the existing baseline.

## Direct launcher

Inside an allocated one-GPU Slurm shell, or on a standalone H200:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/path/to/imagenet_sd14_latents \
RUN_ROOT=/path/to/run \
  bash scripts/train_h200_single.sh
```

All paths and operational settings can be overridden with environment
variables documented at the top of `scripts/train_h200_single.sh`.
