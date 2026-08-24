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

Continuation checkpoints from 210K through 265K are available every 5K steps.
Later checkpoints are available every 10K from 270K through 320K, together
with the final 324K and 325K checkpoints. Select one by step, then pass it as
the initial checkpoint when starting a new run directory:

```bash
CHECKPOINT_STEP=325000 bash scripts/download_checkpoint.sh
python scripts/validate_checkpoint.py checkpoints/checkpoint-0325000.pt

INITIAL_CHECKPOINT="$PWD/checkpoints/checkpoint-0325000.pt" \
DATA_ROOT="$SCRATCH/looped-diffusion/imagenet_sd14_latents" \
RUN_ROOT="$SCRATCH/looped-diffusion/runs/fixed_2n12l_d1024_seed0" \
  sbatch --partition=<h200-partition> scripts/train_h200_single.sbatch
```

Set `CHECKPOINT_STEP` to any listed step to select an earlier continuation
checkpoint. If `RUN_ROOT/latest.pt` already exists, the launcher
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

## 4. Submit on H200s

Create the Slurm output directory before submission and pass the cluster's
partition and account on the command line. Choose the GPU count at submission
time; the same launcher supports 1, 2, 4, or 8 GPUs:

```bash
DATA_ROOT="$SCRATCH/looped-diffusion/imagenet_sd14_latents" \
RUN_ROOT="$SCRATCH/looped-diffusion/runs/fixed_2n12l_d1024_seed0" \
  bash scripts/submit_h200.sh 1 --partition=<h200-partition>

# The same baseline continuation on four H200s:
DATA_ROOT="$SCRATCH/looped-diffusion/imagenet_sd14_latents" \
RUN_ROOT="$SCRATCH/looped-diffusion/runs/fixed_2n12l_d1024_seed0" \
  bash scripts/submit_h200.sh 4 --partition=<h200-partition>
```

The helper requests `--gpus-per-node=h200:N` by default. For a cluster that
uses GRES syntax, set `GPU_REQUEST_STYLE=gres`. Set `GPU_TYPE=''` and pass
`--constraint=h200` when GPU type is selected by a constraint instead:

```bash
GPU_REQUEST_STYLE=gres bash scripts/submit_h200.sh 4 --partition=<partition>
GPU_TYPE='' bash scripts/submit_h200.sh 4 --partition=<partition> --constraint=h200
```

For a multi-node job, pass the node count followed by GPUs per node. All nodes
must see `DATA_ROOT`, `RUN_ROOT`, and the checkpoint at the same paths, so use
fast shared or parallel storage rather than node-local scratch:

```bash
DATA_ROOT="$SCRATCH/looped-diffusion/imagenet_sd14_latents" \
RUN_ROOT="$SCRATCH/looped-diffusion/runs/fixed_2nodes_8gpus" \
  bash scripts/submit_h200.sh 2 8 --partition=<h200-partition>
```

The batch script starts one supervisor per node and derives the rendezvous
address from the first allocated hostname. `MASTER_PORT` defaults to `29500`
and may be overridden if the cluster reserves or filters that port. Nodes must
be able to reach that TCP port. If NCCL chooses the wrong network interface,
set the cluster-appropriate `NCCL_SOCKET_IFNAME` when submitting the job.

Use `DRY_RUN=1` to print the complete `sbatch` command without submitting it.
The older `train_h200_single.sbatch` entry point remains available for
backward compatibility.

Defaults:

- resume from `checkpoints/checkpoint-0200000.pt`;
- train to 400K steps;
- global batch 512;
- one optimizer pass per step, with micro-batch set to `512 / GPU count`;
- a target of eight data-loader workers in total, with at least one worker per
  rank; logical shard splitting keeps all 40 shards uniformly sampled when the
  rank/worker count does not divide 40;
- selective activation checkpointing on a 512 micro-batch and no checkpointing
  on smaller per-GPU micro-batches;
- save every 5K steps for Slurm preemption recovery.

In the default `BATCH_POLICY=fixed` mode, GPU count only changes how the fixed
global batch is divided:

| H200s | Per-GPU micro-batch | Workers per rank | Global batch |
| ---: | ---: | ---: | ---: |
| 1 | 512 | 8 | 512 |
| 2 | 256 | 4 | 512 |
| 4 | 128 | 2 | 512 |
| 8 | 64 | 1 | 512 |

The same batch rule uses the global rank count across nodes. For example, two
8-GPU nodes use 16 ranks and a micro-batch of 32 at global batch 512.

This keeps step count, sample count, learning rate, and EMA semantics aligned
with the existing baseline. The GPU count must divide 512; if three GPUs are
idle, request two rather than silently changing the experiment. Custom
`WORKERS` values are accepted only when `WORKERS * GPU count` divides the 40
latent shards evenly, which prevents world-size-dependent shard reweighting.

The run directory's `latest.pt` is preferred automatically. After one job has
exited, re-submit with a different GPU count and the new job resumes the newest
saved step. Do not overlap jobs that use the same `RUN_ROOT`; the launcher also
uses an advisory lock to catch accidental overlap. Changing DDP world size is
not bitwise identical because rank-local RNG and batch ordering change, but the
optimizer and EMA state remain loadable. A running Slurm allocation cannot
grow or shrink in place.

If card availability changes often, reduce `CHECKPOINT_EVERY`, wait for a new
checkpoint, cancel the old job, then submit the same `RUN_ROOT` with the new
count. For example, `CHECKPOINT_EVERY=1000` limits a forced cancellation to at
most 1K lost steps.

For a short operational test:

```bash
DATA_ROOT="$SCRATCH/looped-diffusion/imagenet_sd14_latents" \
RUN_ROOT="$SCRATCH/looped-diffusion/runs/fixed_smoke" \
TARGET_STEPS=200100 CHECKPOINT_EVERY=100 \
  bash scripts/submit_h200.sh 1 --partition=<h200-partition>
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

### High-occupancy throughput mode

Fixed global batch inevitably lowers per-GPU memory use as more data-parallel
GPUs are added. If the cluster requires high HBM occupancy on every H200, the
launcher can instead keep micro-batch 512 on every GPU:

```bash
DATA_ROOT="$SCRATCH/looped-diffusion/imagenet_sd14_latents" \
RUN_ROOT="$SCRATCH/looped-diffusion/runs/fixed_throughput_gb2048" \
BATCH_POLICY=throughput ALLOW_GLOBAL_BATCH_CHANGE=1 \
  bash scripts/submit_h200.sh 4 --partition=<h200-partition>
```

On four GPUs this means global batch 2048. It is intentionally guarded and
must use a separate run directory: optimizer-update count, samples per step,
and EMA timescale no longer match the baseline, even though the checkpoint is
technically loadable. Do not report its 400K result as the fixed-batch 400K
baseline. A run contract prevents changing GPU count inside one throughput-mode
`RUN_ROOT`, because that would silently change global batch; use another run
directory for each throughput configuration.

## Direct launcher

Inside an allocated one-node Slurm shell, or on a standalone H200 node:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATA_ROOT=/path/to/imagenet_sd14_latents \
RUN_ROOT=/path/to/run \
GPUS_PER_NODE=4 \
  bash scripts/train_h200.sh
```

`GPUS_PER_NODE=auto` detects all GPUs visible to PyTorch. Limit
`CUDA_VISIBLE_DEVICES` first if the allocation exposes devices that should not
belong to this run.

For a direct multi-node launch, run the command once on every node with the
same `NNODES`, `MASTER_ADDR`, and `MASTER_PORT`, and a unique zero-based
`NODE_RANK`. For example, on node 0 and node 1 respectively:

```bash
# node 0
NNODES=2 NODE_RANK=0 GPUS_PER_NODE=8 \
MASTER_ADDR=node0.example MASTER_PORT=29500 \
DATA_ROOT=/shared/latents RUN_ROOT=/shared/run bash scripts/train_h200.sh

# node 1
NNODES=2 NODE_RANK=1 GPUS_PER_NODE=8 \
MASTER_ADDR=node0.example MASTER_PORT=29500 \
DATA_ROOT=/shared/latents RUN_ROOT=/shared/run bash scripts/train_h200.sh
```

If one node exits during training, `srun --kill-on-bad-exit` terminates the
remaining supervisors. Re-submit the job to resume from `latest.pt`; only
global rank zero writes logs and checkpoints.
