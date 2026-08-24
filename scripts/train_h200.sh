#!/usr/bin/env bash
set -euo pipefail

# Multi-node H200 launcher. Slurm fills the topology variables through
# train_h200.sbatch; they may also be supplied directly for a manual launch.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/imagenet_sd14_latents}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/elt_fixed_2n12l_d1024_seed0}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$ROOT/checkpoints/checkpoint-0200000.pt}"
TARGET_STEPS="${TARGET_STEPS:-400000}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${NUM_GPUS:-auto}}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
BATCH_POLICY="${BATCH_POLICY:-fixed}"
GLOBAL_BATCH="${GLOBAL_BATCH:-auto}"
MICRO_BATCH="${MICRO_BATCH:-auto}"
PER_GPU_BATCH="${PER_GPU_BATCH:-512}"
ALLOW_GLOBAL_BATCH_CHANGE="${ALLOW_GLOBAL_BATCH_CHANGE:-0}"
WORKERS="${WORKERS:-auto}"
TOTAL_DATA_WORKERS="${TOTAL_DATA_WORKERS:-8}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
ACTIVATION_CHECKPOINT_EVERY="${ACTIVATION_CHECKPOINT_EVERY:-auto}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
WARMUP_STEPS="${WARMUP_STEPS:-10000}"
REQUIRE_H200="${REQUIRE_H200:-1}"

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if [[ ! -x "$PYTHON" ]]; then
  echo "missing Python environment: $PYTHON" >&2
  exit 1
fi

DETECTED_GPUS="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
if ! is_positive_integer "$DETECTED_GPUS"; then
  echo "no CUDA GPUs are visible to PyTorch" >&2
  exit 1
fi
if [[ "$GPUS_PER_NODE" == "auto" ]]; then
  GPUS_PER_NODE="$DETECTED_GPUS"
elif ! is_positive_integer "$GPUS_PER_NODE"; then
  echo "GPUS_PER_NODE must be 'auto' or a positive integer" >&2
  exit 1
elif (( GPUS_PER_NODE != DETECTED_GPUS )); then
  echo "GPUS_PER_NODE=$GPUS_PER_NODE but PyTorch sees $DETECTED_GPUS GPUs" >&2
  exit 1
fi
if ! is_positive_integer "$NNODES"; then
  echo "NNODES must be a positive integer" >&2
  exit 1
fi
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
  echo "NODE_RANK must be an integer in [0, NNODES)" >&2
  exit 1
fi
if [[ -z "$MASTER_ADDR" ]] || ! [[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] || (( MASTER_PORT > 65535 )); then
  echo "MASTER_ADDR must be non-empty and MASTER_PORT must be in [1, 65535]" >&2
  exit 1
fi
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

if [[ "$WORKERS" == "auto" ]]; then
  if ! is_positive_integer "$TOTAL_DATA_WORKERS"; then
    echo "TOTAL_DATA_WORKERS must be a positive integer" >&2
    exit 1
  fi
  WORKERS=$((TOTAL_DATA_WORKERS / WORLD_SIZE))
  if (( WORKERS < 1 )); then
    WORKERS=1
  fi
fi
if ! is_positive_integer "$WORKERS"; then
  echo "WORKERS must be 'auto' or a positive integer" >&2
  exit 1
fi
DATA_PARTITIONS=$((WORKERS * WORLD_SIZE))

if ! is_positive_integer "$PER_GPU_BATCH"; then
  echo "PER_GPU_BATCH must be a positive integer" >&2
  exit 1
fi
case "$BATCH_POLICY" in
  fixed)
    if [[ "$GLOBAL_BATCH" == "auto" ]]; then
      GLOBAL_BATCH=512
    fi
    if ! is_positive_integer "$GLOBAL_BATCH"; then
      echo "GLOBAL_BATCH must be a positive integer" >&2
      exit 1
    fi
    if [[ "$MICRO_BATCH" == "auto" ]]; then
      if (( GLOBAL_BATCH % WORLD_SIZE != 0 )); then
        echo "GLOBAL_BATCH=$GLOBAL_BATCH is not divisible by WORLD_SIZE=$WORLD_SIZE" >&2
        exit 1
      fi
      MICRO_BATCH=$((GLOBAL_BATCH / WORLD_SIZE))
    fi
    ;;
  throughput)
    if [[ "$MICRO_BATCH" == "auto" ]]; then
      MICRO_BATCH="$PER_GPU_BATCH"
    fi
    if ! is_positive_integer "$MICRO_BATCH"; then
      echo "MICRO_BATCH must be a positive integer" >&2
      exit 1
    fi
    if [[ "$GLOBAL_BATCH" == "auto" ]]; then
      GLOBAL_BATCH=$((MICRO_BATCH * WORLD_SIZE))
    fi
    if ! is_positive_integer "$GLOBAL_BATCH"; then
      echo "GLOBAL_BATCH must be a positive integer" >&2
      exit 1
    fi
    if (( GLOBAL_BATCH != 512 )) && [[ "$ALLOW_GLOBAL_BATCH_CHANGE" != "1" ]]; then
      echo "throughput policy changes global batch to $GLOBAL_BATCH" >&2
      echo "set ALLOW_GLOBAL_BATCH_CHANGE=1 and use a separate RUN_ROOT to confirm" >&2
      exit 1
    fi
    ;;
  *)
    echo "BATCH_POLICY must be 'fixed' or 'throughput'" >&2
    exit 1
    ;;
esac

if ! is_positive_integer "$GLOBAL_BATCH" || ! is_positive_integer "$MICRO_BATCH"; then
  echo "GLOBAL_BATCH and MICRO_BATCH must be positive integers" >&2
  exit 1
fi
if (( GLOBAL_BATCH % (MICRO_BATCH * WORLD_SIZE) != 0 )); then
  echo "GLOBAL_BATCH must be divisible by MICRO_BATCH * WORLD_SIZE" >&2
  exit 1
fi
if [[ "$ACTIVATION_CHECKPOINT_EVERY" == "auto" ]]; then
  if (( MICRO_BATCH >= 512 )); then
    ACTIVATION_CHECKPOINT_EVERY=4
  else
    ACTIVATION_CHECKPOINT_EVERY=0
  fi
fi
if ! [[ "$ACTIVATION_CHECKPOINT_EVERY" =~ ^[0-9]+$ ]]; then
  echo "ACTIVATION_CHECKPOINT_EVERY must be 'auto' or a non-negative integer" >&2
  exit 1
fi
if ! is_positive_integer "$TARGET_STEPS" || ! is_positive_integer "$CHECKPOINT_EVERY"; then
  echo "TARGET_STEPS and CHECKPOINT_EVERY must be positive integers" >&2
  exit 1
fi
shard_count="$(
  find "$DATA_ROOT" -maxdepth 1 -type f -name 'train-*.pt' 2>/dev/null | wc -l
)"
if [[ "$shard_count" -ne 40 ]]; then
  echo "expected 40 latent shards under $DATA_ROOT" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
if (( NODE_RANK == 0 )); then
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$RUN_ROOT/.train.lock"
    if ! flock -n 9; then
      echo "another training process already holds $RUN_ROOT/.train.lock" >&2
      exit 1
    fi
  elif [[ "${ALLOW_NO_RUN_LOCK:-0}" != "1" ]]; then
    echo "flock is required to protect RUN_ROOT from concurrent writers" >&2
    echo "set ALLOW_NO_RUN_LOCK=1 only if the cluster provides external locking" >&2
    exit 1
  fi
fi
if [[ -s "$RUN_ROOT/latest.pt" ]]; then
  RESUME="$RUN_ROOT/latest.pt"
else
  RESUME="$INITIAL_CHECKPOINT"
fi
if [[ ! -s "$RESUME" ]]; then
  echo "missing resume checkpoint: $RESUME" >&2
  exit 1
fi

runtime_args=(--visible-gpus "$GPUS_PER_NODE")
if [[ "$REQUIRE_H200" == "1" ]]; then
  runtime_args+=(--require-h200)
fi
"$PYTHON" "$ROOT/scripts/check_runtime.py" "${runtime_args[@]}"
"$PYTHON" "$ROOT/scripts/validate_checkpoint.py" "$RESUME"
if (( NODE_RANK == 0 )); then
  "$PYTHON" "$ROOT/scripts/check_run_contract.py" \
    --run-root "$RUN_ROOT" \
    --batch-policy "$BATCH_POLICY" \
    --global-batch "$GLOBAL_BATCH"
fi

if [[ "$BATCH_POLICY" == "throughput" && "$GLOBAL_BATCH" != "512" ]]; then
  echo "WARNING: global batch is $GLOBAL_BATCH; this is not baseline-equivalent" >&2
fi
echo "launch: node_rank=$NODE_RANK/$NNODES gpus_per_node=$GPUS_PER_NODE world_size=$WORLD_SIZE \
master=$MASTER_ADDR:$MASTER_PORT policy=$BATCH_POLICY global_batch=$GLOBAL_BATCH \
micro_batch=$MICRO_BATCH activation_checkpoint_every=$ACTIVATION_CHECKPOINT_EVERY \
workers_per_rank=$WORKERS data_partitions=$DATA_PARTITIONS resume=$RESUME"

export PYTHONSAFEPATH=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NNODES NODE_RANK GPUS_PER_NODE MASTER_ADDR MASTER_PORT WORLD_SIZE

"$PYTHON" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --node-rank="$NODE_RANK" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  -m research.diffusion_loop.imagenet.elt_fixed.train \
  --data "$DATA_ROOT" \
  --output "$RUN_ROOT" \
  --steps "$TARGET_STEPS" \
  --global-batch "$GLOBAL_BATCH" \
  --micro-batch "$MICRO_BATCH" \
  --workers "$WORKERS" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --learning-rate "$LEARNING_RATE" \
  --warmup-steps "$WARMUP_STEPS" \
  --weight-decay 0.01 \
  --label-drop 0.1 \
  --ema-decay 0.9999 \
  --vae-scale 0.18215 \
  --latent-size 32 \
  --patch-size 2 \
  --channels 4 \
  --hidden-size 1024 \
  --heads 16 \
  --mlp-ratio 4 \
  --unique-blocks 2 \
  --loops 12 \
  --activation-checkpoint-every "$ACTIVATION_CHECKPOINT_EVERY" \
  --image-resolution 256 \
  --noise-resolution 64 \
  --logsnr-min -15 \
  --logsnr-max 15 \
  --sigmoid-bias -1 \
  --seed 0 \
  --resume "$RESUME"
