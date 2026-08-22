#!/usr/bin/env bash
set -euo pipefail

# Multi-H200 launcher. NUM_GPUS may be "auto" or the exact number of GPUs
# allocated to this one-node job. See README.md for the batch-policy tradeoff.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/imagenet_sd14_latents}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/elt_fixed_2n12l_d1024_seed0}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$ROOT/checkpoints/checkpoint-0200000.pt}"
TARGET_STEPS="${TARGET_STEPS:-400000}"
NUM_GPUS="${NUM_GPUS:-auto}"
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
if [[ "$NUM_GPUS" == "auto" ]]; then
  NUM_GPUS="$DETECTED_GPUS"
elif ! is_positive_integer "$NUM_GPUS"; then
  echo "NUM_GPUS must be 'auto' or a positive integer" >&2
  exit 1
elif (( NUM_GPUS != DETECTED_GPUS )); then
  echo "NUM_GPUS=$NUM_GPUS but PyTorch sees $DETECTED_GPUS GPUs" >&2
  exit 1
fi

if [[ "$WORKERS" == "auto" ]]; then
  if ! is_positive_integer "$TOTAL_DATA_WORKERS"; then
    echo "TOTAL_DATA_WORKERS must be a positive integer" >&2
    exit 1
  fi
  if (( TOTAL_DATA_WORKERS % NUM_GPUS != 0 )); then
    echo "TOTAL_DATA_WORKERS=$TOTAL_DATA_WORKERS is not divisible by NUM_GPUS=$NUM_GPUS" >&2
    exit 1
  fi
  WORKERS=$((TOTAL_DATA_WORKERS / NUM_GPUS))
fi
if ! is_positive_integer "$WORKERS"; then
  echo "WORKERS must be 'auto' or a positive integer" >&2
  exit 1
fi
DATA_PARTITIONS=$((WORKERS * NUM_GPUS))
if (( 40 % DATA_PARTITIONS != 0 )); then
  echo "40 latent shards must be divisible by WORKERS * NUM_GPUS=$DATA_PARTITIONS" >&2
  echo "use the default auto workers or choose a divisor of 40" >&2
  exit 1
fi

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
      if (( GLOBAL_BATCH % NUM_GPUS != 0 )); then
        echo "GLOBAL_BATCH=$GLOBAL_BATCH is not divisible by NUM_GPUS=$NUM_GPUS" >&2
        exit 1
      fi
      MICRO_BATCH=$((GLOBAL_BATCH / NUM_GPUS))
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
      GLOBAL_BATCH=$((MICRO_BATCH * NUM_GPUS))
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
if (( GLOBAL_BATCH % (MICRO_BATCH * NUM_GPUS) != 0 )); then
  echo "GLOBAL_BATCH must be divisible by MICRO_BATCH * NUM_GPUS" >&2
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
if [[ -s "$RUN_ROOT/latest.pt" ]]; then
  RESUME="$RUN_ROOT/latest.pt"
else
  RESUME="$INITIAL_CHECKPOINT"
fi
if [[ ! -s "$RESUME" ]]; then
  echo "missing resume checkpoint: $RESUME" >&2
  exit 1
fi

runtime_args=(--visible-gpus "$NUM_GPUS")
if [[ "$REQUIRE_H200" == "1" ]]; then
  runtime_args+=(--require-h200)
fi
"$PYTHON" "$ROOT/scripts/check_runtime.py" "${runtime_args[@]}"
"$PYTHON" "$ROOT/scripts/validate_checkpoint.py" "$RESUME"
"$PYTHON" "$ROOT/scripts/check_run_contract.py" \
  --run-root "$RUN_ROOT" \
  --batch-policy "$BATCH_POLICY" \
  --global-batch "$GLOBAL_BATCH"

if [[ "$BATCH_POLICY" == "throughput" && "$GLOBAL_BATCH" != "512" ]]; then
  echo "WARNING: global batch is $GLOBAL_BATCH; this is not baseline-equivalent" >&2
fi
echo "launch: gpus=$NUM_GPUS policy=$BATCH_POLICY global_batch=$GLOBAL_BATCH \
micro_batch=$MICRO_BATCH activation_checkpoint_every=$ACTIVATION_CHECKPOINT_EVERY \
workers_per_rank=$WORKERS resume=$RESUME"

export PYTHONSAFEPATH=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$NUM_GPUS" \
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
