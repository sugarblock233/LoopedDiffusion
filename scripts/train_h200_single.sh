#!/usr/bin/env bash
set -euo pipefail

# Override these variables as needed: PYTHON, DATA_ROOT, RUN_ROOT,
# INITIAL_CHECKPOINT, TARGET_STEPS, GLOBAL_BATCH, MICRO_BATCH, WORKERS,
# CHECKPOINT_EVERY, and REQUIRE_H200.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/imagenet_sd14_latents}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/elt_fixed_2n12l_d1024_seed0}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$ROOT/checkpoints/checkpoint-0200000.pt}"
TARGET_STEPS="${TARGET_STEPS:-400000}"
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
MICRO_BATCH="${MICRO_BATCH:-128}"
WORKERS="${WORKERS:-8}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
REQUIRE_H200="${REQUIRE_H200:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing Python environment: $PYTHON" >&2
  exit 1
fi
if [[ "$(find "$DATA_ROOT" -maxdepth 1 -type f -name 'train-*.pt' 2>/dev/null | wc -l)" -ne 40 ]]; then
  echo "expected 40 latent shards under $DATA_ROOT" >&2
  exit 1
fi
if (( GLOBAL_BATCH < 1 || MICRO_BATCH < 1 || GLOBAL_BATCH % MICRO_BATCH != 0 )); then
  echo "GLOBAL_BATCH must be divisible by MICRO_BATCH for one GPU" >&2
  exit 1
fi
if (( TARGET_STEPS <= 200000 || CHECKPOINT_EVERY < 1 )); then
  echo "TARGET_STEPS must exceed 200000 and CHECKPOINT_EVERY must be positive" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
if [[ -s "$RUN_ROOT/latest.pt" ]]; then
  RESUME="$RUN_ROOT/latest.pt"
else
  RESUME="$INITIAL_CHECKPOINT"
fi
if [[ ! -s "$RESUME" ]]; then
  echo "missing resume checkpoint: $RESUME" >&2
  exit 1
fi

runtime_args=(--visible-gpus 1)
if [[ "$REQUIRE_H200" == "1" ]]; then
  runtime_args+=(--require-h200)
fi
"$PYTHON" "$ROOT/scripts/check_runtime.py" "${runtime_args[@]}"
"$PYTHON" "$ROOT/scripts/validate_checkpoint.py" "$RESUME"

export PYTHONSAFEPATH=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=1 \
  -m research.diffusion_loop.imagenet.elt_fixed.train \
  --data "$DATA_ROOT" \
  --output "$RUN_ROOT" \
  --steps "$TARGET_STEPS" \
  --global-batch "$GLOBAL_BATCH" \
  --micro-batch "$MICRO_BATCH" \
  --workers "$WORKERS" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --learning-rate 0.0001 \
  --warmup-steps 10000 \
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
  --image-resolution 256 \
  --noise-resolution 64 \
  --logsnr-min -15 \
  --logsnr-max 15 \
  --sigmoid-bias -1 \
  --seed 0 \
  --resume "$RESUME"
