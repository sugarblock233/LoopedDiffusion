#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: submit_h200.sh GPUS [sbatch options]

Examples:
  DATA_ROOT=/scratch/latents RUN_ROOT=/scratch/run \
    bash scripts/submit_h200.sh 4 --partition=h200

  DRY_RUN=1 bash scripts/submit_h200.sh 8 --partition=h200 --account=my-account

Environment:
  GPU_TYPE=h200             GPU type used in --gpus-per-node (empty for untyped)
  GPU_REQUEST_STYLE=gpus    Use "gpus" or "gres" Slurm request syntax
  CPUS_PER_GPU=8            CPU allocation used by the submission helper
  SLURM_MEMORY=96G          Node memory request
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  usage >&2
  exit 2
fi

NUM_GPUS="$1"
shift
GPU_TYPE="${GPU_TYPE-h200}"
GPU_REQUEST_STYLE="${GPU_REQUEST_STYLE:-gpus}"
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
SLURM_MEMORY="${SLURM_MEMORY:-96G}"

if [[ ! "$CPUS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPUS_PER_GPU must be a positive integer" >&2
  exit 2
fi
CPUS=$((CPUS_PER_GPU * NUM_GPUS))
if (( CPUS < 16 )); then
  CPUS=16
fi

case "$GPU_REQUEST_STYLE" in
  gpus)
    if [[ -n "$GPU_TYPE" ]]; then
      gpu_request=(--gpus-per-node="$GPU_TYPE:$NUM_GPUS")
    else
      gpu_request=(--gpus-per-node="$NUM_GPUS")
    fi
    ;;
  gres)
    if [[ -n "$GPU_TYPE" ]]; then
      gpu_request=(--gres="gpu:$GPU_TYPE:$NUM_GPUS")
    else
      gpu_request=(--gres="gpu:$NUM_GPUS")
    fi
    ;;
  *)
    echo "GPU_REQUEST_STYLE must be 'gpus' or 'gres'" >&2
    exit 2
    ;;
esac

mkdir -p "$ROOT/slurm-logs"
cd "$ROOT"
export NUM_GPUS
command=(
  sbatch
  --nodes=1
  --ntasks=1
  "${gpu_request[@]}"
  --cpus-per-task="$CPUS"
  --mem="$SLURM_MEMORY"
  "$@"
  "$ROOT/scripts/train_h200.sbatch"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'NUM_GPUS=%q ' "$NUM_GPUS"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
