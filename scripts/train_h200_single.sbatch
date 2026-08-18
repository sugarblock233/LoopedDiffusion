#!/usr/bin/env bash
#SBATCH --job-name=looped-dit-fixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --requeue

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${DATA_ROOT:-}" ]]; then
  echo "DATA_ROOT must point to the 40 cached ImageNet latent shards" >&2
  exit 1
fi

export PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export RUN_ROOT="${RUN_ROOT:-$ROOT/runs/elt_fixed_2n12l_d1024_seed0}"
export TARGET_STEPS="${TARGET_STEPS:-400000}"
export GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
export MICRO_BATCH="${MICRO_BATCH:-512}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
export ACTIVATION_CHECKPOINT_EVERY="${ACTIVATION_CHECKPOINT_EVERY:-4}"
export WORKERS="${WORKERS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

srun --ntasks=1 bash "$ROOT/scripts/train_h200_single.sh"
