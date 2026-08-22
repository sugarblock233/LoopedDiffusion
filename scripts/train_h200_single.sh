#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible one-GPU entry point.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NUM_GPUS=1
export BATCH_POLICY="${BATCH_POLICY:-fixed}"
export GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
export MICRO_BATCH="${MICRO_BATCH:-512}"
export ACTIVATION_CHECKPOINT_EVERY="${ACTIVATION_CHECKPOINT_EVERY:-4}"
exec bash "$ROOT/scripts/train_h200.sh"
