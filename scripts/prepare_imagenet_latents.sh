#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
RAW_ROOT="${RAW_ROOT:-$ROOT/data/imagenet_256_parquet}"
LATENT_ROOT="${LATENT_ROOT:-$ROOT/data/imagenet_sd14_latents}"
PREP_BATCH="${PREP_BATCH:-128}"
VAE="${VAE:-CompVis/stable-diffusion-v1-4}"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing Python environment: $PYTHON" >&2
  exit 1
fi
if [[ "$(find "$RAW_ROOT/data" -maxdepth 1 -type f -name 'train-*.parquet' 2>/dev/null | wc -l)" -ne 40 ]]; then
  "$PYTHON" -m research.diffusion_loop.imagenet.elt_fixed.download_imagenet \
    --output "$RAW_ROOT"
fi

mkdir -p "$LATENT_ROOT"
"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=1 \
  -m research.diffusion_loop.imagenet.elt_fixed.prepare_latents \
  --input "$RAW_ROOT" \
  --output "$LATENT_ROOT" \
  --vae "$VAE" \
  --batch-size "$PREP_BATCH"

count="$(find "$LATENT_ROOT" -maxdepth 1 -type f -name 'train-*.pt' | wc -l)"
if [[ "$count" -ne 40 ]]; then
  echo "expected 40 latent shards, found $count" >&2
  exit 1
fi
echo "prepared 40 latent shards under $LATENT_ROOT"
