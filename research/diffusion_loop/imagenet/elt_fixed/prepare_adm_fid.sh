#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ASSETS="$ROOT/research/diffusion_loop/data/adm_fid"
REFERENCE="$ASSETS/VIRTUAL_imagenet256_labeled.npz"
EVALUATOR="$ASSETS/evaluator.py"
INCEPTION="$ASSETS/classify_image_graph_def.pb"

mkdir -p "$ASSETS"

if [[ ! -s "$REFERENCE" ]]; then
  curl -L --fail --retry 5 --retry-delay 5 --continue-at - \
    --output "$REFERENCE.part" \
    https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
  mv "$REFERENCE.part" "$REFERENCE"
fi

if [[ ! -s "$EVALUATOR" ]]; then
  curl -L --fail --retry 5 --retry-delay 5 \
    --output "$EVALUATOR.part" \
    https://raw.githubusercontent.com/openai/guided-diffusion/main/evaluations/evaluator.py
  mv "$EVALUATOR.part" "$EVALUATOR"
fi

if [[ ! -s "$INCEPTION" ]]; then
  curl -L --fail --retry 5 --retry-delay 5 --continue-at - \
    --output "$INCEPTION.part" \
    https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb
  mv "$INCEPTION.part" "$INCEPTION"
fi

sha256sum "$REFERENCE" "$EVALUATOR" "$INCEPTION" > "$ASSETS/SHA256SUMS"
