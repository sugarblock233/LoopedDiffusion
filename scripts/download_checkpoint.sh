#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${REPO:-sugarblock233/LoopedDiffusion}"
TAG="${TAG:-fixed-baseline-200k-v1}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-200000}"
printf -v ASSET 'checkpoint-%07d.pt' "$CHECKPOINT_STEP"
DESTINATION="${DESTINATION:-$ROOT/checkpoints}"
MANIFEST="$ROOT/checkpoints/SHA256SUMS"
OUTPUT="$DESTINATION/$ASSET"
PART="$OUTPUT.part"
URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

command -v curl >/dev/null
command -v sha256sum >/dev/null
mkdir -p "$DESTINATION"

expected="$(awk -v asset="$ASSET" '$2 == asset {print $1}' "$MANIFEST")"
if [[ -z "$expected" ]]; then
  echo "checkpoint is not listed in the manifest: $ASSET" >&2
  exit 1
fi
if [[ -s "$OUTPUT" ]]; then
  actual="$(sha256sum "$OUTPUT" | awk '{print $1}')"
  if [[ "$actual" == "$expected" ]]; then
    echo "checkpoint already downloaded and verified: $OUTPUT"
    exit 0
  fi
  echo "existing checkpoint failed SHA-256 validation: $OUTPUT" >&2
  exit 1
fi

curl --fail --location --retry 8 --retry-delay 5 --continue-at - \
  --output "$PART" "$URL"

actual="$(sha256sum "$PART" | awk '{print $1}')"
if [[ "$actual" != "$expected" ]]; then
  echo "checkpoint SHA-256 mismatch: expected=$expected actual=$actual" >&2
  exit 1
fi
mv "$PART" "$OUTPUT"
echo "downloaded and verified: $OUTPUT"
