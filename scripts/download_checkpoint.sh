#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${REPO:-sugarblock233/LoopedDiffusion}"
TAG="${TAG:-fixed-baseline-200k-v1}"
ASSET="checkpoint-0200000.pt"
DESTINATION="${DESTINATION:-$ROOT/checkpoints}"
MANIFEST="$ROOT/checkpoints/SHA256SUMS"
OUTPUT="$DESTINATION/$ASSET"
PART="$OUTPUT.part"
URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

command -v curl >/dev/null
command -v sha256sum >/dev/null
mkdir -p "$DESTINATION"

if [[ -s "$OUTPUT" ]] && (cd "$DESTINATION" && sha256sum -c "$MANIFEST"); then
  echo "checkpoint already downloaded and verified: $OUTPUT"
  exit 0
fi
if [[ -e "$OUTPUT" ]]; then
  echo "existing checkpoint failed SHA-256 validation: $OUTPUT" >&2
  exit 1
fi

curl --fail --location --retry 8 --retry-delay 5 --continue-at - \
  --output "$PART" "$URL"

expected="$(awk -v asset="$ASSET" '$2 == asset {print $1}' "$MANIFEST")"
actual="$(sha256sum "$PART" | awk '{print $1}')"
if [[ -z "$expected" || "$actual" != "$expected" ]]; then
  echo "checkpoint SHA-256 mismatch: expected=$expected actual=$actual" >&2
  exit 1
fi
mv "$PART" "$OUTPUT"
echo "downloaded and verified: $OUTPUT"
