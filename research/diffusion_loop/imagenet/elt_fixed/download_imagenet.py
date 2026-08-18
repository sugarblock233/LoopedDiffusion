from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", default="benjamin-paine/imagenet-1k-256x256")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(args.repo)
    names = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.startswith("data/train-") and sibling.rfilename.endswith(".parquet")
    )
    if len(names) != 40:
        raise RuntimeError(f"expected 40 train shards, found {len(names)}")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=names,
        local_dir=args.output,
        max_workers=8,
    )
    missing = [name for name in names if not (args.output / name).is_file()]
    if missing:
        raise RuntimeError(f"download incomplete; missing {len(missing)} shards")
    print(f"downloaded={len(names)}/{len(names)}", flush=True)


if __name__ == "__main__":
    main()
