#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def ensure_run_contract(
    run_root: Path, *, batch_policy: str, global_batch: int
) -> dict:
    expected = {
        "batch_policy": batch_policy,
        "global_batch": int(global_batch),
        "schema_version": 1,
    }
    contract_path = run_root / "run_contract.json"
    if contract_path.exists():
        actual = read_json(contract_path)
        mismatches = {
            key: (actual.get(key), value)
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: existing={old!r}, requested={new!r}"
                for key, (old, new) in mismatches.items()
            )
            raise ValueError(
                f"run contract mismatch for {run_root}: {details}; use a new RUN_ROOT"
            )
        return actual

    config_path = run_root / "config.json"
    if config_path.exists():
        previous_config = read_json(config_path)
        previous_batch = previous_config.get("global_batch")
        if previous_batch is not None and int(previous_batch) != global_batch:
            raise ValueError(
                f"existing {config_path} used global_batch={previous_batch}, "
                f"requested={global_batch}; use a new RUN_ROOT"
            )

    temporary = contract_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    temporary.replace(contract_path)
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--batch-policy", choices=("fixed", "throughput"), required=True)
    parser.add_argument("--global-batch", type=int, required=True)
    args = parser.parse_args()

    try:
        contract = ensure_run_contract(
            args.run_root,
            batch_policy=args.batch_policy,
            global_batch=args.global_batch,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(contract, sort_keys=True))


if __name__ == "__main__":
    main()
