from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts" / "submit_h200.sh"


def dry_run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DRY_RUN"] = "1"
    return subprocess.run(
        ["bash", str(SUBMIT), *arguments],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_single_node_form_remains_supported() -> None:
    result = dry_run("4", "--partition=h200")

    assert result.returncode == 0
    assert "--nodes=1" in result.stdout
    assert "--gpus-per-node=h200:4" in result.stdout


def test_multi_node_form_exports_topology() -> None:
    result = dry_run("2", "8", "--partition=h200")

    assert result.returncode == 0
    assert "NNODES=2 GPUS_PER_NODE=8" in result.stdout
    assert "--nodes=2" in result.stdout
    assert "--gpus-per-node=h200:8" in result.stdout


def test_multi_node_form_rejects_zero_gpus() -> None:
    result = dry_run("2", "0", "--partition=h200")

    assert result.returncode == 2
    assert "must be positive integers" in result.stderr
