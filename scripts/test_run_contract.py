from __future__ import annotations

import json

import pytest

from scripts.check_run_contract import ensure_run_contract


def test_run_contract_allows_world_size_changes_with_fixed_batch(tmp_path) -> None:
    expected = ensure_run_contract(tmp_path, batch_policy="fixed", global_batch=512)
    actual = ensure_run_contract(tmp_path, batch_policy="fixed", global_batch=512)

    assert actual == expected
    assert json.loads((tmp_path / "run_contract.json").read_text()) == expected


def test_run_contract_rejects_batch_regime_changes(tmp_path) -> None:
    ensure_run_contract(tmp_path, batch_policy="throughput", global_batch=2048)

    with pytest.raises(ValueError, match="use a new RUN_ROOT"):
        ensure_run_contract(tmp_path, batch_policy="throughput", global_batch=512)
    with pytest.raises(ValueError, match="use a new RUN_ROOT"):
        ensure_run_contract(tmp_path, batch_policy="fixed", global_batch=512)


def test_existing_training_config_cannot_be_silently_adopted(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"global_batch": 2048}))

    with pytest.raises(ValueError, match="used global_batch=2048"):
        ensure_run_contract(tmp_path, batch_policy="fixed", global_batch=512)
