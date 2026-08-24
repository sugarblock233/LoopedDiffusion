from pathlib import Path

import pytest

from .data import partition_shards


@pytest.mark.parametrize("partitions", [1, 8, 16, 32, 48])
def test_partition_shards_balances_slots_and_assigns_each_sample_once(
    partitions: int,
) -> None:
    shards = [Path(f"train-{index:05d}.pt") for index in range(40)]
    assignments = [partition_shards(shards, index, partitions) for index in range(partitions)]

    assert len({len(value) for value in assignments}) == 1
    claimed = []
    for assignment in assignments:
        for shard, sample_partition, sample_partitions in assignment:
            claimed.extend(
                (shard, sample_index)
                for sample_index in range(sample_partition, 101, sample_partitions)
            )

    expected = [(shard, sample_index) for shard in shards for sample_index in range(101)]
    assert sorted(claimed) == sorted(expected)


def test_partition_shards_validates_partition_identity() -> None:
    with pytest.raises(ValueError, match="partition_id"):
        partition_shards([Path("train-00000.pt")], 2, 2)
