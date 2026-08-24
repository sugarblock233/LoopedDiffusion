from __future__ import annotations

import io
import math
import random
import tarfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


def center_crop(image: Image.Image, size: int = 256) -> torch.Tensor:
    width, height = image.size
    scale = size / min(width, height)
    resized = TF.resize(
        image,
        [round(height * scale), round(width * scale)],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    top = (resized.height - size) // 2
    left = (resized.width - size) // 2
    return TF.to_tensor(TF.crop(resized, top, left, size, size)).mul_(2.0).sub_(1.0)


def read_image_shard(path: Path) -> list[tuple[torch.Tensor, int]]:
    pending: dict[str, dict[str, bytes]] = {}
    with tarfile.open(path, "r") as archive:
        for member in archive:
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".cls"}:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            key = str(Path(member.name).with_suffix(""))
            pending.setdefault(key, {})[suffix] = handle.read()
    samples = []
    for key in sorted(pending):
        item = pending[key]
        image_bytes = item.get(".jpg") or item.get(".jpeg")
        if image_bytes is None or ".cls" not in item:
            continue
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        samples.append((center_crop(image), int(item[".cls"].decode().strip())))
    return samples


def iter_parquet_batches(
    path: Path, batch_size: int, decode_workers: int = 8
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Decode a Hugging Face ImageNet parquet shard without materializing it all."""
    if decode_workers < 1:
        raise ValueError("decode_workers must be positive")

    def decode(encoded: dict[str, bytes | str | None]) -> torch.Tensor:
        with Image.open(io.BytesIO(encoded["bytes"])) as source:
            image = source.convert("RGB")
            if image.size != (256, 256):
                raise ValueError(f"expected a 256x256 image in {path}, got {image.size}")
            return TF.pil_to_tensor(image)

    parquet = pq.ParquetFile(path)
    images: list[torch.Tensor] = []
    labels: list[int] = []
    with ThreadPoolExecutor(max_workers=decode_workers) as executor:
        for row_group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(row_group, columns=["image", "label"])
            decoded = executor.map(decode, table["image"].to_pylist())
            for image, label in zip(decoded, table["label"].to_pylist()):
                images.append(image)
                labels.append(int(label))
                if len(images) == batch_size:
                    batch = torch.stack(images).float().div_(127.5).sub_(1.0)
                    yield batch, torch.tensor(labels, dtype=torch.int64)
                    images.clear()
                    labels.clear()
    if images:
        batch = torch.stack(images).float().div_(127.5).sub_(1.0)
        yield batch, torch.tensor(labels, dtype=torch.int64)


def iter_parquet_encoded_batches(
    path: Path, batch_size: int
) -> Iterator[tuple[list[bytes], torch.Tensor]]:
    """Read encoded images in batches for NVJPEG decoding on the training GPU."""
    parquet = pq.ParquetFile(path)
    images: list[bytes] = []
    labels: list[int] = []
    for row_group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(row_group, columns=["image", "label"])
        for encoded, label in zip(table["image"].to_pylist(), table["label"].to_pylist()):
            image_bytes = encoded["bytes"]
            if not image_bytes.startswith(b"\xff\xd8"):
                raise ValueError(f"expected JPEG data in {path}")
            images.append(image_bytes)
            labels.append(int(label))
            if len(images) == batch_size:
                yield images, torch.tensor(labels, dtype=torch.int64)
                images = []
                labels = []
    if images:
        yield images, torch.tensor(labels, dtype=torch.int64)


def partition_shards(
    shards: list[Path], partition_id: int, partitions: int
) -> list[tuple[Path, int, int]]:
    """Assign every sample once while giving each partition equal logical slots.

    When the shard count is not divisible by the number of consumers, shards
    are split between multiple consumers. This avoids the sampling bias caused
    by making some ranks cycle through fewer complete shards than others.
    """
    if not shards:
        raise ValueError("shards must not be empty")
    if partitions < 1 or not 0 <= partition_id < partitions:
        raise ValueError("partition_id must be in [0, partitions)")
    replicas = partitions // math.gcd(len(shards), partitions)
    slots = [
        (shard, replica, replicas)
        for shard in shards
        for replica in range(replicas)
    ]
    return slots[partition_id::partitions]


class LatentShardDataset(IterableDataset):
    def __init__(self, root: Path | str, *, seed: int = 0, flip_probability: float = 0.5) -> None:
        super().__init__()
        self.root = Path(root)
        self.seed = int(seed)
        self.flip_probability = float(flip_probability)

    def _partition(self) -> list[tuple[Path, int, int]]:
        shards = sorted(self.root.glob("train-*.pt"))
        if not shards:
            raise FileNotFoundError(f"no latent shards under {self.root}")
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
        else:
            rank, world = 0, 1
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        partition_id = rank * workers + worker_id
        partitions = world * workers
        return partition_shards(shards, partition_id, partitions)

    def __iter__(self):
        shard_slots = self._partition()
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        partition_id = rank * (1 if worker is None else worker.num_workers) + worker_id
        epoch = 0
        while True:
            rng = random.Random(self.seed + 1_000_003 * epoch + partition_id)
            order = list(shard_slots)
            rng.shuffle(order)
            for shard, sample_partition, sample_partitions in order:
                payload = torch.load(shard, map_location="cpu", weights_only=True)
                moments = payload["moments"]
                labels = payload["labels"]
                indices = list(range(sample_partition, labels.shape[0], sample_partitions))
                rng.shuffle(indices)
                for index in indices:
                    value = moments[index]
                    if rng.random() < self.flip_probability:
                        value = value.flip(-1)
                    yield value, labels[index]
            epoch += 1


__all__ = [
    "LatentShardDataset",
    "center_crop",
    "iter_parquet_encoded_batches",
    "iter_parquet_batches",
    "partition_shards",
    "read_image_shard",
]
