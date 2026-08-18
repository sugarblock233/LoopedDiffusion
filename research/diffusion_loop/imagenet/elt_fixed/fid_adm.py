from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def image_batches(paths: list[Path], batch_size: int):
    for offset in range(0, len(paths), batch_size):
        arrays = []
        for path in paths[offset : offset + batch_size]:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if image.size != (256, 256):
                    raise ValueError(f"unexpected image size for {path}: {image.size}")
                arrays.append(np.asarray(image, dtype=np.uint8))
        yield np.stack(arrays)


def main() -> None:
    args = parse_args()
    if args.num_samples < 2 or args.batch_size < 1:
        raise ValueError("FID needs at least two samples and a positive batch size")
    paths = sorted(args.samples.glob("*.png"))
    if len(paths) != args.num_samples:
        raise ValueError(f"expected {args.num_samples} PNG files, found {len(paths)}")
    expected_names = [f"{index:06d}.png" for index in range(args.num_samples)]
    actual_names = [path.name for path in paths]
    if actual_names != expected_names:
        raise ValueError("sample filenames must be contiguous from 000000.png")

    sys.path.insert(0, str(args.evaluator_dir.resolve()))
    import evaluator  # type: ignore[import-not-found]  # Official ADM evaluator.

    evaluator.tf.disable_eager_execution()
    evaluator.INCEPTION_V3_PATH = str(
        (args.evaluator_dir / "classify_image_graph_def.pb").resolve()
    )
    config = evaluator.tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = evaluator.tf.Session(config=config)
    image_input = evaluator.tf.placeholder(evaluator.tf.float32, shape=[None, None, None, 3])
    pool_features, _ = evaluator._create_feature_graph(image_input)
    session.run(pool_features, {image_input: np.zeros((1, 64, 64, 3), dtype=np.float32)})

    feature_sum = np.zeros(2048, dtype=np.float64)
    feature_product = np.zeros((2048, 2048), dtype=np.float64)
    count = 0
    batches = image_batches(paths, args.batch_size)
    total_batches = (len(paths) + args.batch_size - 1) // args.batch_size
    for batch in tqdm(batches, total=total_batches, desc="ADM Inception features"):
        features = session.run(pool_features, {image_input: batch.astype(np.float32)})
        features = features.reshape(features.shape[0], -1).astype(np.float64)
        feature_sum += features.sum(axis=0)
        feature_product += features.T @ features
        count += features.shape[0]
    session.close()

    sample_mean = feature_sum / count
    sample_covariance = (
        feature_product - count * np.outer(sample_mean, sample_mean)
    ) / (count - 1)
    with np.load(args.reference) as reference:
        reference_statistics = evaluator.FIDStatistics(reference["mu"], reference["sigma"])
    sample_statistics = evaluator.FIDStatistics(sample_mean, sample_covariance)
    fid = float(sample_statistics.frechet_distance(reference_statistics))
    result = {
        "fid": fid,
        "metric": "ADM TensorFlow InceptionV3 FID",
        "num_samples": count,
        "reference": str(args.reference.resolve()),
        "samples": str(args.samples.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
