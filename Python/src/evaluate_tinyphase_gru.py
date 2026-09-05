"""Evaluate a frozen TinyPhase-GRU checkpoint on one HDF5 cache."""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tinyphase_gru import TinyPhaseGRU
from train_tinyphase_gru import PhaseWindowDataset


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Evaluate a frozen TinyPhase-GRU checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=root / "runs" / "full_50000" / "best.pt")
    parser.add_argument("--test-cache", type=Path, default=root / "data" / "cache" / "test_windows.hdf5")
    parser.add_argument("--output-dir", type=Path, default=root / "runs" / "full_50000" / "internal_test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normal_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def require_empty_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            "Evaluation output is not empty. Refusing to overwrite an existing official result: {}".format(output_dir)
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def check_preprocessing(checkpoint: Dict[str, Any], cache_attributes: Dict[str, Any]) -> None:
    preprocessing = checkpoint.get("preprocessing", {})
    keys = (
        "sampling_rate_hz", "window_samples", "channel_order", "detrend",
        "filter_type", "filter_order", "filter_low_hz", "filter_high_hz",
        "normalization", "normalization_epsilon",
    )
    mismatches = []
    for key in keys:
        expected = preprocessing.get(key)
        actual = cache_attributes.get(key)
        if isinstance(expected, float) or isinstance(actual, float):
            same = expected is not None and actual is not None and np.isclose(float(expected), float(actual))
        else:
            same = expected == actual
        if not same:
            mismatches.append("{}: checkpoint={!r}, cache={!r}".format(key, expected, actual))
    if mismatches:
        raise ValueError("Preprocessing contract mismatch:\n" + "\n".join(mismatches))


def atomic_json(payload: Dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def decode_strings(values: np.ndarray) -> List[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    test_cache_path = args.test_cache.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    if not test_cache_path.is_file():
        raise FileNotFoundError("Test cache not found: {}".format(test_cache_path))
    require_empty_output(output_dir, args.overwrite)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TinyPhaseGRU(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    sigma = float(checkpoint["preprocessing"]["gaussian_sigma_samples"])
    dataset = PhaseWindowDataset(test_cache_path, sigma_samples=sigma)
    check_preprocessing(checkpoint, dataset.cache_attributes)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    predicted_parts = []
    truth_parts = []
    peak_parts = []
    entropy_parts = []
    with torch.inference_mode():
        for waveforms, _labels, p_indices in tqdm(loader, desc="internal test", unit="batch"):
            waveforms = waveforms.to(device, non_blocking=True)
            logits = model(waveforms)
            probabilities = torch.softmax(logits, dim=1)
            predicted_parts.append(torch.argmax(probabilities, dim=1).cpu().numpy())
            truth_parts.append(p_indices.numpy())
            peak_parts.append(torch.amax(probabilities, dim=1).cpu().numpy())
            entropy_parts.append((-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1)).cpu().numpy())

    predicted = np.concatenate(predicted_parts).astype(np.int64)
    truth = np.concatenate(truth_parts).astype(np.int64)
    peak_probability = np.concatenate(peak_parts).astype(np.float64)
    entropy = np.concatenate(entropy_parts).astype(np.float64)
    signed_error = predicted - truth
    absolute_error = np.abs(signed_error)

    with h5py.File(test_cache_path, "r") as handle:
        trace_names = decode_strings(handle["trace_names"][:])
        source_ids = decode_strings(handle["source_ids"][:])
        cache_attributes = {key: normal_scalar(value) for key, value in handle.attrs.items()}
    if len(trace_names) != len(predicted):
        raise RuntimeError("Metadata and prediction lengths differ")

    quantiles = {
        "p50": float(np.percentile(absolute_error, 50)),
        "p75": float(np.percentile(absolute_error, 75)),
        "p90": float(np.percentile(absolute_error, 90)),
        "p95": float(np.percentile(absolute_error, 95)),
        "p99": float(np.percentile(absolute_error, 99)),
    }
    metrics = {
        "evaluation_type": "frozen_internal_test",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("training_state", {}).get("epoch", -1)),
        "test_cache": str(test_cache_path),
        "test_examples": int(len(predicted)),
        "device": str(device),
        "mae_samples": float(np.mean(absolute_error)),
        "mae_seconds": float(np.mean(absolute_error) / 100.0),
        "median_ae_samples": float(np.median(absolute_error)),
        "median_ae_seconds": float(np.median(absolute_error) / 100.0),
        "rmse_samples": float(np.sqrt(np.mean(signed_error.astype(np.float64) ** 2))),
        "rmse_seconds": float(np.sqrt(np.mean(signed_error.astype(np.float64) ** 2)) / 100.0),
        "mean_signed_error_samples": float(np.mean(signed_error)),
        "within_0_1s_percent": float(np.mean(absolute_error <= 10) * 100.0),
        "within_0_2s_percent": float(np.mean(absolute_error <= 20) * 100.0),
        "within_0_5s_percent": float(np.mean(absolute_error <= 50) * 100.0),
        "over_0_5s_count": int(np.count_nonzero(absolute_error > 50)),
        "over_1_0s_count": int(np.count_nonzero(absolute_error > 100)),
        "over_2_0s_count": int(np.count_nonzero(absolute_error > 200)),
        "absolute_error_quantiles_samples": quantiles,
        "mean_peak_probability": float(np.mean(peak_probability)),
        "mean_probability_entropy": float(np.mean(entropy)),
        "cache_attributes": cache_attributes,
    }

    predictions_path = output_dir / "predictions.csv"
    temporary_csv = predictions_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "test_index", "trace_name", "source_id", "true_p_index", "predicted_p_index",
            "signed_error_samples", "absolute_error_samples", "absolute_error_seconds",
            "peak_probability", "probability_entropy",
        ))
        for index in range(len(predicted)):
            writer.writerow((
                index, trace_names[index], source_ids[index], int(truth[index]), int(predicted[index]),
                int(signed_error[index]), int(absolute_error[index]), float(absolute_error[index] / 100.0),
                float(peak_probability[index]), float(entropy[index]),
            ))
    os.replace(temporary_csv, predictions_path)
    atomic_json(metrics, output_dir / "metrics.json")
    dataset.close()

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Predictions: {}".format(predictions_path))


if __name__ == "__main__":
    main()
