"""Train TinyPhase-GRU from the deterministic 8-second HDF5 caches.

The model predicts one P-arrival logit for every input sample. This program
does not alter the source caches and supports bounded smoke runs through
``--train-limit`` and ``--validation-limit``.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tinyphase_gru import TinyPhaseGRU


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train the TinyPhase-GRU P picker.")
    parser.add_argument("--config", type=Path, default=root / "configs" / "tinyphase_gru.yaml")
    parser.add_argument("--train-cache", type=Path, default=root / "data" / "cache" / "train_windows.hdf5")
    parser.add_argument("--validation-cache", type=Path, default=root / "data" / "cache" / "validation_windows.hdf5")
    parser.add_argument("--output-dir", type=Path, default=root / "runs" / "tinyphase_gru")
    parser.add_argument("--train-limit", type=int, default=None, help="Use only the first N training records.")
    parser.add_argument("--validation-limit", type=int, default=None, help="Use only the first N validation records.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config max_epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers; 0 is safest for Windows/HDF5.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a last.pt training checkpoint.")
    parser.add_argument("--overwrite", action="store_true", help="Allow a new run in a non-empty output directory.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Config not found: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section in ("model", "data", "training"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError("Config is missing mapping: {}".format(section))
    return config


class PhaseWindowDataset(Dataset):
    """Lazy, worker-safe reader for waveform and P-index cache datasets."""

    def __init__(self, path: Path, sigma_samples: float, limit: Optional[int] = None) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError("Window cache not found: {}".format(self.path))
        if sigma_samples <= 0:
            raise ValueError("sigma_samples must be positive")
        self.sigma_samples = float(sigma_samples)
        self._handle = None
        self._waveforms = None
        self._p_indices = None
        with h5py.File(self.path, "r") as handle:
            if "waveforms" not in handle or "p_indices" not in handle:
                raise ValueError("Cache lacks waveforms or p_indices: {}".format(self.path))
            total = int(handle["waveforms"].shape[0])
            shape = tuple(handle["waveforms"].shape[1:])
            if len(shape) != 2 or shape[1] != 3 or tuple(handle["p_indices"].shape) != (total,):
                raise ValueError("Unexpected cache shapes: waveforms={}, p_indices={}".format(handle["waveforms"].shape, handle["p_indices"].shape))
            self.window_samples = int(shape[0])
            self.cache_attributes = {key: _normal_scalar(value) for key, value in handle.attrs.items()}
        if limit is not None and limit <= 0:
            raise ValueError("Dataset limit must be positive")
        self.length = total if limit is None else min(total, int(limit))
        self.sample_axis = np.arange(self.window_samples, dtype=np.float32)

    def _open(self) -> None:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
            self._waveforms = self._handle["waveforms"]
            self._p_indices = self._handle["p_indices"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        self._open()
        waveform = np.asarray(self._waveforms[index], dtype=np.float32)
        p_index = int(self._p_indices[index])
        label = np.exp(-0.5 * ((self.sample_axis - p_index) / self.sigma_samples) ** 2).astype(np.float32)
        # Every cached event window contains exactly one P arrival. Treat the
        # Gaussian as a probability distribution over all candidate positions
        # rather than independent binary decisions.
        label /= np.sum(label, dtype=np.float32)
        return torch.from_numpy(waveform), torch.from_numpy(label), torch.tensor(p_index, dtype=torch.long)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_waveforms"] = None
        state["_p_indices"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = self._waveforms = self._p_indices = None

    def __del__(self) -> None:
        self.close()


def _normal_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class TemporalSoftTargetCrossEntropy(nn.Module):
    """Cross entropy for a normalized soft target along the time axis."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                "Logit and target shapes differ: {} versus {}".format(
                    tuple(logits.shape), tuple(targets.shape)
                )
            )
        return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def checkpoint_payload(
    model: TinyPhaseGRU,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_validation_mae: float,
    epochs_without_improvement: int,
    config: Dict[str, Any],
    preprocessing: Dict[str, Any],
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "model_class": "TinyPhaseGRU",
        "model_config": dict(model.config),
        "model_state_dict": model.state_dict(),
        "preprocessing": preprocessing,
        "training_state": {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_mae_samples": best_validation_mae,
            "epochs_without_improvement": epochs_without_improvement,
        },
        "training_config": config,
        "metrics": metrics,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Any,
    amp_enabled: bool,
    gradient_clip_norm: float,
    description: str,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    example_count = 0
    errors = []
    probabilities_at_peak = []

    progress = tqdm(loader, desc=description, unit="batch")
    for waveforms, labels, p_indices in progress:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        p_indices = p_indices.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(waveforms)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        batch_size = int(waveforms.shape[0])
        loss_sum += float(loss.detach()) * batch_size
        example_count += batch_size
        predicted = torch.argmax(logits.detach(), dim=1)
        errors.append((predicted - p_indices).abs().cpu().numpy())
        peak_probability = torch.softmax(logits.detach(), dim=1).amax(dim=1)
        probabilities_at_peak.append(peak_probability.cpu().numpy())
        progress.set_postfix(loss="{:.4f}".format(loss_sum / example_count))

    absolute_errors = np.concatenate(errors).astype(np.float64)
    peak_probabilities = np.concatenate(probabilities_at_peak).astype(np.float64)
    return {
        "loss": loss_sum / example_count,
        "mae_samples": float(np.mean(absolute_errors)),
        "mae_seconds": float(np.mean(absolute_errors) / 100.0),
        "median_ae_samples": float(np.median(absolute_errors)),
        "rmse_samples": float(np.sqrt(np.mean(absolute_errors ** 2))),
        "within_0_1s_percent": float(np.mean(absolute_errors <= 10) * 100.0),
        "within_0_2s_percent": float(np.mean(absolute_errors <= 20) * 100.0),
        "within_0_5s_percent": float(np.mean(absolute_errors <= 50) * 100.0),
        "mean_peak_probability": float(np.mean(peak_probabilities)),
    }


def append_history(path: Path, row: Dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_config = config["training"]
    seed = int(config["seed"])
    seed_everything(seed)

    output_dir = args.output_dir.resolve()
    existing_files = list(output_dir.iterdir()) if output_dir.exists() else []
    if args.resume is None and existing_files and not args.overwrite:
        raise FileExistsError("Output directory is not empty; choose another directory or pass --overwrite: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    sigma = float(config["data"]["gaussian_sigma_samples"])
    train_dataset = PhaseWindowDataset(args.train_cache, sigma, args.train_limit)
    validation_dataset = PhaseWindowDataset(args.validation_cache, sigma, args.validation_limit)
    if train_dataset.cache_attributes.get("channel_order") != validation_dataset.cache_attributes.get("channel_order"):
        raise ValueError("Training and validation channel orders differ")

    batch_size = int(args.batch_size or training_config["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyPhaseGRU(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    loss_name = str(training_config.get("loss", ""))
    if loss_name != "temporal_softmax_cross_entropy":
        raise ValueError("Unsupported training loss: {}".format(loss_name))
    criterion = TemporalSoftTargetCrossEntropy()
    amp_enabled = bool(training_config["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    max_epochs = int(args.epochs or training_config["max_epochs"])
    patience = int(training_config["early_stopping_patience"])
    gradient_clip_norm = float(training_config["gradient_clip_norm"])
    start_epoch = 1
    best_mae = math.inf
    stale_epochs = 0

    preprocessing = {
        "sampling_rate_hz": train_dataset.cache_attributes.get("sampling_rate_hz", 100.0),
        "window_samples": train_dataset.cache_attributes.get("window_samples", 800),
        "channel_order": train_dataset.cache_attributes.get("channel_order", '["E", "N", "Z"]'),
        "detrend": train_dataset.cache_attributes.get("detrend"),
        "filter_type": train_dataset.cache_attributes.get("filter_type"),
        "filter_order": train_dataset.cache_attributes.get("filter_order"),
        "filter_low_hz": train_dataset.cache_attributes.get("filter_low_hz"),
        "filter_high_hz": train_dataset.cache_attributes.get("filter_high_hz"),
        "normalization": train_dataset.cache_attributes.get("normalization"),
        "normalization_epsilon": train_dataset.cache_attributes.get("normalization_epsilon"),
        "gaussian_sigma_samples": sigma,
        "input_domain": "STEAD waveform values; see cache provenance",
    }

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        state = checkpoint["training_state"]
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state.get("scaler_state_dict", {}))
        start_epoch = int(state["epoch"]) + 1
        best_mae = float(state["best_validation_mae_samples"])
        stale_epochs = int(state["epochs_without_improvement"])

    run_manifest = {
        "config": config,
        "train_cache": str(args.train_cache.resolve()),
        "validation_cache": str(args.validation_cache.resolve()),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "preprocessing": preprocessing,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(run_manifest, indent=2, ensure_ascii=False))
    history_path = output_dir / "history.csv"
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp_enabled, gradient_clip_norm, "train {}/{}".format(epoch, max_epochs))
        validation_metrics = run_epoch(model, validation_loader, criterion, device, None, scaler, amp_enabled, gradient_clip_norm, "valid {}/{}".format(epoch, max_epochs))
        improved = validation_metrics["mae_samples"] < best_mae
        if improved:
            best_mae = validation_metrics["mae_samples"]
            stale_epochs = 0
        else:
            stale_epochs += 1

        row = {"epoch": epoch, "seconds": time.perf_counter() - epoch_start}
        row.update({"train_" + key: value for key, value in train_metrics.items()})
        row.update({"validation_" + key: value for key, value in validation_metrics.items()})
        append_history(history_path, row)
        payload = checkpoint_payload(model, optimizer, scaler, epoch, best_mae, stale_epochs, config, preprocessing, validation_metrics)
        atomic_torch_save(payload, output_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, output_dir / "best.pt")
        print("epoch={} train_loss={:.5f} validation_loss={:.5f} validation_MAE={:.2f} samples ({:.3f} s) best={:.2f}".format(epoch, train_metrics["loss"], validation_metrics["loss"], validation_metrics["mae_samples"], validation_metrics["mae_seconds"], best_mae))
        if stale_epochs >= patience:
            print("Early stopping after {} epochs without validation MAE improvement.".format(stale_epochs))
            break

    train_dataset.close()
    validation_dataset.close()
    print("Training finished. Best checkpoint: {}".format(output_dir / "best.pt"))


if __name__ == "__main__":
    main()
