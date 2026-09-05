"""Build verified 10 s, P-index 100..900 caches from chunk2+chunk3."""

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "indexes_10s_p100_900"
OUTPUT_DIR = ROOT / "data" / "cache_10s_p100_900"
SOURCE_HDF5 = {
    "chunk2": ROOT / "data" / "raw" / "STEAD" / "chunk2.hdf5",
    "chunk3": Path(r"G:\Matlab_IC\Main_project_matlab\data\STEAD\chunk3 (1)\chunk3.hdf5"),
}
INDEX_FILES = {"train": "train_40000.csv", "validation": "validation_5000.csv", "test": "test_5000.csv"}
OUTPUT_FILES = {"train": "train_windows_10s.hdf5", "validation": "validation_windows_10s.hdf5", "test": "test_windows_10s.hdf5"}
SEED = 20260823
SAMPLING_RATE_HZ = 100.0
WINDOW_SAMPLES = 1000
P_INDEX_MIN, P_INDEX_MAX = 100, 900
FILTER_ORDER, FILTER_LOW_HZ, FILTER_HIGH_HZ = 4, 1.0, 7.0
EPSILON = 1.0e-6


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rng_for(split, trace_name):
    token = "{}:{}:{}".format(SEED, split, trace_name).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))


def crop_bounds(original_p, source_length, split, trace_name):
    original = int(np.rint(float(original_p)))
    feasible_min = max(P_INDEX_MIN, original - (source_length - WINDOW_SAMPLES))
    feasible_max = min(P_INDEX_MAX, original)
    if feasible_min > feasible_max:
        raise RuntimeError("No feasible crop for {}".format(trace_name))
    p_index = int(rng_for(split, trace_name).integers(feasible_min, feasible_max + 1))
    start = original - p_index
    end = start + WINDOW_SAMPLES
    if start < 0 or end > source_length:
        raise RuntimeError("Crop bounds failure for {}".format(trace_name))
    return start, end, p_index


def preprocess(raw, sos):
    waveform = np.asarray(raw, dtype=np.float64)
    if waveform.shape != (WINDOW_SAMPLES, 3) or not np.isfinite(waveform).all():
        raise RuntimeError("Invalid raw crop")
    waveform = signal.detrend(waveform, axis=0, type="constant")
    waveform = signal.detrend(waveform, axis=0, type="linear")
    waveform = signal.sosfiltfilt(sos, waveform, axis=0)
    means = waveform.mean(axis=0, keepdims=True)
    stds = waveform.std(axis=0, keepdims=True)
    near_constant = int(np.count_nonzero(stds < EPSILON))
    waveform = ((waveform - means) / np.maximum(stds, EPSILON)).astype(np.float32)
    if not np.isfinite(waveform).all():
        raise RuntimeError("Non-finite preprocessed crop")
    return waveform, near_constant


def source_group(handle):
    return handle["data"] if "data" in handle else handle


def create_datasets(handle, count):
    strings = h5py.string_dtype("utf-8")
    chunk_rows = min(128, count)
    return {
        "waveforms": handle.create_dataset("waveforms", (count, WINDOW_SAMPLES, 3), dtype="float32", chunks=(chunk_rows, WINDOW_SAMPLES, 3), compression="lzf"),
        "p_indices": handle.create_dataset("p_indices", (count,), dtype="int16"),
        "crop_starts": handle.create_dataset("crop_starts", (count,), dtype="int16"),
        "original_p_samples": handle.create_dataset("original_p_samples", (count,), dtype="float32"),
        "trace_names": handle.create_dataset("trace_names", (count,), dtype=strings),
        "source_ids": handle.create_dataset("source_ids", (count,), dtype=strings),
        "source_chunks": handle.create_dataset("source_chunks", (count,), dtype=strings),
    }


def build_split(split, index_path, output_path, handles):
    index = pd.read_csv(index_path, dtype={"trace_name": "string", "source_id": "string", "source_chunk": "string"}, low_memory=False)
    required = {"trace_name", "source_id", "source_chunk", "p_arrival_sample", "split"}
    if required.difference(index.columns) or not index["split"].astype(str).eq(split).all():
        raise RuntimeError("Invalid index {}".format(index_path))
    if index["trace_name"].duplicated().any():
        raise RuntimeError("Duplicate trace_name in {}".format(index_path))
    if output_path.exists():
        raise FileExistsError("Refusing to overwrite {}".format(output_path))
    tmp = output_path.with_suffix(".hdf5.tmp")
    if tmp.exists():
        tmp.unlink()
    sos = signal.butter(FILTER_ORDER, [FILTER_LOW_HZ, FILTER_HIGH_HZ], btype="bandpass", fs=SAMPLING_RATE_HZ, output="sos")
    histogram = np.zeros(P_INDEX_MAX - P_INDEX_MIN + 1, dtype=np.int64)
    near_constant = 0
    try:
        with h5py.File(tmp, "w") as out:
            ds = create_datasets(out, len(index))
            attrs = {
                "split": split, "records": len(index), "sampling_rate_hz": SAMPLING_RATE_HZ,
                "window_samples": WINDOW_SAMPLES, "channel_order": json.dumps(["E", "N", "Z"]),
                "p_index_min": P_INDEX_MIN, "p_index_max": P_INDEX_MAX, "crop_seed": SEED,
                "detrend": "demean_then_linear", "filter_type": "butterworth_sosfiltfilt",
                "filter_order": FILTER_ORDER, "filter_low_hz": FILTER_LOW_HZ, "filter_high_hz": FILTER_HIGH_HZ,
                "normalization": "per_window_per_channel_zscore", "normalization_epsilon": EPSILON,
                "contains_noise_windows": False, "source_index": str(index_path.resolve()),
                "source_index_sha256": sha256_file(index_path), "source_hdf5_count": 2,
            }
            for k, v in attrs.items(): out.attrs[k] = v
            for i, row in tqdm(index.iterrows(), total=len(index), desc="Preparing {}".format(split), unit="trace"):
                trace, sid, chunk = str(row.trace_name), str(row.source_id), str(row.source_chunk)
                if chunk not in handles:
                    raise RuntimeError("Unknown source chunk {}".format(chunk))
                group = handles[chunk]
                if trace not in group or tuple(group[trace].shape) != (6000, 3):
                    raise RuntimeError("Missing or bad source waveform {}".format(trace))
                start, end, p_index = crop_bounds(row.p_arrival_sample, 6000, split, trace)
                waveform, constant = preprocess(group[trace][start:end, :], sos)
                near_constant += constant
                ds["waveforms"][i], ds["p_indices"][i], ds["crop_starts"][i] = waveform, p_index, start
                ds["original_p_samples"][i], ds["trace_names"][i], ds["source_ids"][i], ds["source_chunks"][i] = float(row.p_arrival_sample), trace, sid, chunk
                histogram[p_index - P_INDEX_MIN] += 1
            out.attrs["near_constant_channel_count"] = near_constant
            out.create_dataset("p_index_histogram", data=histogram)
            out.flush()
        os.replace(tmp, output_path)
    except Exception:
        if tmp.exists(): tmp.unlink()
        raise
    return verify(output_path, len(index))


def verify(path, count):
    with h5py.File(path, "r") as handle:
        if handle["waveforms"].shape != (count, WINDOW_SAMPLES, 3):
            raise RuntimeError("Cache shape mismatch")
        p = handle["p_indices"][:]
        starts = handle["crop_starts"][:]
        original = np.rint(handle["original_p_samples"][:])
        if p.min() < P_INDEX_MIN or p.max() > P_INDEX_MAX or np.max(np.abs(starts + p - original)) != 0:
            raise RuntimeError("P alignment/range failure")
        if int(handle["p_index_histogram"][:].sum()) != count:
            raise RuntimeError("P histogram failure")
        names = handle["trace_names"].asstr()[:]
        if len(set(names.tolist())) != count:
            raise RuntimeError("Duplicate cache trace names")
        nonfinite, max_mean, max_std_error = 0, 0.0, 0.0
        for start in range(0, count, 256):
            batch = handle["waveforms"][start:start + 256]
            nonfinite += int(batch.size - np.isfinite(batch).sum())
            max_mean = max(max_mean, float(np.abs(batch.mean(axis=1)).max()))
            std = batch.std(axis=1)
            valid = std > 0.5
            if valid.any(): max_std_error = max(max_std_error, float(np.abs(std[valid] - 1.0).max()))
        if nonfinite or max_mean > 1e-4 or max_std_error > 1e-3:
            raise RuntimeError("Cache numerical verification failure")
        hist = handle["p_index_histogram"][:]
        occupied = np.flatnonzero(hist) + P_INDEX_MIN
        return {
            "path": str(path.resolve()), "bytes": path.stat().st_size, "records": count,
            "waveform_shape": list(handle["waveforms"].shape), "waveform_dtype": str(handle["waveforms"].dtype),
            "p_index_min": int(p.min()), "p_index_median": float(np.median(p)), "p_index_max": int(p.max()),
            "occupied_p_positions": int(len(occupied)), "empty_p_positions": int(WINDOW_SAMPLES - 199 - len(occupied)),
            "max_p_alignment_error_samples": 0.0, "nonfinite_cached_values": nonfinite,
            "maximum_abs_channel_mean": max_mean, "maximum_channel_std_error": max_std_error,
            "near_constant_channel_count": int(handle.attrs["near_constant_channel_count"]),
            "chunk_counts": {c: int(np.count_nonzero(handle["source_chunks"].asstr()[:] == c)) for c in SOURCE_HDF5},
            "source_index_sha256": str(handle.attrs["source_index_sha256"]),
        }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [OUTPUT_DIR / name for name in OUTPUT_FILES.values()] + [OUTPUT_DIR / "window_cache_summary.json"]
    if any(p.exists() for p in targets):
        raise FileExistsError("Refusing to overwrite existing 10 s cache outputs")
    opened = {name: h5py.File(path, "r") for name, path in SOURCE_HDF5.items()}
    try:
        groups = {name: source_group(handle) for name, handle in opened.items()}
        summaries = {}
        for split in ("train", "validation", "test"):
            summaries[split] = build_split(split, INDEX_DIR / INDEX_FILES[split], OUTPUT_DIR / OUTPUT_FILES[split], groups)
    finally:
        for handle in opened.values(): handle.close()
    result = {
        "preprocessing_contract": {
            "sampling_rate_hz": SAMPLING_RATE_HZ, "window_samples": WINDOW_SAMPLES,
            "channel_order": ["E", "N", "Z"], "p_index_range": [P_INDEX_MIN, P_INDEX_MAX],
            "detrend": "demean_then_linear", "filter": {"type": "Butterworth SOS zero-phase", "order": FILTER_ORDER, "low_hz": FILTER_LOW_HZ, "high_hz": FILTER_HIGH_HZ},
            "normalization": "per-window per-channel z-score", "normalization_epsilon": EPSILON,
            "contains_noise_windows": False, "crop_seed": SEED,
        }, "splits": summaries,
    }
    tmp = OUTPUT_DIR / "window_cache_summary.json.tmp"
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, OUTPUT_DIR / "window_cache_summary.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
