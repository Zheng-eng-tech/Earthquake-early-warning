"""Create leakage-safe chunk2+chunk3 indexes for 10 s TinyPhase-GRU."""

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260823
TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE = 40000, 5000, 5000
MINIMUM_P_SAMPLE = 900.0
SOURCES = {
    "chunk2": {
        "csv": ROOT / "data" / "raw" / "STEAD" / "chunk2.csv",
        "hdf5": ROOT / "data" / "raw" / "STEAD" / "chunk2.hdf5",
    },
    "chunk3": {
        "csv": Path(r"G:\Matlab_IC\Main_project_matlab\data\STEAD\chunk3 (1)\chunk3.csv"),
        "hdf5": Path(r"G:\Matlab_IC\Main_project_matlab\data\STEAD\chunk3 (1)\chunk3.hdf5"),
    },
}
LOCKED_XLSX = ROOT / "data" / "locked_test" / "ATKA" / "ATKA_test50_selection_and_truth.xlsx"
OUTPUT_DIR = ROOT / "data" / "indexes_10s_p100_900"
LOCKED_SHEETS = ("Final_Test_50", "MAG_Calibration_100")


def norm(series):
    return series.astype("string").str.strip()


def atomic_csv(frame, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_locked():
    frames = []
    for sheet in LOCKED_SHEETS:
        frame = pd.read_excel(LOCKED_XLSX, sheet_name=sheet, dtype={"trace_name": "string", "source_id": "string"})
        frame["trace_name"], frame["source_id"] = norm(frame["trace_name"]), norm(frame["source_id"])
        frame["locked_subset"] = sheet
        frames.append(frame[["trace_name", "source_id", "locked_subset"]])
    locked = pd.concat(frames, ignore_index=True)
    if len(locked) != 150 or locked["trace_name"].nunique() != 150 or locked["source_id"].nunique() != 150:
        raise RuntimeError("Locked workbook must contain 150 unique traces and source_ids")
    return locked


def read_source(name, csv_path, locked_sources):
    frame = pd.read_csv(csv_path, low_memory=False)
    required = {"trace_name", "source_id", "trace_category", "p_status", "p_arrival_sample"}
    if required.difference(frame.columns):
        raise RuntimeError("{} lacks columns {}".format(name, sorted(required.difference(frame.columns))))
    frame["trace_name"], frame["source_id"] = norm(frame["trace_name"]), norm(frame["source_id"])
    frame["p_arrival_sample"] = pd.to_numeric(frame["p_arrival_sample"], errors="coerce")
    locked_mask = frame["source_id"].isin(locked_sources)
    excluded = frame.loc[locked_mask].copy()
    excluded["source_chunk"] = name
    eligible_mask = (
        norm(frame["trace_category"]).str.lower().eq("earthquake_local")
        & norm(frame["p_status"]).str.lower().eq("manual")
        & frame["p_arrival_sample"].ge(MINIMUM_P_SAMPLE)
        & frame["trace_name"].notna() & frame["source_id"].notna() & ~locked_mask
    )
    eligible = frame.loc[eligible_mask].copy()
    eligible["source_chunk"] = name
    return len(frame), eligible, excluded


def verify_hdf5(frame):
    audit = {}
    for chunk, part in frame.groupby("source_chunk"):
        missing, bad = [], []
        with h5py.File(SOURCES[chunk]["hdf5"], "r") as handle:
            group = handle["data"] if "data" in handle else handle
            for trace in part["trace_name"].astype(str):
                if trace not in group:
                    missing.append(trace)
                elif tuple(group[trace].shape) != (6000, 3):
                    bad.append(trace)
        if missing or bad:
            raise RuntimeError("{} HDF5 verification failed: missing={}, bad_shape={}".format(chunk, missing[:5], bad[:5]))
        audit[chunk] = {"records": len(part), "missing": 0, "bad_shape": 0}
    return audit


def sample(pool, size, seed, split):
    if len(pool) < size:
        raise RuntimeError("{} pool {} is smaller than requested {}".format(split, len(pool), size))
    out = pool.sample(n=size, replace=False, random_state=seed).copy()
    out["split"], out["split_seed"] = split, seed
    return out.sort_values(["source_chunk", "trace_name"], kind="stable").reset_index(drop=True)


def main():
    outputs = [OUTPUT_DIR / n for n in ("train_40000.csv", "validation_5000.csv", "test_5000.csv", "excluded_locked.csv", "split_summary.json")]
    if any(p.exists() for p in outputs) and "--overwrite" not in sys.argv:
        raise FileExistsError("Refusing to overwrite existing 10 s indexes")
    for item in [LOCKED_XLSX] + [v[k] for v in SOURCES.values() for k in ("csv", "hdf5")]:
        if not item.is_file():
            raise FileNotFoundError(item)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    locked = read_locked()
    locked_sources = set(locked["source_id"].astype(str))
    eligible_parts, excluded_parts, csv_counts = [], [], {}
    for chunk, paths in SOURCES.items():
        count, eligible, excluded = read_source(chunk, paths["csv"], locked_sources)
        csv_counts[chunk] = count
        eligible_parts.append(eligible)
        excluded_parts.append(excluded)
    eligible = pd.concat(eligible_parts, ignore_index=True)
    if eligible["trace_name"].duplicated().any():
        raise RuntimeError("Duplicate trace_name across combined candidates")
    excluded = pd.concat(excluded_parts, ignore_index=True)

    source_ids = np.asarray(sorted(eligible["source_id"].astype(str).unique()), dtype=object)
    rng = np.random.default_rng(SEED)
    rng.shuffle(source_ids)
    n = len(source_ids)
    validation_sources = set(source_ids[: n // 10])
    test_sources = set(source_ids[n // 10 : 2 * (n // 10)])
    train_sources = set(source_ids[2 * (n // 10) :])
    pools = {
        "train": eligible[eligible["source_id"].isin(train_sources)],
        "validation": eligible[eligible["source_id"].isin(validation_sources)],
        "test": eligible[eligible["source_id"].isin(test_sources)],
    }
    selected = {
        "train": sample(pools["train"], TRAIN_SIZE, SEED + 1, "train"),
        "validation": sample(pools["validation"], VALIDATION_SIZE, SEED + 2, "validation"),
        "test": sample(pools["test"], TEST_SIZE, SEED + 3, "test"),
    }
    source_sets = {k: set(v["source_id"].astype(str)) for k, v in selected.items()}
    intersections = {
        "train_vs_validation": len(source_sets["train"] & source_sets["validation"]),
        "train_vs_test": len(source_sets["train"] & source_sets["test"]),
        "validation_vs_test": len(source_sets["validation"] & source_sets["test"]),
    }
    if any(intersections.values()) or set().union(*source_sets.values()) & locked_sources:
        raise RuntimeError("source_id leakage detected")
    all_selected = pd.concat(selected.values(), ignore_index=True)
    hdf5_audit = verify_hdf5(all_selected)
    output_columns = [c for c in ["split", "split_seed", "source_chunk", "trace_name", "source_id", "network_code", "receiver_code", "receiver_type", "p_arrival_sample", "p_status", "p_weight", "s_arrival_sample", "s_status", "source_magnitude", "source_magnitude_type", "source_distance_km", "source_depth_km", "back_azimuth_deg"] if c in selected["train"].columns]
    atomic_csv(selected["train"][output_columns], OUTPUT_DIR / "train_40000.csv")
    atomic_csv(selected["validation"][output_columns], OUTPUT_DIR / "validation_5000.csv")
    atomic_csv(selected["test"][output_columns], OUTPUT_DIR / "test_5000.csv")
    atomic_csv(excluded[[c for c in ["source_chunk", "trace_name", "source_id", "network_code", "receiver_code", "p_arrival_sample", "locked_subset"] if c in excluded.columns]], OUTPUT_DIR / "excluded_locked.csv")
    summary = {
        "seed": SEED,
        "criteria": {"trace_category": "earthquake_local", "p_status": "manual", "minimum_p_arrival_sample": MINIMUM_P_SAMPLE, "include_noise": False},
        "sources": {k: {"csv": str(v["csv"]), "hdf5": str(v["hdf5"]), "csv_rows": csv_counts[k]} for k, v in SOURCES.items()},
        "locked": {"trace_count": len(locked), "source_id_count": len(locked_sources), "excluded_rows_across_chunks": len(excluded)},
        "eligible": {"rows": len(eligible), "source_ids": int(eligible["source_id"].nunique()), "chunk_counts": {k: int(v) for k, v in eligible["source_chunk"].value_counts().items()}},
        "pool_rows": {k: len(v) for k, v in pools.items()},
        "selected": {k: {"rows": len(v), "source_ids": int(v["source_id"].nunique()), "chunk_counts": {c: int(n) for c, n in v["source_chunk"].value_counts().items()}, "p_arrival_sample": {"min": float(v["p_arrival_sample"].min()), "median": float(v["p_arrival_sample"].median()), "max": float(v["p_arrival_sample"].max())}} for k, v in selected.items()},
        "audit": {"source_id_intersections": intersections, "locked_source_id_hits": 0, "duplicate_selected_trace_names": int(all_selected["trace_name"].duplicated().sum()), "hdf5": hdf5_audit},
    }
    atomic_json(summary, OUTPUT_DIR / "split_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise
