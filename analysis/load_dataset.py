"""
Dataset loader for Stage 1 VTune timeline data.

Loads timeline CSVs from stage1_v4 (non-7zip) and stage1_v4_7zip (7zip retry),
returning a unified long-format DataFrame.

Usage:
    from analysis.load_dataset import load_timelines
    df = load_timelines()
"""

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd


# Default data root
DATA_ROOT = Path(__file__).parent.parent / "data" / "raw" / "FZA633700090_Ultra9-290HX"

# Folders to load from
STAGE1_MAIN = "stage1_v4"
STAGE1_7ZIP = "stage1_v4_7zip"


def _load_run_metadata(run_dir: Path) -> Optional[dict]:
    """Load results.json for a run directory."""
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return None
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _load_timeline_csv(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load timeline CSV for a run directory."""
    csv_path = run_dir / "vtune_hw_events_timeline.csv"
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def load_timelines(
    data_root: Optional[Path] = None,
    workloads: Optional[list] = None,
    skip_failed: bool = True,
) -> pd.DataFrame:
    """
    Load all timeline data into a single long-format DataFrame.

    Args:
        data_root: Path to machine data folder (default: FZA633700090_Ultra9-290HX)
        workloads: List of workloads to include (default: all)
        skip_failed: Skip runs with status != "ok" (default: True)

    Returns:
        DataFrame with columns:
        - run_id, workload, core_type, thread_count, rep (from metadata)
        - bin_start_sec, bin_end_sec, tid, pid (from timeline)
        - All hardware counter columns
        - ipc, l1_miss_rate, l3_miss_rate
    """
    if data_root is None:
        data_root = DATA_ROOT
    else:
        data_root = Path(data_root)

    all_frames = []
    loaded_runs = set()
    skipped_count = 0

    # Load from both folders, prioritizing 7zip retry for 7zip workloads
    folders = [
        (data_root / STAGE1_7ZIP, {"7zip"}),  # 7zip from retry folder
        (data_root / STAGE1_MAIN, None),       # Everything else from main
    ]

    for folder, only_workloads in folders:
        if not folder.exists():
            print(f"Warning: {folder} not found, skipping")
            continue

        for run_dir in sorted(folder.iterdir()):
            if not run_dir.is_dir():
                continue
            if run_dir.name.startswith("__"):  # Skip stale folders
                continue

            # Parse run_id to get workload
            parts = run_dir.name.split("_")
            if len(parts) < 4:
                continue
            workload = parts[0]

            # Filter by workload list
            if only_workloads is not None and workload not in only_workloads:
                continue
            if workloads is not None and workload not in workloads:
                continue

            # Skip if we already loaded this run_id (7zip from retry takes priority)
            run_id = run_dir.name
            if run_id in loaded_runs:
                continue

            # Load metadata
            meta = _load_run_metadata(run_dir)
            if meta is None:
                skipped_count += 1
                continue

            # Skip failed runs if requested
            if skip_failed and meta.get("status") != "ok":
                skipped_count += 1
                continue

            # Load timeline
            df = _load_timeline_csv(run_dir)
            if df is None or df.empty:
                skipped_count += 1
                continue

            # Add metadata columns
            df["run_id"] = run_id
            df["workload"] = meta.get("workload", workload)
            df["core_type"] = meta.get("core_type")
            df["thread_count"] = meta.get("thread_count")
            df["rep"] = meta.get("rep")
            df["source_folder"] = folder.name

            all_frames.append(df)
            loaded_runs.add(run_id)

    if not all_frames:
        raise ValueError(f"No timeline data found in {data_root}")

    result = pd.concat(all_frames, ignore_index=True)

    print(f"Loaded {len(loaded_runs)} runs ({len(result)} rows)")
    print(f"Skipped {skipped_count} runs (failed or missing data)")
    print(f"Workloads: {sorted(result['workload'].unique())}")

    return result


def load_aggregate(
    data_root: Optional[Path] = None,
    workloads: Optional[list] = None,
    skip_failed: bool = True,
) -> pd.DataFrame:
    """
    Load aggregate (non-timeline) VTune data for all runs.

    Returns one row per run with aggregate metrics from vtune_hw_events.csv
    and results.json metadata.
    """
    if data_root is None:
        data_root = DATA_ROOT
    else:
        data_root = Path(data_root)

    all_rows = []
    loaded_runs = set()

    folders = [
        (data_root / STAGE1_7ZIP, {"7zip"}),
        (data_root / STAGE1_MAIN, None),
    ]

    for folder, only_workloads in folders:
        if not folder.exists():
            continue

        for run_dir in sorted(folder.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("__"):
                continue

            parts = run_dir.name.split("_")
            if len(parts) < 4:
                continue
            workload = parts[0]

            if only_workloads is not None and workload not in only_workloads:
                continue
            if workloads is not None and workload not in workloads:
                continue

            run_id = run_dir.name
            if run_id in loaded_runs:
                continue

            meta = _load_run_metadata(run_dir)
            if meta is None:
                continue
            if skip_failed and meta.get("status") != "ok":
                continue

            # Load aggregate CSV
            agg_path = run_dir / "vtune_hw_events.csv"
            if not agg_path.exists():
                continue

            try:
                agg_df = pd.read_csv(agg_path)
                if agg_df.empty:
                    continue
                # Take first row (should be only row for aggregate)
                row = agg_df.iloc[0].to_dict()
            except Exception:
                continue

            # Add metadata
            row["run_id"] = run_id
            row["workload"] = meta.get("workload", workload)
            row["core_type"] = meta.get("core_type")
            row["thread_count"] = meta.get("thread_count")
            row["rep"] = meta.get("rep")
            row["energy_mj"] = meta.get("energy_mj")
            row["workload_span_sec"] = meta.get("workload_span_sec")
            row["ipc"] = meta.get("ipc")

            all_rows.append(row)
            loaded_runs.add(run_id)

    if not all_rows:
        raise ValueError(f"No aggregate data found in {data_root}")

    result = pd.DataFrame(all_rows)
    print(f"Loaded {len(result)} aggregate runs")

    return result


def get_run_summary(data_root: Optional[Path] = None) -> pd.DataFrame:
    """
    Get a summary of all runs: run_id, workload, core_type, thread_count, rep,
    status, timeline_bins, energy_mj.
    """
    if data_root is None:
        data_root = DATA_ROOT
    else:
        data_root = Path(data_root)

    rows = []
    seen = set()

    for folder in [data_root / STAGE1_7ZIP, data_root / STAGE1_MAIN]:
        if not folder.exists():
            continue
        for run_dir in sorted(folder.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("__"):
                continue

            run_id = run_dir.name
            parts = run_id.split("_")
            if len(parts) < 4:
                continue
            workload = parts[0]

            # 7zip from retry folder takes priority
            if run_id in seen:
                continue
            if folder.name == STAGE1_MAIN and workload == "7zip":
                continue  # Skip 7zip from main folder

            meta = _load_run_metadata(run_dir)
            if meta is None:
                continue

            timeline_path = run_dir / "vtune_hw_events_timeline.csv"
            has_timeline = timeline_path.exists()

            rows.append({
                "run_id": run_id,
                "workload": meta.get("workload", workload),
                "core_type": meta.get("core_type"),
                "thread_count": meta.get("thread_count"),
                "rep": meta.get("rep"),
                "status": meta.get("status"),
                "timeline_bins": meta.get("timeline_bins"),
                "energy_mj": meta.get("energy_mj"),
                "ipc": meta.get("ipc"),
                "has_timeline": has_timeline,
                "source": folder.name,
            })
            seen.add(run_id)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick test
    print("=== Run Summary ===")
    summary = get_run_summary()
    print(summary.groupby(["workload", "status"]).size().unstack(fill_value=0))
    print()

    print("=== Loading Timelines ===")
    df = load_timelines()
    print(f"\nShape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nRuns per workload:")
    print(df.groupby("workload")["run_id"].nunique())
