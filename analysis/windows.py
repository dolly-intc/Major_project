"""
Sliding window extraction for phase discovery.

Creates fixed-size sliding windows from timeline data, suitable for
autoencoder training. Each window captures 2 seconds of execution
context (4 bins at 0.5s each).

Usage:
    from analysis.windows import create_windows
    X, meta = create_windows(df_features)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# Default window parameters
WINDOW_SIZE = 4      # 4 bins = 2 seconds of context
HORIZON_BINS = 2     # 2 bins = 1 second lookahead (for Model B, not used here)
MIN_CONSECUTIVE = 6  # Minimum consecutive bins needed (window + horizon)


def _get_longest_consecutive_run(bins: np.ndarray, bin_size: float = 0.5) -> Tuple[int, int]:
    """
    Find the longest consecutive run of bins for a thread.

    Args:
        bins: Sorted array of bin_start_sec values
        bin_size: Expected gap between consecutive bins

    Returns:
        (start_idx, length) of the longest consecutive run
    """
    if len(bins) == 0:
        return 0, 0
    if len(bins) == 1:
        return 0, 1

    # Find gaps larger than expected
    gaps = np.diff(bins)
    expected_gap = bin_size * 1.5  # Allow some tolerance

    # Find break points (where gap is too large)
    breaks = np.where(gaps > expected_gap)[0]

    # Split into runs
    run_starts = np.concatenate([[0], breaks + 1])
    run_ends = np.concatenate([breaks + 1, [len(bins)]])
    run_lengths = run_ends - run_starts

    # Find longest
    best_idx = np.argmax(run_lengths)
    return int(run_starts[best_idx]), int(run_lengths[best_idx])


def create_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    feature_cols: Optional[list] = None,
    drop_bin_zero: bool = True,
    min_consecutive: int = MIN_CONSECUTIVE,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Create sliding windows from timeline features.

    Args:
        df: DataFrame with features (from extract_features or get_phase_features)
        window_size: Number of consecutive bins per window
        feature_cols: List of feature columns to include (auto-detected if None)
        drop_bin_zero: Drop the first bin (startup noise)
        min_consecutive: Minimum consecutive bins needed per thread

    Returns:
        X: numpy array of shape (n_windows, window_size * n_features)
        meta: DataFrame with window metadata (run_id, tid, window_start_sec, etc.)
    """
    # Auto-detect feature columns
    if feature_cols is None:
        meta_cols = {"run_id", "workload", "core_type", "thread_count", "rep",
                     "bin_start_sec", "bin_end_sec", "tid", "pid", "source_folder", "thread"}
        feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"Using {len(feature_cols)} features: {feature_cols}")

    all_windows = []
    all_meta = []

    # Group by (run_id, tid) - each thread in each run
    groups = df.groupby(["run_id", "tid"])
    n_groups = len(groups)

    skipped_short = 0
    skipped_nonconsec = 0

    for (run_id, tid), group in groups:
        # Sort by bin_start_sec
        group = group.sort_values("bin_start_sec").reset_index(drop=True)

        # Drop bin 0 (startup noise)
        if drop_bin_zero:
            group = group[group["bin_start_sec"] > 0].reset_index(drop=True)

        if len(group) < min_consecutive:
            skipped_short += 1
            continue

        # Find longest consecutive run
        bins = group["bin_start_sec"].values
        start_idx, run_length = _get_longest_consecutive_run(bins)

        if run_length < min_consecutive:
            skipped_nonconsec += 1
            continue

        # Extract the consecutive portion
        consec = group.iloc[start_idx:start_idx + run_length]

        # Get feature values (make a copy since we'll modify it)
        feat_values = consec[feature_cols].values.copy()

        # Handle NaN: fill with column mean (computed from this thread's data)
        col_means = np.nanmean(feat_values, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)  # If all NaN, use 0
        for j in range(feat_values.shape[1]):
            mask = np.isnan(feat_values[:, j])
            feat_values[mask, j] = col_means[j]

        # Create sliding windows
        n_windows = run_length - window_size + 1
        for i in range(n_windows):
            window = feat_values[i:i + window_size].flatten()
            all_windows.append(window)

            # Metadata for this window
            window_start_sec = consec.iloc[i]["bin_start_sec"]
            window_end_sec = consec.iloc[i + window_size - 1]["bin_end_sec"]

            all_meta.append({
                "run_id": run_id,
                "tid": tid,
                "workload": consec.iloc[0]["workload"],
                "core_type": consec.iloc[0]["core_type"],
                "thread_count": consec.iloc[0]["thread_count"],
                "rep": consec.iloc[0]["rep"],
                "window_start_sec": window_start_sec,
                "window_end_sec": window_end_sec,
                "window_idx": i,
            })

    if not all_windows:
        raise ValueError("No windows created - check data or parameters")

    X = np.array(all_windows, dtype=np.float32)
    meta = pd.DataFrame(all_meta)

    print(f"Created {len(X)} windows from {n_groups} thread-runs")
    print(f"Skipped: {skipped_short} too short, {skipped_nonconsec} non-consecutive")
    print(f"Window shape: {X.shape} ({window_size} bins × {len(feature_cols)} features)")

    return X, meta


def create_windows_by_workload(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    feature_cols: Optional[list] = None,
) -> dict:
    """
    Create windows grouped by workload.

    Returns dict: {workload: (X, meta)}
    """
    result = {}
    for workload in df["workload"].unique():
        df_wl = df[df["workload"] == workload]
        try:
            X, meta = create_windows(df_wl, window_size, feature_cols)
            result[workload] = (X, meta)
        except ValueError as e:
            print(f"Warning: {workload}: {e}")
    return result


def get_window_statistics(meta: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics for windows by workload and core type."""
    return meta.groupby(["workload", "core_type"]).agg({
        "run_id": "nunique",
        "tid": "nunique",
        "window_idx": "count",
    }).rename(columns={
        "run_id": "n_runs",
        "tid": "n_threads",
        "window_idx": "n_windows",
    })


if __name__ == "__main__":
    from load_dataset import load_timelines
    from features import extract_features, get_phase_features, PHASE_FEATURES

    print("=== Loading Data ===")
    df = load_timelines()

    print("\n=== Extracting Features ===")
    df_feat = extract_features(df)

    # Use phase features for windowing
    df_phase = get_phase_features(df_feat)

    # Get actual feature columns (excluding metadata)
    feature_cols = [c for c in PHASE_FEATURES if c in df_phase.columns]

    print("\n=== Creating Windows ===")
    X, meta = create_windows(df_phase, feature_cols=feature_cols)

    print("\n=== Window Statistics ===")
    print(get_window_statistics(meta))

    print("\n=== Windows per Workload ===")
    print(meta.groupby("workload").size())

    print("\n=== Sample Window Values ===")
    print(f"X min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}")
