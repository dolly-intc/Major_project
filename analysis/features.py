"""
Feature extraction for phase discovery.

Extracts scale-invariant features from VTune hardware counters.
All features are ratios (0-1 or small integers), not raw counts.

P-cores and E-cores expose different PMU events, so the phase features are computed per core
type from each core's own counters (2026-10-21; see CORE_COUNTERS). Before this, the top-down
features read 0 on every P-core bin and the L1/L3 miss rates read 0 on every E-core bin, which made
Model A's phases split by core type.

Missing vs. zero: VTune multiplexes ~217 events, so a large counter (e.g. P-core backend-bound
slots) can read 0 in a 0.5 s bin simply because it was not sampled. Those bins are NaN and carried
forward from the same thread's previous bin (causal, so the live service can do the same).
Rare-event counters (memory stalls, mispredicts) read 0 because the event was below one sample,
so their 0 is kept as 0. They are recorded in steps of one VTune sample (SAMPLE_UNIT), so their
ratio is only trusted when the denominator is at least MIN_UNITS samples' worth (one sample <= 5%
or 10% of the value); otherwise the bin is NaN and carried forward. Without this, low-activity
bins gave stall fractions up to 13x and mispredict rates up to 2. Values are then clipped to
physical limits (fractions to [0, 1], IPC to [0, 8] = sustained pipeline width).

Usage:
    from analysis.features import extract_features
    df_features = extract_features(df_timeline)
"""

import numpy as np
import pandas as pd


# Top-down analysis columns (use _P variants for "precise" counts)
TOPDOWN_COLS = [
    "TOPDOWN_FE_BOUND.ALL_P",      # Frontend bound
    "TOPDOWN_BE_BOUND.ALL_P",      # Backend bound
    "TOPDOWN_BAD_SPECULATION.ALL_P",  # Bad speculation
    "TOPDOWN_RETIRING.ALL_P",      # Retiring (useful work)
]

# Pre-computed metrics in the timeline CSV
PRECOMPUTED = ["ipc", "l1_miss_rate", "l3_miss_rate"]

# Additional derived metrics
DERIVED_METRICS = {
    # Branch prediction
    "branch_mispredict_rate": (
        "BR_MISP_RETIRED.ALL_BRANCHES",
        "BR_INST_RETIRED.ALL_BRANCHES"
    ),
    # Memory hierarchy
    "l2_miss_rate": (
        "MEM_LOAD_RETIRED.L3_HIT",  # L2 miss = L3 hit + L3 miss
        "MEM_LOAD_RETIRED.L1_MISS"  # Denominator: L1 misses
    ),
    # Frontend efficiency
    "dsb_coverage": (
        "IDQ.DSB_UOPS",  # Uops from decoded stream buffer (cache)
        "UOPS_ISSUED.ANY"  # Total uops issued
    ),
    # Execution port utilization
    "port_saturation": (
        "EXE_ACTIVITY.2_3_PORTS_UTIL",  # Cycles using 2-3 ports
        "CPU_CLK_UNHALTED.THREAD"
    ),
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute ratio, returning 0 where denominator is 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        result = result.replace([np.inf, -np.inf], 0)
        result = result.fillna(0)
    return result


def extract_topdown_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract top-down fractions (sum to ~1 per bin).

    Returns DataFrame with columns: fe_bound, be_bound, bad_spec, retiring
    """
    result = pd.DataFrame(index=df.index)

    # Check which columns exist
    available = [c for c in TOPDOWN_COLS if c in df.columns]
    if not available:
        raise ValueError(f"No top-down columns found. Expected: {TOPDOWN_COLS}")

    # Sum of all top-down categories
    total = df[available].sum(axis=1)

    # Compute fractions
    if "TOPDOWN_FE_BOUND.ALL_P" in df.columns:
        result["fe_bound"] = _safe_ratio(df["TOPDOWN_FE_BOUND.ALL_P"], total)
    if "TOPDOWN_BE_BOUND.ALL_P" in df.columns:
        result["be_bound"] = _safe_ratio(df["TOPDOWN_BE_BOUND.ALL_P"], total)
    if "TOPDOWN_BAD_SPECULATION.ALL_P" in df.columns:
        result["bad_spec"] = _safe_ratio(df["TOPDOWN_BAD_SPECULATION.ALL_P"], total)
    if "TOPDOWN_RETIRING.ALL_P" in df.columns:
        result["retiring"] = _safe_ratio(df["TOPDOWN_RETIRING.ALL_P"], total)

    return result


def extract_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract additional derived ratio metrics."""
    result = pd.DataFrame(index=df.index)

    for name, (num_col, denom_col) in DERIVED_METRICS.items():
        if num_col in df.columns and denom_col in df.columns:
            result[name] = _safe_ratio(df[num_col], df[denom_col])

    return result


def extract_features(
    df: pd.DataFrame,
    include_precomputed: bool = True,
    include_topdown: bool = True,
    include_derived: bool = True,
    copy_metadata: bool = True,
) -> pd.DataFrame:
    """
    Extract scale-invariant features from timeline data.

    Args:
        df: Timeline DataFrame from load_timelines()
        include_precomputed: Include ipc, l1_miss_rate, l3_miss_rate
        include_topdown: Include top-down fractions
        include_derived: Include derived ratio metrics
        copy_metadata: Include run_id, workload, etc.

    Returns:
        DataFrame with feature columns and optional metadata
    """
    parts = []

    # Metadata columns to preserve
    meta_cols = ["run_id", "workload", "core_type", "thread_count", "rep",
                 "bin_start_sec", "bin_end_sec", "tid", "pid"]

    if copy_metadata:
        available_meta = [c for c in meta_cols if c in df.columns]
        parts.append(df[available_meta].copy())

    # Pre-computed metrics
    if include_precomputed:
        precomp_cols = [c for c in PRECOMPUTED if c in df.columns]
        if precomp_cols:
            parts.append(df[precomp_cols].copy())

    # Top-down fractions
    if include_topdown:
        topdown = extract_topdown_fractions(df)
        parts.append(topdown)

    # Derived metrics
    if include_derived:
        derived = extract_derived_metrics(df)
        parts.append(derived)

    result = pd.concat(parts, axis=1)
    # Core-aware phase features replace the single-counter-set versions of the same names
    core_aware = extract_core_aware_features(df)
    result = result.drop(columns=[c for c in core_aware.columns if c in result.columns])
    result = pd.concat([result, core_aware], axis=1)

    # Drop any all-zero columns (counters not collected)
    feature_cols = [c for c in result.columns if c not in meta_cols]
    zero_cols = [c for c in feature_cols if (result[c] == 0).all() and c not in PHASE_FEATURES]
    if zero_cols:
        result = result.drop(columns=zero_cols)
        print(f"Dropped {len(zero_cols)} all-zero columns: {zero_cols}")

    return result


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature column names (excluding metadata)."""
    meta_cols = {"run_id", "workload", "core_type", "thread_count", "rep",
                 "bin_start_sec", "bin_end_sec", "tid", "pid", "source_folder"}
    return [c for c in df.columns if c not in meta_cols]


def describe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics for feature columns."""
    feature_cols = get_feature_columns(df)
    return df[feature_cols].describe()


# Per-core-type counters for the phase features. The four top-down counters are normalised by
# their own sum, so the shares add to 1 on both core types (both are 8 slots/cycle on this chip).
CORE_COUNTERS = {
    "p": {
        "retiring": "UOPS_RETIRED.SLOTS",
        "fe_bound": "IDQ_BUBBLES.CORE",
        "be_bound": "TOPDOWN.BACKEND_BOUND_SLOTS",
        "bad_spec": "TOPDOWN.BR_MISPREDICT_SLOTS",   # excludes machine clears (rare here)
        "mem_stall": ["MEMORY_STALLS.L2", "MEMORY_STALLS.L3", "MEMORY_STALLS.MEM"],
        "dram_stall": ["MEMORY_STALLS.MEM"],
    },
    "e": {
        "retiring": "TOPDOWN_RETIRING.ALL_P",
        "fe_bound": "TOPDOWN_FE_BOUND.ALL_P",
        "be_bound": "TOPDOWN_BE_BOUND.ALL_P",
        "bad_spec": "TOPDOWN_BAD_SPECULATION.ALL_P",
        "mem_stall": ["MEM_BOUND_STALLS_LOAD.ALL"],   # = L2_HIT + LLC_HIT + LLC_MISS
        "dram_stall": ["MEM_BOUND_STALLS_LOAD.LLC_MISS"],
    },
}
TOPDOWN_FEATURES = ["retiring", "fe_bound", "be_bound", "bad_spec"]

# Size of one VTune sample (sample-after value) for the rare-event numerators, per core type;
# measured as the smallest non-zero reading in stage1_v4.
SAMPLE_UNIT = {
    "p": {"stall": 25_000_075, "mispredict": 10_000_225},
    "e": {"stall": 10_000_030, "mispredict": 2_000_030},
}
MIN_UNITS = {"stall": 20, "mispredict": 10}
IPC_MAX = 8.0

# Feature set definitions for different use cases
PHASE_FEATURES = [
    # Core efficiency
    "ipc",
    "retiring",
    # Bottleneck type
    "fe_bound",
    "be_bound",
    "bad_spec",
    # Memory behavior (fraction of cycles stalled on L2/L3/DRAM, and on DRAM alone)
    "mem_stall_frac",
    "dram_stall_frac",
    # Branch behavior
    "branch_mispredict_rate",
]
# Before 2026-10-21; l1/l3 miss rates exist only on P-cores. Kept for reading old results.
LEGACY_PHASE_FEATURES = ["ipc", "retiring", "fe_bound", "be_bound", "bad_spec",
                         "l1_miss_rate", "l3_miss_rate", "branch_mispredict_rate"]


def _ratio_or_nan(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN where den <= 0 (not measured)."""
    return (num / den.where(den > 0)).astype(float)


def extract_core_aware_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    PHASE_FEATURES from each core type's own counters, before the carry-forward fill.

    NaN marks "not measured in this bin": a top-down counter of a busy core reading 0, or a
    zero denominator. Rows of an unknown core type are all NaN.
    """
    out = pd.DataFrame(index=df.index, columns=PHASE_FEATURES, dtype=float)
    cycles = df["CPU_CLK_UNHALTED.THREAD"]
    branches = df["BR_INST_RETIRED.ALL_BRANCHES"]
    for ct, cols in CORE_COUNTERS.items():
        m = df["core_type"] == ct
        d = df.loc[m]
        td = d[[cols[f] for f in TOPDOWN_FEATURES]]
        # retiring / fe / be are never truly 0 on a busy core; bad_spec can be
        measured = (td[[cols["retiring"], cols["fe_bound"], cols["be_bound"]]] > 0).all(axis=1)
        total = td.sum(axis=1).where(measured)
        for f in TOPDOWN_FEATURES:
            out.loc[m, f] = d[cols[f]] / total
        unit = SAMPLE_UNIT[ct]
        cyc = cycles[m].where(cycles[m] >= MIN_UNITS["stall"] * unit["stall"])
        out.loc[m, "mem_stall_frac"] = _ratio_or_nan(d[cols["mem_stall"]].sum(axis=1), cyc)
        out.loc[m, "dram_stall_frac"] = _ratio_or_nan(d[cols["dram_stall"]].sum(axis=1), cyc)
        br = branches[m].where(branches[m] >= MIN_UNITS["mispredict"] * unit["mispredict"])
        out.loc[m, "branch_mispredict_rate"] = _ratio_or_nan(d["BR_MISP_RETIRED.ALL_BRANCHES"], br)
    out["ipc"] = _ratio_or_nan(df["INST_RETIRED.ANY"], cycles).clip(upper=IPC_MAX)
    return out.clip(lower=0).pipe(lambda o: o.assign(**{c: o[c].clip(upper=1.0) for c in PHASE_FEATURES[1:]}))


def fill_unmeasured(df: pd.DataFrame, feature_cols: list = PHASE_FEATURES) -> pd.DataFrame:
    """
    Carry each thread's last measured value forward over unmeasured (NaN) bins, then drop the
    bins at the start of a thread that have no earlier measurement. Causal: uses past bins only.
    """
    df = df.sort_values(["run_id", "tid", "bin_start_sec"]).copy()
    filled = df.groupby(["run_id", "tid"], sort=False)[feature_cols].ffill()
    n_nan = int(df[feature_cols].isna().any(axis=1).sum())
    df[feature_cols] = filled
    keep = filled.notna().all(axis=1)
    print(f"Carry-forward: {n_nan:,} of {len(df):,} bins had an unmeasured feature; "
          f"{int((~keep).sum()):,} leading bins dropped")
    return df[keep]


def get_phase_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the minimal feature set for phase discovery.

    Returns DataFrame with only the PHASE_FEATURES columns plus metadata, with unmeasured bins
    filled by fill_unmeasured().
    """
    meta_cols = ["run_id", "workload", "core_type", "thread_count", "rep",
                 "bin_start_sec", "bin_end_sec", "tid"]

    available_meta = [c for c in meta_cols if c in df.columns]
    available_features = [c for c in PHASE_FEATURES if c in df.columns]

    return fill_unmeasured(df[available_meta + available_features].copy(), available_features)


if __name__ == "__main__":
    from load_dataset import load_timelines

    print("=== Loading Data ===")
    df = load_timelines()

    print("\n=== Extracting Features ===")
    df_feat = extract_features(df)
    print(f"Shape: {df_feat.shape}")
    print(f"Feature columns: {get_feature_columns(df_feat)}")

    print("\n=== Feature Statistics ===")
    print(describe_features(df_feat))

    print("\n=== Phase Features ===")
    df_phase = get_phase_features(df_feat)
    print(f"Columns: {list(df_phase.columns)}")
    print(df_phase[get_feature_columns(df_phase)].describe())
