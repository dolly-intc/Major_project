"""
Phase catalogue for the progress evaluation: data and figures for the Model A phases.

Uses the frozen Model A artifacts (models/phase_discovery/) through model_b_v2.load_windows();
nothing is trained, so the phase ids match PROJECT_CONTEXT.md §3.

Writes to results/presentation/phases/:
  phase_summary.csv / phase_cards.md   one row / one card per phase
  stage2_samples.md                    windows and thread-runs per workload (Stage 2 sample counts)
  phase_workload_share.csv             phase x workload, share of the phase's windows
  transition_matrix.csv                P(next window's phase | current phase), same thread
  feature_profiles.png                 mean of the 8 features per phase (colour = z-score across phases)
  workload_composition.png             phase x workload heatmap
  dwell_time.png                       how long a thread stays in each phase
  transitions.png                      where a phase goes when it changes
  core_type_by_phase.png               share of each phase's windows that ran on P vs E cores
  example_timelines.png                phase sequence of one thread per real application and core type

Run:  python analysis/phase_catalogue.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis import model_b_v2 as v2
from analysis.features import PHASE_FEATURES
from analysis.dataset_catalogue import md_table, WORKLOAD_ORDER, REAL_APPS, CORE_COLORS, CORE_NAMES

OUT = v2.ROOT / "results" / "presentation" / "phases"
WINDOW_BINS = 4
STRIDE_SEC = 0.5  # windows advance one 0.5 s bin at a time

# Names from the 2026-10-21 cluster profiles (core-aware features; PROJECT_CONTEXT.md §3).
PHASE_NAMES = {
    1: "Main Compute",
    3: "Bad Speculation",
    2: "Frontend-Bound",
    0: "Memory-Bound",
}
K_SELECTED, SILHOUETTE = 4, 0.3185  # from analysis/model_a.py's k=4..12 sweep (results/model_a/train_log.txt)
FEATURE_LABELS = {
    "ipc": "IPC", "retiring": "retiring", "fe_bound": "frontend-bound", "be_bound": "backend-bound",
    "bad_spec": "bad speculation", "mem_stall_frac": "memory stalls", "dram_stall_frac": "DRAM stalls",
    "branch_mispredict_rate": "branch mispredict",
}
# Validated categorical palette, slots 1-8 in fixed order (assigned by phase size rank).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.tick_params(colors=INK_MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d0cfcb")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def thread_segments(df: pd.DataFrame):
    """Yield per-thread label arrays in time order."""
    for _, g in df.sort_values(["run_id", "tid", "window_idx"]).groupby(["run_id", "tid"], sort=False):
        yield g


def dwell_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Every maximal stretch of one phase within a thread, with its length in seconds.
    Stretches touching the start/end of a thread are kept but flagged (their true length is unknown)."""
    rows = []
    for g in thread_segments(df):
        lab = g["phase"].to_numpy()
        change = np.flatnonzero(np.diff(lab)) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(lab)]])
        for s, e in zip(starts, ends):
            rows.append({"phase": lab[s], "windows": e - s, "censored": s == 0 or e == len(lab)})
    out = pd.DataFrame(rows)
    out["seconds"] = out["windows"] * STRIDE_SEC
    return out


def transition_matrix(df: pd.DataFrame, k: int) -> np.ndarray:
    counts = np.zeros((k, k))
    for g in thread_segments(df):
        lab = g["phase"].to_numpy()
        np.add.at(counts, (lab[:-1], lab[1:]), 1)
    return counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X_scaled, emb, dists, labels, meta = v2.load_windows()
    k = dists.shape[1]
    sc = np.load(v2.ROOT / "models" / "phase_discovery" / "scaler.npz")
    X = X_scaled * sc["scale"] + sc["mean"]
    feats = [f for f in PHASE_FEATURES]
    assert X.shape[1] == WINDOW_BINS * len(feats)
    per_window = X.reshape(-1, WINDOW_BINS, len(feats)).mean(axis=1)

    df = meta.copy()
    df["phase"] = labels
    for i, f in enumerate(feats):
        df[f] = per_window[:, i]

    order = df["phase"].value_counts().index.tolist()  # largest first
    color = {p: CATEGORICAL[i] for i, p in enumerate(order)}
    label = {p: f"P{p} {PHASE_NAMES.get(p, '')}" for p in order}
    print("Windows per phase:", df["phase"].value_counts().to_dict())

    # ---- Stage 2 sample counts per workload --------------------------------------------
    threads = df[["workload", "run_id", "tid"]].drop_duplicates()
    s2 = df.groupby("workload").agg(windows=("phase", "size"), runs=("run_id", "nunique")).reindex(WORKLOAD_ORDER)
    s2["thread_runs"] = threads.groupby("workload").size().reindex(WORKLOAD_ORDER)
    s2["share_of_windows"] = (s2["windows"] / s2["windows"].sum()).round(3)
    s2["seconds_of_thread_time"] = (s2["windows"] * STRIDE_SEC).round(0)
    s2.loc["total"] = [s2["windows"].sum(), s2["runs"].sum(), s2["thread_runs"].sum(), 1.0,
                       s2["seconds_of_thread_time"].sum()]
    s2 = s2.astype({"windows": int, "runs": int, "thread_runs": int, "seconds_of_thread_time": int})
    (OUT / "stage2_samples.md").write_text(
        "# Stage 2 samples (after windowing)\n\nA window = 4 consecutive 0.5 s bins (2 s) of one thread; "
        "windows slide by one bin. Each thread contributes its longest gap-free stretch.\n\n"
        + md_table(s2.reset_index().rename(columns={"index": "workload"})) + "\n", encoding="utf-8")

    # ---- Per-phase summary --------------------------------------------------------------
    dwell = dwell_runs(df)
    counts = transition_matrix(df, k)
    trans = counts / counts.sum(axis=1, keepdims=True)
    pd.DataFrame(trans, index=[f"from_{i}" for i in range(k)], columns=[f"to_{i}" for i in range(k)]) \
        .round(4).to_csv(OUT / "transition_matrix.csv")
    share = pd.crosstab(df["phase"], df["workload"], normalize="index").reindex(columns=WORKLOAD_ORDER).fillna(0)
    share.round(4).to_csv(OUT / "phase_workload_share.csv")

    rows = []
    for p in order:
        g = df[df["phase"] == p]
        d = dwell[(dwell["phase"] == p)]
        off = trans[p].copy()
        off[p] = 0
        nxt = np.argsort(off)[::-1][:2]
        row = {
            "phase": p, "name": PHASE_NAMES.get(p, ""), "windows": len(g),
            "share": round(len(g) / len(df), 3), "runs": g["run_id"].nunique(),
            "thread_runs": g[["run_id", "tid"]].drop_duplicates().shape[0],
            "workloads_present": int((share.loc[p] > 0.01).sum()),
            "top_workloads": ", ".join(f"{w} {share.loc[p, w]:.0%}" for w in share.loc[p].sort_values(ascending=False).index[:3]),
            "p_core_share": round((g["core_type"] == "p").mean(), 3),
        }
        for f in feats:
            row[f] = round(g[f].mean(), 3)
        row["windows_p"] = int((g["core_type"] == "p").sum())
        row["windows_e"] = int((g["core_type"] == "e").sum())
        row["dwell_median_sec"] = d["seconds"].median()
        row["dwell_mean_sec"] = round(d["seconds"].mean(), 2)
        row["stay_prob_next_window"] = round(trans[p, p], 3)
        row["most_likely_next_phases"] = ", ".join(f"P{j} ({off[j] / off.sum():.0%} of exits)" for j in nxt)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "phase_summary.csv", index=False)

    cards = [f"# The {k} phases found by Model A", "",
             f"{len(df):,} windows from {df['run_id'].nunique()} runs / "
             f"{df[['run_id', 'tid']].drop_duplicates().shape[0]:,} thread-runs, "
             f"clustered with KMeans (k={K_SELECTED}, silhouette {SILHOUETTE:.3f}, best of k=4..12) "
             "on the 8-d autoencoder embedding.", "",
             "Feature means are over every 0.5 s bin of every window in the phase. Top-down "
             "fractions (retiring / frontend / backend / bad speculation) are shares of pipeline slots.",
             "Dwell = length of an uninterrupted stretch in the phase (stretches cut off by the start "
             "or end of a thread are included, so long-lived phases are if anything under-stated).",
             "",
             "**Features are core-type-aware.** P-cores and E-cores expose different hardware "
             "counters, so each feature is computed from that core type's own counters (P-core top-down "
             "from `TOPDOWN.SLOTS`-family events; memory stalls from `MEMORY_STALLS.*` on P and "
             "`MEM_BOUND_STALLS_LOAD.*` on E). An earlier feature set used one counter set for both, "
             "which made the phases split almost entirely by core type; that version is archived in "
             "`results/archive_pre_core_aware_features_2026-10-21/`.", ""]
    for r in rows:
        cards += [f"## P{r['phase']} - {r['name']}", "",
                  f"- **Size:** {r['windows']:,} windows ({r['share']:.1%}), {r['runs']} runs, "
                  f"{r['thread_runs']:,} thread-runs; {r['workloads_present']} of {len(WORKLOAD_ORDER)} workloads present (>1%)",
                  f"- **Workload mix:** {r['top_workloads']}",
                  f"- **Signature:** IPC {r['ipc']:.2f}, retiring {r['retiring']:.0%}, frontend-bound "
                  f"{r['fe_bound']:.0%}, backend-bound {r['be_bound']:.0%}, bad speculation {r['bad_spec']:.0%}, "
                  f"memory stalls {r['mem_stall_frac']:.0%} of cycles (DRAM {r['dram_stall_frac']:.0%}), "
                  f"branch mispredict {r['branch_mispredict_rate']:.2%}",
                  f"- **Stability:** stays in this phase for the next window {r['stay_prob_next_window']:.0%} "
                  f"of the time; median dwell {r['dwell_median_sec']:.1f} s (mean {r['dwell_mean_sec']:.1f} s)",
                  f"- **When it changes, goes to:** {r['most_likely_next_phases']}",
                  f"- **Core type:** {r['p_core_share']:.0%} of its windows ran on P-cores "
                  f"({r['windows_p']:,} P / {r['windows_e']:,} E)", ""]
    (OUT / "phase_cards.md").write_text("\n".join(cards), encoding="utf-8")

    # ---- Figures ---------------------------------------------------------------------------
    ylabels = [label[p] for p in order]

    # Feature profiles: colour = z-score of the phase mean across phases, text = raw mean.
    prof = summary.set_index("phase").loc[order, feats].to_numpy(dtype=float)
    z = (prof - prof.mean(0)) / prof.std(0)
    div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(order) + 1.6), facecolor=SURFACE)
    ax.imshow(z, cmap=div, vmin=-2.2, vmax=2.2, aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = prof[i, j]
            ax.text(j, i, f"{v:.2f}" if feats[j] == "ipc" else f"{v:.0%}" if v >= 0.01 else f"{v:.2%}",
                    ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(feats)), [FEATURE_LABELS[f] for f in feats], rotation=20, ha="right")
    ax.set_yticks(range(len(order)), ylabels)
    ax.set_title("Phase signatures: mean of each feature (colour: blue = below, orange = above the "
                 "average phase)", loc="left", color=INK, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, "feature_profiles.png")

    # Workload composition heatmap (sequential, one hue).
    seq = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
    m = share.loc[order].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(order) + 1.6), facecolor=SURFACE)
    ax.imshow(m, cmap=seq, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] >= 0.005:
                ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if m[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(WORKLOAD_ORDER)), WORKLOAD_ORDER)
    ax.set_yticks(range(len(order)), ylabels)
    ax.set_title("Which workloads each phase comes from (row = 100%)", loc="left", color=INK, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, "workload_composition.png")

    # Dwell time per phase (box plot of all stretches).
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(order) + 1.6), facecolor=SURFACE)
    data = [dwell[(dwell["phase"] == p)]["seconds"].to_numpy() for p in order]
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, showfliers=False, widths=0.55,
                    medianprops={"color": INK, "linewidth": 1.5}, whiskerprops={"color": INK_MUTED},
                    capprops={"color": INK_MUTED})
    for patch, p in zip(bp["boxes"], order):
        patch.set_facecolor(color[p])
        patch.set_edgecolor(SURFACE)
    ax.set_yticks(range(1, len(order) + 1), ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("seconds in phase before switching (outliers hidden)", color=INK_MUTED)
    _style(ax, "How long a thread stays in each phase")
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, "dwell_time.png")

    # Transitions: off-diagonal only, each row normalised over exits.
    off = trans.copy()
    np.fill_diagonal(off, np.nan)
    off = off / np.nansum(off, axis=1, keepdims=True)
    off = off[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(8.5, 0.7 * len(order) + 1.8), facecolor=SURFACE)
    cmap = ListedColormap(seq(np.linspace(0, 1, 256)))
    cmap.set_bad("#e6e5e1")
    ax.imshow(off, cmap=cmap, vmin=0, vmax=np.nanmax(off), aspect="auto")
    for i in range(len(order)):
        for j in range(len(order)):
            if i == j:
                ax.text(j, i, f"stay\n{trans[order[i], order[i]]:.0%}", ha="center", va="center",
                        fontsize=7, color=INK_MUTED)
            elif off[i, j] >= 0.005:
                ax.text(j, i, f"{off[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if off[i, j] > 0.55 * np.nanmax(off) else INK)
    ax.set_xticks(range(len(order)), [f"P{p}" for p in order])
    ax.set_yticks(range(len(order)), ylabels)
    ax.set_xlabel("next phase", color=INK_MUTED)
    ax.set_title("When a phase ends, where does the thread go? (row = 100% of exits)",
                 loc="left", color=INK, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, "transitions.png")

    # Core type per phase: every phase should contain both core types (Model C compares them).
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(order) + 1.6), facecolor=SURFACE)
    y = np.arange(len(order))
    p_share = summary.set_index("phase").loc[order, "p_core_share"].to_numpy()
    ax.barh(y, p_share, 0.6, color=CORE_COLORS["p"], label=CORE_NAMES["p"])
    ax.barh(y, 1 - p_share, 0.6, left=p_share, color=CORE_COLORS["e"], label=CORE_NAMES["e"],
            edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, p_share):
        ax.text(1.01, yi, f"{v:.0%} P", va="center", fontsize=8, color=INK_MUTED)
    ax.set_yticks(y, ylabels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    _style(ax, "Share of each phase's windows by core type")
    ax.legend(frameon=False, labelcolor=INK, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    _save(fig, "core_type_by_phase.png")

    # Example timelines: the longest thread of each real application on each core type (1-thread runs).
    apps = [w for w in WORKLOAD_ORDER if w in REAL_APPS]
    tracks = [(w, c) for w in apps for c in ("p", "e")]
    fig, axes = plt.subplots(len(tracks), 1, figsize=(10, 0.55 * len(tracks) + 1.6), facecolor=SURFACE, sharex=True)
    for ax, (w, c) in zip(axes, tracks):
        g = df[(df["workload"] == w) & (df["core_type"] == c) & (df["thread_count"] == 1)]
        best = g.groupby(["run_id", "tid"]).size().idxmax()
        seg = g[(g["run_id"] == best[0]) & (g["tid"] == best[1])].sort_values("window_idx")
        t = seg["window_start_sec"].to_numpy()
        ax.bar(t, 1, width=STRIDE_SEC, align="edge", color=[color[p] for p in seg["phase"]], linewidth=0)
        ax.set_yticks([])
        ax.set_ylabel(f"{w} ({CORE_NAMES[c]})", rotation=0, ha="right", va="center", color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_facecolor(SURFACE)
    axes[-1].set_xlabel("time in run (s)", color=INK_MUTED)
    axes[-1].tick_params(colors=INK_MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color[p]) for p in order]
    fig.legend(handles, ylabels, loc="lower center", ncol=min(4, len(order)), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    axes[0].set_title("Phase sequence of the longest thread per application and core type (1-thread runs)",
                      loc="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "example_timelines.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    print(summary[["phase", "name", "windows", "share", "runs", "ipc", "p_core_share",
                   "dwell_median_sec", "stay_prob_next_window"]].to_string(index=False))
    print(f"Outputs in {OUT}")


if __name__ == "__main__":
    main()
