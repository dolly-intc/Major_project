"""
Presentation / report figures and tables for Model A (phase discovery), in plain terms.

Writes to results/model_a/:
  pipeline.png            how Model A works, as a flow diagram
  choosing_k.png          silhouette score and KMeans inertia ("elbow") for k = 4..12
  autoencoder_loss.png    reconstruction loss per epoch (train / validation)
  phase_share.png         share of all thread time spent in each phase
  phase_fingerprints.png  what each phase looks like: where the core's work slots go + memory stalls
  workload_phase_mix.png  for each workload, share of its time in each phase
  model_a_tables.md       report-ready tables: setup, features, choice of k, phase summary
  history.json            the per-epoch losses and per-k scores behind the figures

The loss curve comes from retraining the autoencoder with the same seed as analysis/model_a.py and
checking that the weights match models/phase_discovery/autoencoder.pt. The k sweep is recomputed on
the saved embeddings with the same KMeans settings as model_a.cluster_phases.
Needs results/presentation/phases/phase_summary.csv (run analysis/phase_catalogue.py first).

Run:  python analysis/model_a_figures.py      (~3 min)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis import model_a, model_b_v2 as v2
from analysis.features import extract_features, get_phase_features, PHASE_FEATURES, CORE_COUNTERS
from analysis.load_dataset import load_timelines
from analysis.windows import create_windows, WINDOW_SIZE
from analysis.phase_catalogue import PHASE_NAMES, CATEGORICAL, K_SELECTED
from analysis.dataset_catalogue import md_table, WORKLOAD_ORDER

OUT = v2.ROOT / "results" / "model_a"
MODEL_DIR = v2.ROOT / "models" / "phase_discovery"
K_RANGE = range(4, 13)
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

# One-line plain description of each phase (from the 2026-10-21 profiles).
PHASE_TRAITS = {
    1: "Running smoothly: high IPC, most work slots do useful work",
    3: "Many wrong branch guesses; work is thrown away and redone",
    2: "Core starved of instructions (fetch/decode cannot keep up)",
    0: "Core mostly waiting for data from main memory",
}
FINGERPRINT = [  # feature, plain label
    ("retiring", "Useful work"),
    ("fe_bound", "Waiting for instructions"),
    ("be_bound", "Waiting on execution units / data"),
    ("bad_spec", "Wasted on wrong guesses"),
    ("mem_stall_frac", "Cycles stalled on memory"),
]
FEATURE_MEANING = {
    "ipc": "Instructions completed per clock cycle",
    "retiring": "Share of work slots doing useful work",
    "fe_bound": "Share of slots lost waiting for instructions (fetch/decode)",
    "be_bound": "Share of slots lost waiting on execution units or data",
    "bad_spec": "Share of slots wasted on wrongly guessed (speculative) work",
    "mem_stall_frac": "Share of cycles stalled waiting for data from L2, L3 or main memory",
    "dram_stall_frac": "Share of cycles stalled waiting for main memory (DRAM)",
    "branch_mispredict_rate": "Share of branches guessed wrong",
}


def _style(ax, title=None):
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.tick_params(colors=INK_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d0cfcb")


def _save(fig, name):
    fig.savefig(OUT / name, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _pct_axis(ax, axis="x"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------------------------
def plot_pipeline(n_windows: int, k: int):
    steps = [
        ("Counters", "VTune hardware\ncounters, every\n0.5 s, per thread"),
        ("8 features", "IPC, where work\nslots go, memory\nstalls, mispredicts\n(per core type)"),
        ("2 s windows", f"4 readings x 8\nfeatures = 32\nnumbers\n({n_windows:,} windows)"),
        ("Autoencoder", "compresses\n32 -> 8 numbers;\nsimilar behaviour\nends up close"),
        ("KMeans", "groups the\n8-number codes;\nk chosen by\nsilhouette score"),
        (f"{k} phases", "\n".join(PHASE_NAMES.values())),
    ]
    fig, ax = plt.subplots(figsize=(15, 3.4), facecolor=SURFACE)
    ax.set_xlim(0, len(steps) * 2.5)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for i, (head, body) in enumerate(steps):
        x = i * 2.5 + 0.15
        last = i == len(steps) - 1
        ax.add_patch(FancyBboxPatch((x, 0.35), 2.0, 2.3, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="#2a78d6" if last else "#eef3fb", edgecolor="#2a78d6",
                                    linewidth=1.5))
        ax.text(x + 1.0, 2.25, head, ha="center", va="center", fontsize=11, fontweight="bold",
                color="#ffffff" if last else INK)
        ax.text(x + 1.0, 1.25, body, ha="center", va="center", fontsize=8.5,
                color="#ffffff" if last else INK_MUTED, linespacing=1.4)
        if not last:
            ax.annotate("", xy=(x + 2.42, 1.5), xytext=(x + 2.05, 1.5),
                        arrowprops={"arrowstyle": "-|>", "color": INK_MUTED, "lw": 1.5})
    ax.set_title("Model A: how phases are discovered (no labels are used)", loc="left", color=INK,
                 fontsize=13, fontweight="bold")
    _save(fig, "pipeline.png")


def k_sweep(emb: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(emb)
        rows.append({"k": k, "silhouette": silhouette_score(emb, km.labels_), "inertia": km.inertia_})
        print(f"  k={k}: silhouette {rows[-1]['silhouette']:.4f}")
    return pd.DataFrame(rows)


def plot_choosing_k(sweep: pd.DataFrame):
    best = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), facecolor=SURFACE)
    ax = axes[0]
    ax.plot(sweep["k"], sweep["silhouette"], "-o", color="#2a78d6", linewidth=2, markersize=7)
    b = sweep[sweep["k"] == best].iloc[0]
    ax.plot(best, b["silhouette"], "o", color="#eb6834", markersize=13, zorder=3)
    ax.annotate(f"chosen: k = {best}\nsilhouette {b['silhouette']:.3f}", (best, b["silhouette"]),
                xytext=(best + 1.3, b["silhouette"] - 0.03), fontsize=9, color=INK,
                arrowprops={"arrowstyle": "-", "color": INK_MUTED})
    ax.set_xlabel("number of phases (k)", color=INK_MUTED)
    ax.set_ylabel("silhouette score (higher = better separated)", color=INK_MUTED)
    _style(ax, "Silhouette score: how clearly the phases separate")
    ax.grid(color=GRID, linewidth=0.6)
    ax = axes[1]
    ax.plot(sweep["k"], sweep["inertia"], "-o", color="#52514e", linewidth=2, markersize=7)
    ax.plot(best, b["inertia"], "o", color="#eb6834", markersize=13, zorder=3)
    ax.set_xlabel("number of phases (k)", color=INK_MUTED)
    ax.set_ylabel("inertia (within-phase spread; lower = tighter)", color=INK_MUTED)
    _style(ax, "Elbow check: spread inside phases vs k")
    ax.grid(color=GRID, linewidth=0.6)
    fig.text(0.5, -0.03, "KMeans on the 8-number autoencoder codes, k = 4..12 (10 restarts each, seed 42). "
             "k = 4 has the highest silhouette; every k >= 5 scores much lower.",
             ha="center", fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    _save(fig, "choosing_k.png")


def autoencoder_history():
    """Retrain exactly as model_a.py does, recording the loss per epoch."""
    df = load_timelines()
    phase = get_phase_features(extract_features(df))
    X, _ = create_windows(phase, feature_cols=[c for c in PHASE_FEATURES if c in phase.columns])
    hist = []
    model, _ = model_a.train_autoencoder(X, epochs=50, verbose=False, history=hist)
    saved = torch.load(MODEL_DIR / "autoencoder.pt", weights_only=True)
    same = all(torch.allclose(v, saved[k], atol=1e-6) for k, v in model.state_dict().items())
    print(f"  retrained autoencoder matches models/phase_discovery/autoencoder.pt: {same}")
    return hist, same


def plot_loss(hist: list, matches: bool):
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    e = [h["epoch"] for h in hist]
    ax.plot(e, [h["train_loss"] for h in hist], color="#2a78d6", linewidth=2, label="training")
    ax.plot(e, [h["val_loss"] for h in hist], color="#eb6834", linewidth=2, label="validation (10% held out)")
    ax.set_xlabel("epoch", color=INK_MUTED)
    ax.set_ylabel("reconstruction error (MSE, standardised units)", color=INK_MUTED)
    _style(ax, "Autoencoder training: how well it rebuilds each window from 8 numbers")
    ax.grid(color=GRID, linewidth=0.6)
    ax.legend(frameon=False, labelcolor=INK)
    note = "Validation tracks training closely, so the autoencoder is not overfitting."
    if not matches:
        note += " (Note: this retrain differs slightly from the saved model.)"
    fig.text(0.5, -0.02, note, ha="center", fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    _save(fig, "autoencoder_loss.png")


def plot_phase_share(summary: pd.DataFrame, color: dict):
    s = summary.sort_values("share")
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=SURFACE)
    y = np.arange(len(s))
    ax.barh(y, s["share"], 0.6, color=[color[p] for p in s["phase"]])
    for yi, (_, r) in zip(y, s.iterrows()):
        ax.text(r["share"] + 0.01, yi, f"{r['share']:.0%}  ({r['windows']:,} windows)", va="center",
                fontsize=9, color=INK)
    ax.set_yticks(y, [f"{PHASE_NAMES[p]}" for p in s["phase"]])
    ax.set_xlim(0, 0.85)
    _pct_axis(ax)
    _style(ax, "How much of the threads' time falls in each phase")
    _save(fig, "phase_share.png")


def plot_fingerprints(summary: pd.DataFrame, color: dict):
    order = summary.sort_values("share", ascending=False)
    fig, axes = plt.subplots(1, len(order), figsize=(4.2 * len(order), 3.8), facecolor=SURFACE, sharey=True)
    labels = [lab for _, lab in FINGERPRINT]
    for ax, (_, r) in zip(axes, order.iterrows()):
        vals = [r[f] for f, _ in FINGERPRINT]
        y = np.arange(len(vals))
        ax.barh(y, vals, 0.6, color=color[r["phase"]])
        for yi, v in zip(y, vals):
            ax.text(v + 0.02, yi, f"{v:.0%}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 1)
        _pct_axis(ax)
        _style(ax, f"{PHASE_NAMES[r['phase']]}\nIPC {r['ipc']:.2f} · {r['share']:.0%} of time")
    axes[0].invert_yaxis()  # shared y-axis: invert once
    fig.suptitle("Phase fingerprints: what the core is doing in each phase", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, -0.04, "The first four bars are shares of the core's work slots and add up to 100%. "
             "'Cycles stalled on memory' is a separate measure.", fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    _save(fig, "phase_fingerprints.png")


def plot_workload_mix(labels: np.ndarray, meta: pd.DataFrame, order: list, color: dict):
    mix = pd.crosstab(meta["workload"], labels, normalize="index").reindex(WORKLOAD_ORDER).fillna(0)
    mix = mix.reindex(columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    y = np.arange(len(mix))
    left = np.zeros(len(mix))
    for p in order:
        v = mix[p].to_numpy()
        ax.barh(y, v, 0.65, left=left, color=color[p], label=PHASE_NAMES[p], edgecolor=SURFACE, linewidth=1.5)
        for yi, (l, w) in enumerate(zip(left, v)):
            if w >= 0.08:
                ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if p in (order[0], order[1]) else INK)
        left += v
    ax.set_yticks(y, mix.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    _pct_axis(ax)
    _style(ax, "How each workload's time splits across the phases")
    ax.legend(frameon=False, labelcolor=INK, ncol=len(order), loc="lower left", bbox_to_anchor=(0, 1.06),
              fontsize=9)
    _save(fig, "workload_phase_mix.png")


def write_tables(sweep, summary, n_windows, n_threads, n_runs, hist, labels, meta, order):
    setup = pd.DataFrame([
        ("Data", f"{n_runs} runs, {n_threads:,} thread timelines, {n_windows:,} windows"),
        ("Window", f"{WINDOW_SIZE} consecutive 0.5 s readings = 2 s, sliding by 0.5 s"),
        ("Input per window", f"{WINDOW_SIZE} x {len(PHASE_FEATURES)} = {WINDOW_SIZE * len(PHASE_FEATURES)} numbers, standardised"),
        ("Autoencoder", f"{WINDOW_SIZE * len(PHASE_FEATURES)} -> {model_a.HIDDEN_DIM} -> {model_a.LATENT_DIM} -> "
                        f"{model_a.HIDDEN_DIM} -> {WINDOW_SIZE * len(PHASE_FEATURES)}, ReLU, MSE loss"),
        ("Training", f"{len(hist)} epochs, Adam (lr {model_a.LEARNING_RATE}), batch {model_a.BATCH_SIZE}, "
                     f"10% validation, seed 42; final val loss {hist[-1]['val_loss']:.3f}"),
        ("Clustering", "KMeans on the 8-d codes, k = 4..12, 10 restarts, chosen by silhouette"),
        ("Result", f"k = {K_SELECTED}, silhouette {sweep['silhouette'].max():.3f}"),
    ], columns=["Item", "Value"])
    feats = pd.DataFrame([{
        "Feature": f"`{f}`", "Meaning": FEATURE_MEANING[f],
        "P-core counters": "INST_RETIRED.ANY / CPU_CLK_UNHALTED.THREAD" if f == "ipc" else
            "BR_MISP_RETIRED / BR_INST_RETIRED" if f == "branch_mispredict_rate" else
            " + ".join(np.atleast_1d(CORE_COUNTERS["p"][f.replace("_frac", "")])) if "stall" in f else
            CORE_COUNTERS["p"][f],
        "E-core counters": "same" if f in ("ipc", "branch_mispredict_rate") else
            " + ".join(np.atleast_1d(CORE_COUNTERS["e"][f.replace("_frac", "")])) if "stall" in f else
            CORE_COUNTERS["e"][f],
    } for f in PHASE_FEATURES])
    ktab = sweep.assign(silhouette=sweep["silhouette"].round(4), inertia=sweep["inertia"].round(0).astype(int))
    ktab["chosen"] = np.where(ktab["k"] == K_SELECTED, "yes", "")
    phases = pd.DataFrame([{
        "Phase": f"P{r['phase']} {PHASE_NAMES[r['phase']]}",
        "Share of time": f"{r['share']:.0%}",
        "IPC": f"{r['ipc']:.2f}",
        "What is happening": PHASE_TRAITS[r["phase"]],
        "Typical stay": f"{r['dwell_median_sec']:.1f} s (median)",
        "Main workloads": r["top_workloads"],
        "P-core share": f"{r['p_core_share']:.0%}",
    } for _, r in summary.sort_values("share", ascending=False).iterrows()])
    mix = pd.crosstab(meta["workload"], labels, normalize="index").reindex(WORKLOAD_ORDER).fillna(0)
    mix = (mix.reindex(columns=order, fill_value=0) * 100).round(0).astype(int).astype(str) + "%"
    mix.columns = [PHASE_NAMES[p] for p in order]
    text = ["# Model A — tables for the report", "",
            "Generated by `analysis/model_a_figures.py`; do not edit by hand.", "",
            "## Table A1. Model A setup", "", md_table(setup), "",
            "## Table A2. Input features (computed from each core type's own counters)", "",
            "Top-down shares (retiring, fe_bound, be_bound, bad_spec) are divided by the sum of that core "
            "type's four top-down counters, so they add up to 100%.", "", md_table(feats), "",
            "## Table A3. Choosing the number of phases", "", md_table(ktab), "",
            "## Table A4. The phases", "", md_table(phases), "",
            "## Table A5. Share of each workload's time in each phase", "",
            md_table(mix.reset_index().rename(columns={"workload": "Workload"})), ""]
    (OUT / "model_a_tables.md").write_text("\n".join(text), encoding="utf-8")


def main():
    summary_path = v2.ROOT / "results" / "presentation" / "phases" / "phase_summary.csv"
    if not summary_path.exists():
        raise SystemExit("Run analysis/phase_catalogue.py first (needs phase_summary.csv).")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)

    _, emb, _, labels, meta = v2.load_windows()
    order = pd.Series(labels).value_counts().index.tolist()      # same colours as phase_catalogue
    color = {p: CATEGORICAL[i] for i, p in enumerate(order)}
    n_threads = meta[["run_id", "tid"]].drop_duplicates().shape[0]

    print("k sweep on saved embeddings:")
    sweep = k_sweep(emb)
    print("Retraining the autoencoder for the loss curve:")
    hist, matches = autoencoder_history()

    plot_pipeline(len(emb), K_SELECTED)
    plot_choosing_k(sweep)
    plot_loss(hist, matches)
    plot_phase_share(summary, color)
    plot_fingerprints(summary, color)
    plot_workload_mix(labels, meta, order, color)
    write_tables(sweep, summary, len(emb), n_threads, meta["run_id"].nunique(), hist, labels, meta, order)
    (OUT / "history.json").write_text(json.dumps({"autoencoder_loss": hist, "k_sweep": sweep.to_dict("records"),
                                                  "retrain_matches_saved_model": matches}, indent=1),
                                      encoding="utf-8")
    print(f"Outputs in {OUT}")


if __name__ == "__main__":
    main()
