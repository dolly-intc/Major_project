"""
Presentation figures for Model B: MLP vs LSTM vs baselines, from the model_b_v2 evaluation.

All models below use the same inputs (context features), the same x4 transition-weighted loss,
hysteresis labels and the same shared test anchors, so the comparison is like-for-like.

  model_comparison.png            overall / same-phase / transition recall / transition precision,
                                  mean +- std over the 5 seeds in results/model_b_v2/per_seed.csv
                                  (no training; run analysis/model_b_v2.py first)
  training_curves_comparison.png  per-epoch train/val loss, val accuracy and val transition recall
                                  for the MLP and LSTM-4/6/8, retrained once with DEFAULT_SEED on the
                                  run-level split (~1 min)
  confusion_matrix.png            frozen-model-equivalent MLP (same seed/split as
                                  analysis/freeze_model_b.py) on its test anchors, row-normalised
  curves.json                     the per-epoch numbers behind the training-curve figure

Writes to results/lstm_comparison/.
Run:  python analysis/model_b_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis import model_b_v2 as v2
from analysis.seeding import set_seed, DEFAULT_SEED

OUT = v2.ROOT / "results" / "lstm_comparison"
LABEL_DEF = "hyst"
TARGET = 0.60  # transition target shown in the figures (set by the student 2026-10-24; §9 still says 75%)

# Display name, model_b_v2 experiment name, colour (fixed per model across all figures).
MODELS = [
    ("Markov-2\nbaseline", "markov2", "#8a8985"),
    ("GBDT", "gbdt_ctx_weighted", "#e87ba4"),
    ("LSTM-4", "lstm4_ctx_weighted", "#eb6834"),
    ("LSTM-6", "lstm6_ctx_weighted", "#1baf7a"),
    ("LSTM-8", "lstm8_ctx_weighted", "#eda100"),
    ("MLP\n(chosen)", "mlp_ctx_weighted", "#2a78d6"),
]
CURVE_MODELS = ["mlp_ctx_weighted", "lstm4_ctx_weighted", "lstm6_ctx_weighted", "lstm8_ctx_weighted"]
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11)
    ax.tick_params(colors=INK_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d0cfcb")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_comparison(per_seed: pd.DataFrame):
    d = per_seed[per_seed["label_def"] == LABEL_DEF]
    n_seeds = d["seed"].nunique()
    panels = [
        ("overall_acc", "Overall test accuracy", None),
        ("same_acc", "Same-phase accuracy\n(predicting \"stay in current phase\")", None),
        ("trans_recall", "TRANSITION recall\n(phase changes caught - KEY METRIC)", TARGET),
        ("trans_precision", "Transition precision\n(predicted changes that were right)", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.6), facecolor=SURFACE)
    x = np.arange(len(MODELS))
    for ax, (metric, title, target) in zip(axes, panels):
        stats = [d[d["model"] == m][metric].agg(["mean", "std"]) for _, m, _ in MODELS]
        means = np.array([s["mean"] for s in stats])
        stds = np.array([s["std"] for s in stats])
        ax.bar(x, means, 0.62, yerr=stds, capsize=4, color=[c for _, _, c in MODELS],
               edgecolor=SURFACE, linewidth=2, error_kw={"ecolor": INK_MUTED, "elinewidth": 1})
        for xi, m, s in zip(x, means, stds):
            ax.text(xi, m + s + 0.015, f"{m:.1%}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=INK)
        if target is not None:
            ax.axhline(target, color=INK_MUTED, linestyle="--", linewidth=1.2)
            ax.text(-0.45, target + 0.012, f"target {target:.0%}", ha="left",
                    fontsize=8, color=INK_MUTED)
        ax.set_xticks(x, [n for n, _, _ in MODELS], fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
        _style(ax, title)
    fig.suptitle("Model B - phase lookahead (1 s ahead): MLP vs LSTM vs baselines", fontsize=14,
                 fontweight="bold", color=INK)
    fig.text(0.5, 0.005, f"Same inputs (context features), x4 transition-weighted loss and the same test "
             f"windows for every model; run-level split; bars = mean, whiskers = std over {n_seeds} seeds. "
             "Markov-2 = most frequent next phase given the last phases.",
             ha="center", fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT / "model_comparison.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def train_curves():
    """Retrain the curve models once (DEFAULT_SEED split) recording per-epoch history."""
    X_win, emb, dists, labels, meta = v2.load_windows()
    k = dists.shape[1]
    A = v2.build_anchors(labels, meta)
    feats = v2.build_features(X_win, emb, dists, labels, A, k)
    y, ref = v2.targets_for(LABEL_DEF, labels, A)
    train, val, test = v2.split_runs(A["run_id"], DEFAULT_SEED)
    specs = dict(v2.EXPERIMENTS)
    curves, confusion = {}, None
    for name in CURVE_MODELS:
        spec = specs[name]
        X = feats[f"{spec['kind']}_{spec['feats']}"]
        if spec["kind"] == "lstm":
            X = np.ascontiguousarray(X[:, -spec["history"]:])
        set_seed(DEFAULT_SEED)
        hist = []
        model, Xt, _ = v2.train_torch(spec, X, y, ref, train, val, k, v2.EPOCHS, history=hist)
        best = int(np.argmin([h["val_loss"] for h in hist])) + 1
        curves[name] = {"history": hist, "best_epoch": best}
        print(f"{name}: {len(hist)} epochs, best (early-stopping) epoch {best}")
        if name == "mlp_ctx_weighted":
            with torch.no_grad():
                pred = model(Xt[torch.from_numpy(test)])[0].argmax(1).numpy()
            sc = v2.score(pred, y[test], ref[test])
            print(f"  MLP test: transition recall {sc['trans_recall']:.3f}, overall {sc['overall_acc']:.3f} "
                  "(should match models/phase_lookahead_v2/metrics.json)")
            confusion = pd.crosstab(pd.Series(y[test], name="true"), pd.Series(pred, name="pred"))
            confusion = confusion.reindex(index=range(k), columns=range(k), fill_value=0)
    return curves, confusion


def plot_curves(curves: dict):
    colour = {m: c for _, m, c in MODELS}
    label = {m: n.replace("\n", " ") for n, m, _ in MODELS}
    panels = [("train_loss", "Training loss", "loss (x4 weighted cross-entropy)"),
              ("val_loss", "Validation loss", "loss (x4 weighted cross-entropy)"),
              ("val_acc", "Validation accuracy (overall)", "accuracy"),
              ("val_trans_recall", "Validation TRANSITION recall", "transition recall")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=SURFACE)
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        for m in CURVE_MODELS:
            h = curves[m]["history"]
            ax.plot([r["epoch"] for r in h], [r[key] for r in h], color=colour[m], linewidth=2,
                    label=label[m])
            best = curves[m]["best_epoch"]
            ax.plot(best, h[best - 1][key], "o", color=colour[m], markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2)
        if key == "val_trans_recall":
            ax.axhline(TARGET, color=INK_MUTED, linestyle="--", linewidth=1.2)
            ax.text(ax.get_xlim()[1], TARGET + 0.01, f"target {TARGET:.0%}", ha="right", fontsize=8,
                    color=INK_MUTED)
        if key in ("val_acc", "val_trans_recall"):
            ax.set_yticks(ax.get_yticks(), [f"{v:.0%}" for v in ax.get_yticks()])
        ax.set_xlabel("epoch", color=INK_MUTED)
        ax.set_ylabel(ylabel, color=INK_MUTED)
        _style(ax, title)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.suptitle("Training curves: MLP vs LSTM-4/6/8 (same inputs and loss)", fontsize=14,
                 fontweight="bold", color=INK)
    fig.text(0.5, 0.005, f"One training run per model (seed {DEFAULT_SEED}, run-level split). Dot = epoch "
             "with the lowest validation loss, whose weights are kept (early stopping, patience "
             f"{v2.PATIENCE}).", ha="center", fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(OUT / "training_curves_comparison.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(confusion: pd.DataFrame):
    from analysis.phase_catalogue import PHASE_NAMES
    names = [f"P{p} {PHASE_NAMES.get(p, '')}" for p in confusion.index]
    m = confusion.to_numpy(dtype=float)
    norm = m / m.sum(axis=1, keepdims=True).clip(min=1)
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(8, 6.4), facecolor=SURFACE)
    ax.imshow(norm, cmap=seq, vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({int(m[i, j]):,})", ha="center", va="center", fontsize=9,
                    color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted phase 1 s ahead", color=INK_MUTED)
    ax.set_ylabel("actual phase 1 s ahead", color=INK_MUTED)
    ax.set_title(f"Chosen MLP: confusion matrix on test windows (seed {DEFAULT_SEED}; row = 100%)",
                 loc="left", color=INK, fontsize=11)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    per_seed_path = v2.OUT_DIR / "per_seed.csv"
    if not per_seed_path.exists():
        raise SystemExit("Run analysis/model_b_v2.py first (needs results/model_b_v2/per_seed.csv).")
    OUT.mkdir(parents=True, exist_ok=True)
    plot_comparison(pd.read_csv(per_seed_path))
    curves, confusion = train_curves()
    (OUT / "curves.json").write_text(json.dumps(curves, indent=1), encoding="utf-8")
    plot_curves(curves)
    plot_confusion(confusion)
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
