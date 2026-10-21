"""
Model B v2: honest evaluation + transition-focused improvements for phase lookahead.

Fixes three evaluation problems in model_b.py / train_lstm_compare.py:
  1. Leakage. Windows were split at random, so neighbouring windows (which share
     3 of their 4 bins) landed on both sides of the split. Here train/val/test
     are split by run_id, so no run contributes windows to more than one side.
  2. Single-seed numbers. Every row is mean +- std over several seeds; each seed
     re-draws the run split AND the weight init / shuffling.
  3. Different test sets per variant. Every model is scored on one shared set of
     anchor windows: each anchor has ANCHOR_HISTORY windows of past (enough for
     the longest-history model, LSTM-8) and enough future for both label definitions.

Then compares, as ablation rows: baselines (persistence, Markov), the original
MLP / LSTM-6 setup, added context features, transition-weighted loss, a two-head
change/destination model with a validation-tuned threshold, and a GBDT.

Two label definitions are reported side by side:
  raw   - nearest-centroid phase, exactly as in model_b.py. Reference ("current")
          phase = the anchor window's phase.
  hyst  - phase changes shorter than MIN_DURATION windows are noise. Target =
          smoothed phase (short runs merged into the preceding run). Reference =
          the *committed* phase at the anchor: the last phase the thread has held
          for >= MIN_DURATION windows. The committed phase is causal, i.e. a
          scheduler could compute it live.

Transition metrics (a "transition" is target != reference):
  trans_recall    - fraction of true transitions predicted exactly right
                    (this is the "transition accuracy" model_b.py reports)
  trans_precision - of the predictions that say "phase changes", fraction right
  trans_f1        - harmonic mean of the two
  change_recall   - fraction of true transitions where the model predicted *a*
                    change, whatever the destination

Writes only to results/model_b_v2/. Never touches models/.

Run:  python analysis/model_b_v2.py                  (5 seeds, full)
      python analysis/model_b_v2.py --quick          (1 seed, 20 epochs; smoke test)
      python analysis/model_b_v2.py --seeds 0 1 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.load_dataset import load_timelines
from analysis.features import extract_features, get_phase_features, PHASE_FEATURES
from analysis.windows import create_windows
from analysis.model_a import Autoencoder, get_embeddings
from analysis.seeding import set_seed


# Task
HORIZON = 2          # windows ahead = 1 s (same as model_b.py)
HISTORY = 6          # run-length feature cap (windows); also the v1-best LSTM history
LSTM_HISTORIES = [4, 6, 8]
ANCHOR_HISTORY = max(LSTM_HISTORIES)  # every model is scored on anchors with this much past
MIN_DURATION = 3     # hysteresis: a phase must last >= 3 windows (1.5 s) to count

# Training
HIDDEN = 32
DROPOUT = 0.2
BATCH = 256
EPOCHS = 100
PATIENCE = 10
LR = 1e-3
TRANS_WEIGHT = 4.0   # loss weight on transition samples for "weighted" rows
TEST_FRAC = 0.15
VAL_FRAC = 0.15
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
THRESHOLDS = np.round(np.linspace(0.05, 0.95, 19), 2)

LABEL_DEFS = ["raw", "hyst"]
ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "results" / "model_b_v2"

# (name, spec). "baseline" rows have no training; the rest are learned.
EXPERIMENTS = [
    ("persistence",        {"kind": "baseline"}),
    ("markov1",            {"kind": "baseline"}),
    ("markov2",            {"kind": "baseline"}),
    ("mlp_emb_ce",         {"kind": "mlp",  "feats": "emb", "mode": "ce"}),        # = model_b.py MLP
    *[(f"lstm{h}_emb_ce",  {"kind": "lstm", "feats": "emb", "mode": "ce", "history": h})
      for h in LSTM_HISTORIES],                                                    # = v1 LSTM-4/6/8
    ("mlp_ctx_ce",         {"kind": "mlp",  "feats": "ctx", "mode": "ce"}),
    ("mlp_ctx_weighted",   {"kind": "mlp",  "feats": "ctx", "mode": "weighted"}),  # frozen choice
    *[(f"lstm{h}_ctx_weighted", {"kind": "lstm", "feats": "ctx", "mode": "weighted", "history": h})
      for h in LSTM_HISTORIES],
    ("mlp_ctx_twohead",    {"kind": "mlp",  "feats": "ctx", "mode": "twohead"}),
    ("lstm6_ctx_twohead",  {"kind": "lstm", "feats": "ctx", "mode": "twohead", "history": 6}),
    ("gbdt_ctx_weighted",  {"kind": "gbdt", "feats": "ctx", "mode": "weighted"}),
]

EXPERIMENT_NOTES = {
    "persistence": "always predict the current phase",
    "markov1": "most frequent next phase given current phase (train counts)",
    "markov2": "same, given current phase + last two raw phases",
    "mlp_emb_ce": "v1 MLP: anchor embedding only, plain cross-entropy",
    **{f"lstm{h}_emb_ce": f"v1 LSTM-{h}: {h} embeddings ({h / 2:g} s), plain cross-entropy"
       for h in LSTM_HISTORIES},
    "mlp_ctx_ce": "+ centroid distances, margin, current/committed phase, deltas, raw window",
    "mlp_ctx_weighted": "+ transition samples weighted x4 in the loss",
    **{f"lstm{h}_ctx_weighted": f"LSTM over {h} steps of context features, x4 weights"
       for h in LSTM_HISTORIES},
    "mlp_ctx_twohead": "change? head + destination head, threshold tuned on val",
    "lstm6_ctx_twohead": "two-head on 6 steps of context features",
    "gbdt_ctx_weighted": "gradient-boosted trees on context features, x4 weights",
}

# How hard the model is to re-implement in the stdlib-only placement service (service/).
PORTABILITY = {"baseline": "trivial (lookup)", "mlp": "easy (3 matrix multiplies)",
               "lstm": "moderate (hand-written gates, state per thread)",
               "gbdt": "hard (export + walk 200 x 8 trees)"}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_windows():
    """Recompute windows, embeddings and phase labels from the frozen Stage 2 artifacts."""
    model_dir = ROOT / "models" / "phase_discovery"

    scaler_data = np.load(model_dir / "scaler.npz")
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    centroids = np.load(model_dir / "kmeans_centroids.npy")

    ae_model = Autoencoder(len(scaler.mean_), hidden_dim=16, latent_dim=8)
    ae_model.load_state_dict(torch.load(model_dir / "autoencoder.pt", weights_only=True))
    ae_model.eval()

    print("Loading timeline data...")
    df = load_timelines()
    df_phase = get_phase_features(extract_features(df))
    feature_cols = [c for c in PHASE_FEATURES if c in df_phase.columns]
    X_windows, meta = create_windows(df_phase, feature_cols=feature_cols)

    print("Computing embeddings...")
    embeddings = get_embeddings(X_windows, ae_model, scaler)
    distances = cdist(embeddings, centroids, metric="euclidean")
    labels = distances.argmin(axis=1)

    return scaler.transform(X_windows), embeddings, distances, labels, meta.reset_index(drop=True)


def committed_labels(raw: np.ndarray, min_dur: int) -> np.ndarray:
    """Causal: the phase the thread has held for >= min_dur windows, as of each window."""
    committed = np.empty_like(raw)
    committed[0] = raw[0]
    run = 1
    for i in range(1, len(raw)):
        run = run + 1 if raw[i] == raw[i - 1] else 1
        committed[i] = raw[i] if run >= min_dur else committed[i - 1]
    return committed


def smoothed_labels(raw: np.ndarray, min_dur: int) -> np.ndarray:
    """Non-causal ground truth: runs shorter than min_dur (except the first) take the previous phase."""
    smoothed = raw.copy()
    change = np.flatnonzero(np.diff(raw)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(raw)]])
    for s, e in zip(starts[1:], ends[1:]):
        if e - s < min_dur:
            smoothed[s:e] = smoothed[s - 1]
    return smoothed


def run_lengths(raw: np.ndarray):
    """(causal run length so far, full length of the run containing each window)."""
    causal = np.ones(len(raw), dtype=int)
    for i in range(1, len(raw)):
        if raw[i] == raw[i - 1]:
            causal[i] = causal[i - 1] + 1
    full = np.empty(len(raw), dtype=int)
    change = np.flatnonzero(np.diff(raw)) + 1
    for s, e in zip(np.concatenate([[0], change]), np.concatenate([change, [len(raw)]])):
        full[s:e] = e - s
    return causal, full


def build_anchors(labels: np.ndarray, meta: pd.DataFrame) -> dict:
    """
    Per-window label tracks plus the shared anchor set.

    An anchor t is kept only if it has ANCHOR_HISTORY windows of past and its target
    t + HORIZON is at least MIN_DURATION - 1 windows from the end of the thread,
    so the smoothed target is never decided by a run cut off by the sequence end.
    """
    n = len(meta)
    committed = np.empty(n, dtype=int)
    smoothed = np.empty(n, dtype=int)
    causal_run = np.empty(n, dtype=int)
    full_run = np.empty(n, dtype=int)

    anchor, target, hist, run_ids, workloads = [], [], [], [], []
    threads_total = threads_kept = 0

    for (run_id, _tid), g in meta.groupby(["run_id", "tid"], sort=False):
        g = g.sort_values("window_idx")
        idx = g.index.to_numpy()
        if not np.all(np.diff(g["window_idx"].to_numpy()) == 1):
            raise ValueError(f"non-consecutive windows in {run_id}")
        raw = labels[idx]
        committed[idx] = committed_labels(raw, MIN_DURATION)
        smoothed[idx] = smoothed_labels(raw, MIN_DURATION)
        causal_run[idx], full_run[idx] = run_lengths(raw)

        threads_total += 1
        last = len(idx) - 1 - HORIZON - (MIN_DURATION - 1)
        ts = np.arange(ANCHOR_HISTORY - 1, last + 1)
        if len(ts) == 0:
            continue
        threads_kept += 1
        anchor.append(idx[ts])
        target.append(idx[ts + HORIZON])
        hist.append(idx[ts[:, None] + np.arange(-ANCHOR_HISTORY + 1, 1)])
        run_ids.extend([run_id] * len(ts))
        workloads.extend([g["workload"].iloc[0]] * len(ts))

    return {
        "committed": committed,
        "smoothed": smoothed,
        "causal_run": causal_run,
        "full_run": full_run,
        "anchor": np.concatenate(anchor),
        "target": np.concatenate(target),
        "hist": np.concatenate(hist),
        "run_id": np.array(run_ids),
        "workload": np.array(workloads),
        "threads_total": threads_total,
        "threads_kept": threads_kept,
    }


def targets_for(label_def: str, labels: np.ndarray, A: dict):
    """(y, ref): target phase and the reference ('current') phase for each anchor."""
    if label_def == "raw":
        return labels[A["target"]], labels[A["anchor"]]
    return A["smoothed"][A["target"]], A["committed"][A["anchor"]]


def onehot(x: np.ndarray, k: int) -> np.ndarray:
    return np.eye(k, dtype=np.float32)[x]


def build_features(X_win, emb, dists, labels, A, k) -> dict:
    """All feature sets, indexed by anchor. Everything here is causal."""
    sorted_d = np.sort(dists, axis=1)
    margin = (sorted_d[:, 1] - sorted_d[:, 0])[:, None]
    # Per-window step features: what an online scheduler knows about one window.
    step = np.hstack([
        emb, dists, margin,
        onehot(labels, k), onehot(A["committed"], k),
        np.minimum(A["causal_run"], HISTORY)[:, None] / HISTORY,
    ]).astype(np.float32)

    a, h = A["anchor"], A["hist"]
    ctx = np.hstack([
        step[a],
        emb[a] - emb[h[:, -2]],
        emb[a] - emb[h[:, -3]],
        onehot(labels[h[:, -2]], k),
        onehot(labels[h[:, -3]], k),
        X_win[a],
    ]).astype(np.float32)

    return {
        "mlp_emb": emb[a].astype(np.float32),
        "mlp_ctx": ctx,
        "lstm_emb": emb[h].astype(np.float32),   # (N, ANCHOR_HISTORY, d); sliced per model
        "lstm_ctx": step[h],
        "gbdt_ctx": ctx,
    }


def split_runs(run_id: np.ndarray, seed: int):
    """Train/val/test anchor masks, split by run so no run straddles two sides."""
    idx = np.arange(len(run_id))
    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    rest, test = next(outer.split(idx, groups=run_id))
    inner = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC / (1 - TEST_FRAC), random_state=seed)
    tr, va = next(inner.split(rest, groups=run_id[rest]))
    train, val = rest[tr], rest[va]
    assert not (set(run_id[train]) & set(run_id[test]))
    assert not (set(run_id[val]) & set(run_id[test]))
    return train, val, test


# ----------------------------------------------------------------------------
# Metrics and baselines
# ----------------------------------------------------------------------------

def score(pred: np.ndarray, y: np.ndarray, ref: np.ndarray) -> dict:
    trans = y != ref
    pred_change = pred != ref
    hit = pred == y
    recall = hit[trans].mean() if trans.any() else 0.0
    precision = (hit & pred_change).sum() / pred_change.sum() if pred_change.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "overall_acc": hit.mean(),
        "same_acc": hit[~trans].mean() if (~trans).any() else 0.0,
        "trans_recall": recall,
        "trans_precision": precision,
        "trans_f1": f1,
        "change_recall": pred_change[trans].mean() if trans.any() else 0.0,
        "n_test": len(y),
        "n_trans": int(trans.sum()),
    }


def markov_predict(keys_train, y_train, keys_test, fallback):
    table = (pd.DataFrame({"k": keys_train, "y": y_train})
             .groupby("k")["y"].agg(lambda s: s.value_counts().idxmax()))
    lookup = table.to_dict()
    return np.array([lookup.get(k, f) for k, f in zip(keys_test, fallback)])


def run_baseline(name, y, ref, labels, A, train, test):
    if name == "persistence":
        return ref[test]
    prev1, prev2 = labels[A["hist"][:, -1]], labels[A["hist"][:, -2]]
    if name == "markov1":
        keys = ref.astype(str)
    else:
        keys = np.char.add(np.char.add(ref.astype(str), "_"),
                           np.char.add(np.char.add(prev2.astype(str), "_"), prev1.astype(str)))
    return markov_predict(keys[train], y[train], keys[test], ref[test])


# ----------------------------------------------------------------------------
# Neural models
# ----------------------------------------------------------------------------

class LookaheadNet(nn.Module):
    """MLP or LSTM encoder with a destination head and an optional change head."""

    def __init__(self, kind: str, in_dim: int, n_classes: int, two_head: bool):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.enc = nn.Sequential(
                nn.Linear(in_dim, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            )
        else:
            self.enc = nn.LSTM(in_dim, HIDDEN, num_layers=2, batch_first=True, dropout=DROPOUT)
        self.dest = nn.Linear(HIDDEN, n_classes)
        self.change = nn.Linear(HIDDEN, 1) if two_head else None

    def forward(self, x):
        if self.kind == "mlp":
            h = self.enc(x)
        else:
            _, (h_n, _) = self.enc(x)
            h = h_n[-1]
        return self.dest(h), (self.change(h).squeeze(1) if self.change is not None else None)


def compute_loss(model, x, y, ref, mode, n_classes):
    dest, change = model(x)
    trans = y != ref
    if mode == "ce":
        return F.cross_entropy(dest, y)
    if mode == "weighted":
        w = 1.0 + (TRANS_WEIGHT - 1.0) * trans.float()
        return (F.cross_entropy(dest, y, reduction="none") * w).sum() / w.sum()
    # two-head: P(change) + P(destination | change), destination excludes the current phase
    loss = F.binary_cross_entropy_with_logits(change, trans.float())
    if trans.any():
        masked = dest[trans].masked_fill(F.one_hot(ref[trans], n_classes).bool(), -1e9)
        loss = loss + F.cross_entropy(masked, y[trans])
    return loss


def train_torch(spec, X, y, ref, train, val, n_classes, epochs):
    mode = spec["mode"]
    # Standardise on train statistics only (flattened over time steps for the LSTM).
    feat_dim = X.shape[-1]
    sc = StandardScaler().fit(X[train].reshape(-1, feat_dim))
    Xs = sc.transform(X.reshape(-1, feat_dim)).reshape(X.shape).astype(np.float32)

    Xt, yt, rt = torch.from_numpy(Xs), torch.from_numpy(y), torch.from_numpy(ref)
    model = LookaheadNet(spec["kind"], feat_dim, n_classes, mode == "twohead")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tr_idx = torch.from_numpy(train)
    va_idx = torch.from_numpy(val)
    best_loss, best_state, bad = float("inf"), None, 0
    for _epoch in range(epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        for b in range(0, len(perm), BATCH):
            bi = perm[b:b + BATCH]
            opt.zero_grad()
            compute_loss(model, Xt[bi], yt[bi], rt[bi], mode, n_classes).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = compute_loss(model, Xt[va_idx], yt[va_idx], rt[va_idx], mode, n_classes).item()
        if val_loss < best_loss:
            best_loss, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, Xt, sc


def torch_outputs(model, Xt, idx, ref, n_classes):
    """(argmax prediction, P(change), best non-current destination) for the given anchors."""
    with torch.no_grad():
        dest, change = model(Xt[torch.from_numpy(idx)])
    argmax = dest.argmax(1).numpy()
    if change is None:
        return argmax, None, None
    masked = dest.masked_fill(F.one_hot(torch.from_numpy(ref[idx]), n_classes).bool(), -1e9)
    return argmax, torch.sigmoid(change).numpy(), masked.argmax(1).numpy()


def twohead_decide(p_change, alt, ref, thr):
    return np.where(p_change >= thr, alt, ref)


def tune_threshold(p_change, alt, y, ref):
    """Threshold on P(change) that maximises transition F1 on the validation set."""
    f1s = [score(twohead_decide(p_change, alt, ref, t), y, ref)["trans_f1"] for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(f1s))])


# ----------------------------------------------------------------------------
# Diagnostics, outputs
# ----------------------------------------------------------------------------

def diagnostics(labels, A, n_classes) -> dict:
    cur, fut = labels[A["anchor"]], labels[A["target"]]
    raw_trans = cur != fut
    matrix = pd.crosstab(pd.Series(cur, name="current"), pd.Series(fut, name="future"))
    matrix = matrix.reindex(index=range(n_classes), columns=range(n_classes), fill_value=0)
    matrix.to_csv(OUT_DIR / "transition_matrix_raw.csv")

    pairs = (pd.DataFrame({"from": cur[raw_trans], "to": fut[raw_trans]})
             .value_counts().head(10))
    top_pairs = [{"from": int(f), "to": int(t), "count": int(c),
                  "share_of_transitions": float(c / raw_trans.sum())}
                 for (f, t), c in pairs.items()]

    # A raw transition is "flicker" if the phase it lands in lasts < MIN_DURATION windows.
    flicker = A["full_run"][A["target"]][raw_trans] < MIN_DURATION
    y_h, ref_h = targets_for("hyst", labels, A)
    hyst_trans = y_h != ref_h

    return {
        "n_anchors": int(len(cur)),
        "threads_kept": A["threads_kept"],
        "threads_total": A["threads_total"],
        "n_runs": int(len(np.unique(A["run_id"]))),
        "anchors_per_workload": pd.Series(A["workload"]).value_counts().to_dict(),
        "raw_transition_rate": float(raw_trans.mean()),
        "hyst_transition_rate": float(hyst_trans.mean()),
        "raw_transitions_that_are_flicker": float(flicker.mean()),
        "top_raw_transition_pairs": top_pairs,
    }


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = ["overall_acc", "same_acc", "trans_recall", "trans_precision",
               "trans_f1", "change_recall", "n_trans"]
    metrics += [c for c in ("n_params", "train_sec") if c in per_seed.columns]
    order = {name: i for i, (name, _) in enumerate(EXPERIMENTS)}
    agg = per_seed.groupby(["label_def", "model"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["_order"] = agg["model"].map(order)
    agg["_def"] = agg["label_def"].map(LABEL_DEFS.index)
    return agg.sort_values(["_def", "_order"]).drop(columns=["_def", "_order"])


def write_markdown(summary: pd.DataFrame, diag: dict, seeds, out: Path):
    def pm(row, m):
        return f"{100 * row[f'{m}_mean']:.1f} ± {100 * row[f'{m}_std']:.1f}"

    lines = [
        "# Model B v2 — results",
        "",
        f"Seeds: {list(seeds)} · split by run_id (test {TEST_FRAC:.0%}, val {VAL_FRAC:.0%}) · "
        f"horizon {HORIZON} windows · anchors need {ANCHOR_HISTORY} windows of past · hysteresis MIN_DURATION={MIN_DURATION}",
        f"Shared test anchors: every model is scored on the same anchors "
        f"({diag['n_anchors']} anchors from {diag['threads_kept']}/{diag['threads_total']} threads, "
        f"{diag['n_runs']} runs).",
        "",
        f"Transition rate: raw {100 * diag['raw_transition_rate']:.1f}%, "
        f"hyst {100 * diag['hyst_transition_rate']:.1f}%. "
        f"Raw transitions that land in a phase lasting < {MIN_DURATION} windows (flicker): "
        f"{100 * diag['raw_transitions_that_are_flicker']:.1f}%.",
        "",
        "All numbers are % (mean ± std over seeds). `trans_recall` is the 'transition accuracy' "
        "that model_b.py reports.",
        "",
    ]
    for label_def in LABEL_DEFS:
        lines += [f"## Label definition: `{label_def}`", "",
                  "| model | what it adds | overall | same-phase | trans recall | trans precision | trans F1 | change detected | n trans (test) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in summary[summary["label_def"] == label_def].iterrows():
            lines.append(
                f"| {r['model']} | {EXPERIMENT_NOTES[r['model']]} | {pm(r, 'overall_acc')} | "
                f"{pm(r, 'same_acc')} | **{pm(r, 'trans_recall')}** | {pm(r, 'trans_precision')} | "
                f"{pm(r, 'trans_f1')} | {pm(r, 'change_recall')} | {r['n_trans_mean']:.0f} |")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def write_comparison(summary: pd.DataFrame, out: Path):
    """Presentation table: every model on the same anchors, with cost and portability."""
    kinds = {name: spec["kind"] for name, spec in EXPERIMENTS}
    lines = [
        "# Model B — model comparison (presentation table)",
        "",
        "Same shared test anchors for every row; run-level split; mean ± std over 5 seeds. "
        "Labels: hysteresis (phase changes shorter than 1.5 s treated as noise). "
        "Transition = the phase 1 s ahead differs from the current committed phase.",
        "",
        "| model | what it is | trans recall | trans precision | trans F1 | overall | params | train time | service port |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary[summary["label_def"] == "hyst"].iterrows():
        def pm(m):
            return f"{100 * r[f'{m}_mean']:.1f} ± {100 * r[f'{m}_std']:.1f}"
        params = f"{r['n_params_mean']:,.0f}" if pd.notna(r.get("n_params_mean")) else "—"
        if kinds[r["model"]] == "gbdt" and params != "—":
            params += " leaves"
        secs = f"{r['train_sec_mean']:.0f} s" if pd.notna(r.get("train_sec_mean")) else "—"
        name = f"**{r['model']}** (chosen)" if r["model"] == "mlp_ctx_weighted" else r["model"]
        lines.append(f"| {name} | {EXPERIMENT_NOTES[r['model']]} | **{pm('trans_recall')}** | "
                     f"{pm('trans_precision')} | {pm('trans_f1')} | {pm('overall_acc')} | {params} | "
                     f"{secs} | {PORTABILITY[kinds[r['model']]]} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Validated reference palette (dataviz skill), slots 1-2, light surface.
COLOR_RAW, COLOR_HYST = "#2a78d6", "#eb6834"
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_recall(summary: pd.DataFrame, out: Path):
    models = [name for name, _ in EXPERIMENTS]
    y = np.arange(len(models))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(models) + 1.5), facecolor=SURFACE)
    _style(ax)
    for offset, label_def, color in ((-h / 2, "raw", COLOR_RAW), (h / 2, "hyst", COLOR_HYST)):
        s = summary[summary["label_def"] == label_def].set_index("model").reindex(models)
        mean, std = 100 * s["trans_recall_mean"].to_numpy(), 100 * s["trans_recall_std"].to_numpy()
        ax.barh(y + offset, mean, height=h - 0.04, xerr=std, color=color, label=label_def,
                error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 2})
        for yi, m, sd in zip(y + offset, mean, std):
            ax.text(m + sd + 0.8, yi, f"{m:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(models, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Transition recall, % (mean ± std over seeds)", color=INK)
    ax.set_title("Model B v2: transition recall by model and label definition",
                 color=INK, loc="left", fontsize=12)
    ax.legend(title="label definition", frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_threshold_sweep(sweep: pd.DataFrame, out: Path):
    if sweep.empty:
        return
    groups = sweep.groupby(["label_def", "model", "threshold"])[["trans_recall", "trans_precision"]].mean().reset_index()
    panels = list(groups.groupby(["label_def", "model"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, ((label_def, model), g) in zip(axes, panels):
        _style(ax)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.plot(g["threshold"], 100 * g["trans_recall"], color=COLOR_RAW, linewidth=2, label="recall")
        ax.plot(g["threshold"], 100 * g["trans_precision"], color=COLOR_HYST, linewidth=2, label="precision")
        ax.set_title(f"{model} · {label_def}", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("threshold on P(change)", color=INK_MUTED, fontsize=8)
    axes[0].set_ylabel("% on test (mean over seeds)", color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Two-head models: recall/precision trade-off (chosen threshold is tuned on val)",
                 color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--quick", action="store_true", help="1 seed, 20 epochs (smoke test)")
    args = parser.parse_args()
    seeds = [args.seeds[0]] if args.quick else args.seeds
    epochs = 20 if args.quick else args.epochs

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_win, emb, dists, labels, meta = load_windows()
    n_classes = dists.shape[1]

    A = build_anchors(labels, meta)
    feats = build_features(X_win, emb, dists, labels, A, n_classes)
    diag = diagnostics(labels, A, n_classes)
    (OUT_DIR / "diagnostics.json").write_text(json.dumps(diag, indent=2, default=int), encoding="utf-8")

    print(f"\n{diag['n_anchors']} shared anchors from {diag['threads_kept']}/{diag['threads_total']} threads")
    print(f"Transition rate: raw {diag['raw_transition_rate']:.1%}, hyst {diag['hyst_transition_rate']:.1%}")
    print(f"Raw transitions that are flicker (< {MIN_DURATION} windows): "
          f"{diag['raw_transitions_that_are_flicker']:.1%}")
    print("Top raw transition pairs:")
    for p in diag["top_raw_transition_pairs"][:5]:
        print(f"  {p['from']} -> {p['to']}: {p['count']} ({p['share_of_transitions']:.1%})")

    rows, sweep_rows = [], []
    t0 = time.time()
    for seed in seeds:
        train, val, test = split_runs(A["run_id"], seed)
        print(f"\n=== seed {seed}: runs train/val/test = "
              f"{len(np.unique(A['run_id'][train]))}/{len(np.unique(A['run_id'][val]))}/"
              f"{len(np.unique(A['run_id'][test]))}, anchors {len(train)}/{len(val)}/{len(test)} ===")
        for label_def in LABEL_DEFS:
            y, ref = targets_for(label_def, labels, A)
            for name, spec in EXPERIMENTS:
                set_seed(seed)
                extra = {}
                if spec["kind"] == "baseline":
                    pred = run_baseline(name, y, ref, labels, A, train, test)
                elif spec["kind"] == "gbdt":
                    X = feats["gbdt_ctx"]
                    w = np.where(y[train] != ref[train], TRANS_WEIGHT, 1.0)
                    t_fit = time.time()
                    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                         early_stopping=False, random_state=seed)
                    clf.fit(X[train], y[train], sample_weight=w)
                    extra["train_sec"] = time.time() - t_fit
                    extra["n_params"] = sum(est.get_n_leaf_nodes() for row in clf._predictors for est in row)
                    pred = clf.predict(X[test])
                else:
                    X = feats[f"{spec['kind']}_{spec['feats']}"]
                    if spec["kind"] == "lstm":
                        X = np.ascontiguousarray(X[:, -spec["history"]:])
                    t_fit = time.time()
                    model, Xt, _ = train_torch(spec, X, y, ref, train, val, n_classes, epochs)
                    extra["train_sec"] = time.time() - t_fit
                    extra["n_params"] = sum(p.numel() for p in model.parameters())
                    pred, p_te, alt_te = torch_outputs(model, Xt, test, ref, n_classes)
                    if spec["mode"] == "twohead":
                        _, p_va, alt_va = torch_outputs(model, Xt, val, ref, n_classes)
                        thr = tune_threshold(p_va, alt_va, y[val], ref[val])
                        pred = twohead_decide(p_te, alt_te, ref[test], thr)
                        extra["threshold"] = thr
                        for t in THRESHOLDS:
                            s = score(twohead_decide(p_te, alt_te, ref[test], t), y[test], ref[test])
                            sweep_rows.append({"seed": seed, "label_def": label_def, "model": name,
                                               "threshold": t, **s})
                s = score(pred, y[test], ref[test])
                rows.append({"seed": seed, "label_def": label_def, "model": name, **s, **extra})
                print(f"  [{label_def:4}] {name:20} overall {s['overall_acc']:.3f}  "
                      f"trans recall {s['trans_recall']:.3f}  precision {s['trans_precision']:.3f}  "
                      f"F1 {s['trans_f1']:.3f}" + (f"  thr {extra['threshold']}" if "threshold" in extra else ""))
        print(f"  elapsed {time.time() - t0:.0f}s")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT_DIR / "per_seed.csv", index=False)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT_DIR / "threshold_sweep.csv", index=False)
    summary = summarise(per_seed)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    write_markdown(summary, diag, seeds, OUT_DIR / "summary.md")
    write_comparison(summary, OUT_DIR / "model_comparison.md")
    plot_recall(summary, OUT_DIR / "transition_recall.png")
    plot_threshold_sweep(sweep, OUT_DIR / "threshold_sweep.png")
    (OUT_DIR / "config.json").write_text(json.dumps({
        "seeds": list(seeds), "epochs": epochs, "horizon": HORIZON, "history": HISTORY,
        "anchor_history": ANCHOR_HISTORY, "lstm_histories": LSTM_HISTORIES,
        "min_duration": MIN_DURATION, "trans_weight": TRANS_WEIGHT, "test_frac": TEST_FRAC,
        "val_frac": VAL_FRAC, "split": "GroupShuffleSplit by run_id",
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY: transition recall, mean +- std over seeds")
    print("=" * 70)
    for _, r in summary.iterrows():
        print(f"  [{r['label_def']:4}] {r['model']:18} {r['trans_recall_mean']:.3f} +- "
              f"{r['trans_recall_std']:.3f}   (overall {r['overall_acc_mean']:.3f}, "
              f"F1 {r['trans_f1_mean']:.3f})")
    print(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
