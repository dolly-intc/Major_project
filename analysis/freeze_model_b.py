"""
Freeze the chosen Model B variant (2026-10-18 decision) as a deployable artifact.

Chosen: MLP on context features, x4 transition-weighted cross-entropy, hysteresis labels
(`mlp_ctx_weighted` / `hyst` in results/model_b_v2/). Chosen for the highest transition recall
(68.5 +- 5.4% on the 2026-10-21 core-aware phases; 66.8 +- 2.2% on the earlier phases) and
because an MLP is easy to re-implement in the stdlib-only placement service.

Trains once with DEFAULT_SEED and a run-level split (val used for early stopping, test held
out), then writes to models/phase_lookahead_v2/:
  model.pt      - PyTorch state dict
  model.json    - weights, feature scaler, feature layout and constants as plain lists,
                  loadable with the json module alone (for service/)
  metrics.json  - this seed's test metrics + the 5-seed mean/std from results/model_b_v2/

Verifies that a pure-Python forward pass over model.json reproduces the PyTorch predictions.
Does not touch models/phase_lookahead/ or models/phase_lookahead_lstm/.

Run:  python analysis/freeze_model_b.py      (requires results/model_b_v2/summary.csv)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.seeding import set_seed, DEFAULT_SEED
from analysis import model_b_v2 as v2

LABEL_DEF = "hyst"
SPEC = {"kind": "mlp", "feats": "ctx", "mode": "weighted"}
MODEL_NAME = "mlp_ctx_weighted"
OUT = v2.ROOT / "models" / "phase_lookahead_v2"


def feature_layout(n_classes: int, emb_dim: int, raw_dim: int) -> list:
    """Order of the 'ctx' input vector, matching v2.build_features."""
    k = n_classes
    return [
        ("embedding_t", emb_dim), ("centroid_distances_t", k), ("centroid_margin_t", 1),
        ("raw_phase_onehot_t", k), ("committed_phase_onehot_t", k),
        ("run_length_t_div_history_capped", 1),
        ("embedding_t_minus_embedding_t-1", emb_dim), ("embedding_t_minus_embedding_t-2", emb_dim),
        ("raw_phase_onehot_t-1", k), ("raw_phase_onehot_t-2", k),
        ("model_a_scaled_window_t", raw_dim),
    ]


def pure_python_forward(params: dict, x: list) -> int:
    """MLP forward pass with lists only: what service/ will do. Returns the argmax class."""
    x = [(xi - m) / s for xi, m, s in zip(x, params["input_mean"], params["input_scale"])]
    for layer in params["layers"]:
        x = [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(layer["weight"], layer["bias"])]
        if layer["relu"]:
            x = [max(0.0, v) for v in x]
    return max(range(len(x)), key=x.__getitem__)


def main():
    if not (v2.OUT_DIR / "summary.csv").exists():
        raise SystemExit("Run analysis/model_b_v2.py first (needs results/model_b_v2/summary.csv).")

    X_win, emb, dists, labels, meta = v2.load_windows()
    n_classes = dists.shape[1]
    A = v2.build_anchors(labels, meta)
    X = v2.build_features(X_win, emb, dists, labels, A, n_classes)["mlp_ctx"]
    y, ref = v2.targets_for(LABEL_DEF, labels, A)

    train, val, test = v2.split_runs(A["run_id"], DEFAULT_SEED)
    set_seed(DEFAULT_SEED)
    model, Xt, sc = v2.train_torch(SPEC, X, y, ref, train, val, n_classes, v2.EPOCHS)
    pred = model(Xt[torch.from_numpy(test)])[0].argmax(1).numpy()
    test_metrics = {k: float(v) for k, v in v2.score(pred, y[test], ref[test]).items()}
    print(f"Seed {DEFAULT_SEED} test: " + ", ".join(f"{k} {v:.3f}" for k, v in test_metrics.items()))

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "model.pt")

    sd = {k: v.numpy().tolist() for k, v in model.state_dict().items()}
    layout = feature_layout(n_classes, emb.shape[1], X_win.shape[1])
    assert sum(n for _, n in layout) == X.shape[1], "feature layout out of sync with build_features"
    params = {
        "model": MODEL_NAME,
        "label_definition": LABEL_DEF,
        "constants": {"horizon_windows": v2.HORIZON, "history_windows": v2.HISTORY,
                      "min_duration_windows": v2.MIN_DURATION, "n_classes": n_classes},
        "feature_layout": [{"name": n, "size": s} for n, s in layout],
        "input_mean": sc.mean_.tolist(),
        "input_scale": sc.scale_.tolist(),
        "layers": [
            {"weight": sd["enc.0.weight"], "bias": sd["enc.0.bias"], "relu": True},
            {"weight": sd["enc.3.weight"], "bias": sd["enc.3.bias"], "relu": True},
            {"weight": sd["dest.weight"], "bias": sd["dest.bias"], "relu": False},
        ],
        "output": "argmax over n_classes = predicted (smoothed) phase HORIZON windows ahead",
    }
    (OUT / "model.json").write_text(json.dumps(params), encoding="utf-8")

    # The JSON copy must reproduce PyTorch exactly (on raw, unscaled features).
    check = test[:500]
    loaded = json.loads((OUT / "model.json").read_text(encoding="utf-8"))
    py_pred = np.array([pure_python_forward(loaded, X[i].astype(float).tolist()) for i in check])
    agree = float((py_pred == pred[:len(check)]).mean())
    print(f"Pure-Python forward pass agrees with PyTorch on {agree:.1%} of {len(check)} test anchors")
    assert agree == 1.0

    summary = pd.read_csv(v2.OUT_DIR / "summary.csv")
    row = summary[(summary["label_def"] == LABEL_DEF) & (summary["model"] == MODEL_NAME)].iloc[0]
    multi_seed = {c: float(row[c]) for c in summary.columns if c.endswith(("_mean", "_std"))}
    (OUT / "metrics.json").write_text(json.dumps({
        "model": MODEL_NAME, "label_definition": LABEL_DEF,
        "split": "GroupShuffleSplit by run_id", "train_seed": DEFAULT_SEED,
        "this_artifact_test_metrics": test_metrics,
        "five_seed_mean_std": multi_seed,
        "note": "Report the five-seed numbers; the single-seed numbers describe this artifact only.",
    }, indent=2), encoding="utf-8")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
