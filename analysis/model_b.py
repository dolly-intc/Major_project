"""
Model B: Phase Lookahead Predictor

Predicts the next phase (1 second / 2 bins ahead) from the current phase embedding.
This enables proactive thread migration before phase transitions.

Target: ≥75% phase transition prediction accuracy (from PROJECT_CONTEXT.md §9)

Architecture options:
1. MLP: Current embedding (8-dim) → hidden → next phase (k classes)
2. LSTM: Sequence of embeddings → next phase

We use MLP for simplicity — it can be reimplemented in stdlib for Stage 4.

Usage:
    from analysis.model_b import train_lookahead_model, evaluate_model
    model, metrics = train_lookahead_model(embeddings, labels, meta)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.seeding import set_seed, DEFAULT_SEED


# Lookahead parameters
HORIZON_BINS = 2  # 2 bins = 1 second lookahead (at 0.5s per bin)

# Model architecture
HIDDEN_DIM = 32
DROPOUT = 0.2

# Training parameters
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 10  # Early stopping patience


class LookaheadMLP(nn.Module):
    """Simple MLP for next-phase prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class LookaheadLSTM(nn.Module):
    """LSTM for sequence-based next-phase prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int,
                 n_layers: int = 2, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (n_layers, batch, hidden_dim) — take last layer
        out = self.fc(h_n[-1])
        return out


class LookaheadDataset(Dataset):
    """Dataset for (current_embedding, next_phase) pairs."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def create_lookahead_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    horizon_bins: int = HORIZON_BINS,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Create (current_embedding, next_phase_label) pairs for training.

    For each window, the target is the cluster label of the window `horizon_bins`
    steps ahead in the same thread's execution sequence.

    Args:
        embeddings: Window embeddings from autoencoder, shape (n_windows, latent_dim)
        labels: Cluster labels for each window
        meta: Window metadata with run_id, tid, window_idx
        horizon_bins: How many bins ahead to predict (default 2 = 1 second)

    Returns:
        X: Current embeddings for windows that have a valid future
        y: Next-phase labels (cluster of window horizon_bins ahead)
        pair_meta: Metadata for each pair (for analysis)
    """
    meta = meta.copy()
    meta["embedding_idx"] = np.arange(len(meta))
    meta["label"] = labels

    X_list = []
    y_list = []
    pair_meta_list = []

    # Group by (run_id, tid) — each thread's sequence
    for (run_id, tid), group in meta.groupby(["run_id", "tid"]):
        group = group.sort_values("window_idx").reset_index(drop=True)

        # For each window except the last `horizon_bins`
        for i in range(len(group) - horizon_bins):
            current_row = group.iloc[i]
            future_row = group.iloc[i + horizon_bins]

            # Check window indices are actually consecutive
            if future_row["window_idx"] != current_row["window_idx"] + horizon_bins:
                continue  # Gap in sequence, skip

            current_idx = current_row["embedding_idx"]
            future_label = future_row["label"]

            X_list.append(embeddings[current_idx])
            y_list.append(future_label)
            pair_meta_list.append({
                "run_id": run_id,
                "tid": tid,
                "current_window_idx": current_row["window_idx"],
                "future_window_idx": future_row["window_idx"],
                "current_label": current_row["label"],
                "future_label": future_label,
                "is_transition": current_row["label"] != future_label,
            })

    if not X_list:
        raise ValueError("No valid lookahead pairs found")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    pair_meta = pd.DataFrame(pair_meta_list)

    # Statistics
    n_transitions = pair_meta["is_transition"].sum()
    print(f"Created {len(X)} lookahead pairs")
    print(f"  Transitions: {n_transitions} ({100*n_transitions/len(X):.1f}%)")
    print(f"  Same-phase: {len(X) - n_transitions} ({100*(len(X)-n_transitions)/len(X):.1f}%)")

    return X, y, pair_meta


def create_sequence_data(
    embeddings: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    seq_len: int = 4,
    horizon_bins: int = HORIZON_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.

    Each sample is (sequence of `seq_len` embeddings, next_phase_label).
    """
    meta = meta.copy()
    meta["embedding_idx"] = np.arange(len(meta))
    meta["label"] = labels

    X_list = []
    y_list = []

    for (run_id, tid), group in meta.groupby(["run_id", "tid"]):
        group = group.sort_values("window_idx").reset_index(drop=True)

        # Need at least seq_len + horizon_bins windows
        if len(group) < seq_len + horizon_bins:
            continue

        for i in range(len(group) - seq_len - horizon_bins + 1):
            # Check consecutive
            window_indices = group.iloc[i:i + seq_len + horizon_bins]["window_idx"].values
            expected = np.arange(window_indices[0], window_indices[0] + seq_len + horizon_bins)
            if not np.array_equal(window_indices, expected):
                continue

            # Sequence of embeddings
            seq_indices = group.iloc[i:i + seq_len]["embedding_idx"].values
            seq = embeddings[seq_indices]

            # Target: label of window seq_len + horizon_bins - 1 ahead of start
            # (i.e., horizon_bins ahead of the last window in sequence)
            future_idx = group.iloc[i + seq_len + horizon_bins - 1]["embedding_idx"]
            future_label = labels[future_idx]

            X_list.append(seq)
            y_list.append(future_label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"Created {len(X)} sequences (seq_len={seq_len}, horizon={horizon_bins})")
    return X, y


def train_lookahead_model(
    embeddings: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    model_type: str = "mlp",
    horizon_bins: int = HORIZON_BINS,
    hidden_dim: int = HIDDEN_DIM,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    patience: int = PATIENCE,
    val_split: float = 0.15,
    test_split: float = 0.15,
    verbose: bool = True,
    seed: int = DEFAULT_SEED,
) -> Tuple[nn.Module, Dict]:
    """
    Train phase lookahead model.

    Args:
        embeddings: Window embeddings from autoencoder
        labels: Cluster labels
        meta: Window metadata
        model_type: "mlp" or "lstm"
        horizon_bins: Prediction horizon
        hidden_dim: Hidden layer size
        epochs: Max training epochs
        batch_size: Batch size
        lr: Learning rate
        patience: Early stopping patience
        val_split: Validation split fraction
        test_split: Test split fraction
        verbose: Print training progress
        seed: Random seed for reproducible weight init and shuffling

    Returns:
        model: Trained model
        metrics: Dict with accuracy, confusion matrix, etc.
    """
    set_seed(seed)

    # Create pairs
    if model_type == "mlp":
        X, y, pair_meta = create_lookahead_pairs(embeddings, labels, meta, horizon_bins)
    else:
        X, y = create_sequence_data(embeddings, labels, meta, seq_len=4, horizon_bins=horizon_bins)
        pair_meta = None

    n_classes = len(np.unique(labels))
    input_dim = X.shape[-1]  # Last dim is embedding size

    # Train/val/test split (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split / (1 - test_split), random_state=42, stratify=y_temp
    )

    if verbose:
        print(f"\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        print(f"Class distribution (train): {Counter(y_train)}")

    # Datasets and loaders
    train_dataset = LookaheadDataset(X_train, y_train)
    val_dataset = LookaheadDataset(X_val, y_val)
    test_dataset = LookaheadDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    if model_type == "mlp":
        model = LookaheadMLP(input_dim, hidden_dim, n_classes)
    else:
        model = LookaheadLSTM(input_dim, hidden_dim, n_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training with early stopping
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                val_preds.extend(logits.argmax(dim=1).numpy())
                val_true.extend(y_batch.numpy())
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    test_preds = []
    test_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            test_preds.extend(logits.argmax(dim=1).numpy())
            test_true.extend(y_batch.numpy())

    test_acc = accuracy_score(test_true, test_preds)
    conf_matrix = confusion_matrix(test_true, test_preds)

    if verbose:
        print(f"\n{'='*50}")
        print(f"TEST ACCURACY: {test_acc:.3f} ({test_acc*100:.1f}%)")
        print(f"{'='*50}")
        result = "PASS" if test_acc >= 0.75 else "FAIL"
        print(f"\nTarget: >=75% -> {result}")

    # Detailed metrics
    metrics = {
        "test_accuracy": test_acc,
        "val_accuracy": val_acc,
        "confusion_matrix": conf_matrix,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_classes": n_classes,
        "model_type": model_type,
        "horizon_bins": horizon_bins,
    }

    # Per-class report
    if verbose:
        print("\nClassification Report:")
        print(classification_report(test_true, test_preds, zero_division=0))

    return model, metrics


def analyze_transitions(pair_meta: pd.DataFrame, test_preds: np.ndarray, test_true: np.ndarray):
    """Analyze prediction accuracy on phase transitions vs same-phase."""
    # This requires pair_meta to be aligned with test set — skipping for now
    pass


def save_model(model: nn.Module, metrics: Dict, output_dir: Path):
    """Save trained model and metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "lookahead_model.pt")

    # Save metrics (excluding confusion matrix which is numpy)
    import json
    metrics_save = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    metrics_save["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    with open(output_dir / "lookahead_metrics.json", "w") as f:
        json.dump(metrics_save, f, indent=2)

    print(f"Model saved to {output_dir}")


def load_phase_discovery_artifacts():
    """Load embeddings and labels from Stage 2."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analysis.load_dataset import load_timelines
    from analysis.features import extract_features, get_phase_features, PHASE_FEATURES
    from analysis.windows import create_windows
    from analysis.model_a import Autoencoder, get_embeddings

    model_dir = Path(__file__).parent.parent / "models" / "phase_discovery"

    # Load scaler
    scaler_data = np.load(model_dir / "scaler.npz")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    # Load centroids for labeling
    centroids = np.load(model_dir / "kmeans_centroids.npy")

    # Load autoencoder
    input_dim = len(scaler.mean_)
    ae_model = Autoencoder(input_dim, hidden_dim=16, latent_dim=8)
    ae_model.load_state_dict(torch.load(model_dir / "autoencoder.pt", weights_only=True))
    ae_model.eval()

    # Load and process data
    print("Loading timeline data...")
    df = load_timelines()
    df_feat = extract_features(df)
    df_phase = get_phase_features(df_feat)

    feature_cols = [c for c in PHASE_FEATURES if c in df_phase.columns]
    X_windows, meta = create_windows(df_phase, feature_cols=feature_cols)

    # Get embeddings
    print("Computing embeddings...")
    embeddings = get_embeddings(X_windows, ae_model, scaler)

    # Assign clusters
    from scipy.spatial.distance import cdist
    distances = cdist(embeddings, centroids, metric='euclidean')
    labels = np.argmin(distances, axis=1)

    return embeddings, labels, meta


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 60)
    print("Model B: Phase Lookahead Training")
    print("=" * 60)

    # Load data from Stage 2
    embeddings, labels, meta = load_phase_discovery_artifacts()

    print(f"\nEmbeddings: {embeddings.shape}")
    print(f"Labels: {len(labels)} (classes: {len(np.unique(labels))})")

    # Train MLP model
    print("\n" + "=" * 60)
    print("Training MLP Lookahead Model")
    print("=" * 60)

    model, metrics = train_lookahead_model(
        embeddings, labels, meta,
        model_type="mlp",
        epochs=100,
        verbose=True,
    )

    # Save model
    output_dir = Path(__file__).parent.parent / "models" / "phase_lookahead"
    save_model(model, metrics, output_dir)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model type: {metrics['model_type']}")
    print(f"Horizon: {metrics['horizon_bins']} bins ({metrics['horizon_bins'] * 0.5}s)")
    print(f"Test accuracy: {metrics['test_accuracy']:.1%}")
    print(f"Target (>=75%): {'PASS' if metrics['test_accuracy'] >= 0.75 else 'FAIL'}")
