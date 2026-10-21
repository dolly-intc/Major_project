"""
Compare MLP vs LSTM for Phase Lookahead

The hypothesis: LSTM can look at a sequence of past embeddings and detect
patterns that precede phase transitions, improving transition accuracy.

Run: python analysis/train_lstm_compare.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.load_dataset import load_timelines
from analysis.features import extract_features, get_phase_features, PHASE_FEATURES
from analysis.windows import create_windows
from analysis.model_a import Autoencoder, get_embeddings
from analysis.model_b import (
    LookaheadMLP, LookaheadLSTM, LookaheadDataset,
    HIDDEN_DIM, BATCH_SIZE, LEARNING_RATE, HORIZON_BINS
)
from analysis.seeding import set_seed


class SequenceDataset(Dataset):
    """Dataset for (sequence of embeddings, next_phase) pairs."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def create_sequence_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    seq_len: int = 4,
    horizon_bins: int = HORIZON_BINS,
):
    """
    Create (sequence of embeddings, next_phase_label) pairs for LSTM.

    Each sample is a sequence of `seq_len` consecutive embeddings,
    and the target is the phase label `horizon_bins` after the last embedding.
    """
    meta = meta.copy()
    meta['embedding_idx'] = np.arange(len(meta))
    meta['label'] = labels

    X_list = []
    y_list = []
    is_transition_list = []

    for (run_id, tid), group in meta.groupby(['run_id', 'tid']):
        group = group.sort_values('window_idx').reset_index(drop=True)

        # Need at least seq_len + horizon_bins windows
        if len(group) < seq_len + horizon_bins:
            continue

        for i in range(len(group) - seq_len - horizon_bins + 1):
            # Check consecutive window indices
            window_indices = group.iloc[i:i + seq_len + horizon_bins]['window_idx'].values
            expected = np.arange(window_indices[0], window_indices[0] + seq_len + horizon_bins)
            if not np.array_equal(window_indices, expected):
                continue

            # Sequence of embeddings
            seq_indices = group.iloc[i:i + seq_len]['embedding_idx'].values
            seq = embeddings[seq_indices]

            # Current label (last in sequence)
            current_label = labels[seq_indices[-1]]

            # Target: label horizon_bins after the sequence ends
            future_idx = group.iloc[i + seq_len - 1 + horizon_bins]['embedding_idx']
            future_label = labels[future_idx]

            X_list.append(seq)
            y_list.append(future_label)
            is_transition_list.append(current_label != future_label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    is_transition = np.array(is_transition_list)

    n_trans = is_transition.sum()
    print(f"Created {len(X)} sequences (seq_len={seq_len}, horizon={horizon_bins})")
    print(f"  Transitions: {n_trans} ({100*n_trans/len(X):.1f}%)")
    print(f"  Same-phase: {len(X) - n_trans} ({100*(len(X)-n_trans)/len(X):.1f}%)")

    return X, y, is_transition


def create_mlp_pairs(embeddings, labels, meta, horizon_bins=HORIZON_BINS):
    """Create single-embedding pairs for MLP (same as model_b)."""
    meta = meta.copy()
    meta['embedding_idx'] = np.arange(len(meta))
    meta['label'] = labels

    X_list = []
    y_list = []
    is_transition_list = []

    for (run_id, tid), group in meta.groupby(['run_id', 'tid']):
        group = group.sort_values('window_idx').reset_index(drop=True)

        for i in range(len(group) - horizon_bins):
            current_row = group.iloc[i]
            future_row = group.iloc[i + horizon_bins]

            if future_row['window_idx'] != current_row['window_idx'] + horizon_bins:
                continue

            current_idx = current_row['embedding_idx']
            future_label = future_row['label']
            current_label = current_row['label']

            X_list.append(embeddings[current_idx])
            y_list.append(future_label)
            is_transition_list.append(current_label != future_label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    is_transition = np.array(is_transition_list)

    return X, y, is_transition


def train_and_evaluate(model, train_loader, val_loader, test_loader,
                       is_transition_test, epochs=100, patience=15, lr=LEARNING_RATE):
    """Train model and return metrics including transition accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds, train_true = [], []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(logits.argmax(dim=1).numpy())
            train_true.extend(y_batch.numpy())

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_true, train_preds)

        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                val_preds.extend(logits.argmax(dim=1).numpy())
                val_true.extend(y_batch.numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            test_preds.extend(logits.argmax(dim=1).numpy())
            test_true.extend(y_batch.numpy())

    test_preds = np.array(test_preds)
    test_true = np.array(test_true)

    # Overall accuracy
    test_acc = accuracy_score(test_true, test_preds)

    # Transition vs same-phase accuracy
    trans_mask = is_transition_test
    same_mask = ~is_transition_test

    trans_acc = accuracy_score(test_true[trans_mask], test_preds[trans_mask]) if trans_mask.sum() > 0 else 0
    same_acc = accuracy_score(test_true[same_mask], test_preds[same_mask]) if same_mask.sum() > 0 else 0

    return {
        'test_acc': test_acc,
        'trans_acc': trans_acc,
        'same_acc': same_acc,
        'n_trans': trans_mask.sum(),
        'n_same': same_mask.sum(),
        'history': history,
        'test_preds': test_preds,
        'test_true': test_true,
    }


def save_best_model(model, model_name, seq_len, metrics, n_classes, output_dir):
    """Persist the winning model's weights + metrics so results are reproducible from this script."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "lstm_model.pt")

    save_metrics = {
        "model_type": "lstm",
        "model_name": model_name,
        "seq_len": seq_len,
        "hidden_dim": HIDDEN_DIM,
        "n_layers": 2,
        "n_classes": n_classes,
        "horizon_bins": HORIZON_BINS,
        "test_accuracy": float(metrics["test_acc"]),
        "transition_accuracy": float(metrics["trans_acc"]),
        "same_phase_accuracy": float(metrics["same_acc"]),
        "n_trans": int(metrics["n_trans"]),
        "n_same": int(metrics["n_same"]),
    }
    with open(output_dir / "lstm_metrics.json", "w") as f:
        json.dump(save_metrics, f, indent=2)

    print(f"\nSaved best model ({model_name}) to {output_dir}")


def load_data():
    """Load embeddings and labels."""
    model_dir = Path(__file__).parent.parent / "models" / "phase_discovery"

    scaler_data = np.load(model_dir / "scaler.npz")
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    centroids = np.load(model_dir / "kmeans_centroids.npy")

    input_dim = len(scaler.mean_)
    ae_model = Autoencoder(input_dim, hidden_dim=16, latent_dim=8)
    ae_model.load_state_dict(torch.load(model_dir / "autoencoder.pt", weights_only=True))
    ae_model.eval()

    print("Loading data...")
    df = load_timelines()
    df_feat = extract_features(df)
    df_phase = get_phase_features(df_feat)

    feature_cols = [c for c in PHASE_FEATURES if c in df_phase.columns]
    X_windows, meta = create_windows(df_phase, feature_cols=feature_cols)

    embeddings = get_embeddings(X_windows, ae_model, scaler)

    from scipy.spatial.distance import cdist
    distances = cdist(embeddings, centroids, metric='euclidean')
    labels = np.argmin(distances, axis=1)

    return embeddings, labels, meta, len(centroids)


def main():
    # Fix the seed: model init and DataLoader shuffling are otherwise random, which
    # can flip which variant "wins" on transition accuracy between runs.
    set_seed()

    output_dir = Path(__file__).parent.parent / "results" / "lstm_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MLP vs LSTM Comparison for Phase Lookahead")
    print("=" * 70)

    embeddings, labels, meta, n_classes = load_data()
    print(f"\nEmbeddings: {embeddings.shape}, Classes: {n_classes}")

    results = {}
    trained_models = {}

    # ============================================================
    # Train MLP (baseline)
    # ============================================================
    print("\n" + "=" * 70)
    print("Training MLP (single embedding input)")
    print("=" * 70)

    X_mlp, y_mlp, is_trans_mlp = create_mlp_pairs(embeddings, labels, meta)

    # Split
    X_temp, X_test, y_temp, y_test, trans_temp, trans_test = train_test_split(
        X_mlp, y_mlp, is_trans_mlp, test_size=0.15, random_state=42, stratify=y_mlp
    )
    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        X_temp, y_temp, trans_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    train_loader = DataLoader(LookaheadDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(LookaheadDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(LookaheadDataset(X_test, y_test), batch_size=BATCH_SIZE)

    mlp_model = LookaheadMLP(8, HIDDEN_DIM, n_classes)
    results['MLP'] = train_and_evaluate(mlp_model, train_loader, val_loader, test_loader, trans_test)
    trained_models['MLP'] = mlp_model

    print(f"\nMLP Results:")
    print(f"  Overall accuracy: {results['MLP']['test_acc']:.1%}")
    print(f"  Same-phase accuracy: {results['MLP']['same_acc']:.1%} (n={results['MLP']['n_same']})")
    print(f"  Transition accuracy: {results['MLP']['trans_acc']:.1%} (n={results['MLP']['n_trans']})")

    # ============================================================
    # Train LSTM variants with different sequence lengths
    # ============================================================
    for seq_len in [4, 6, 8]:
        print("\n" + "=" * 70)
        print(f"Training LSTM (seq_len={seq_len})")
        print("=" * 70)

        X_seq, y_seq, is_trans_seq = create_sequence_pairs(embeddings, labels, meta, seq_len=seq_len)

        if len(X_seq) < 1000:
            print(f"  Not enough sequences ({len(X_seq)}), skipping")
            continue

        # Split
        X_temp, X_test, y_temp, y_test, trans_temp, trans_test = train_test_split(
            X_seq, y_seq, is_trans_seq, test_size=0.15, random_state=42, stratify=y_seq
        )
        X_train, X_val, y_train, y_val, _, _ = train_test_split(
            X_temp, y_temp, trans_temp, test_size=0.176, random_state=42, stratify=y_temp
        )

        train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE)
        test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=BATCH_SIZE)

        lstm_model = LookaheadLSTM(8, HIDDEN_DIM, n_classes, n_layers=2)
        results[f'LSTM-{seq_len}'] = train_and_evaluate(
            lstm_model, train_loader, val_loader, test_loader, trans_test
        )
        trained_models[f'LSTM-{seq_len}'] = lstm_model

        print(f"\nLSTM-{seq_len} Results:")
        print(f"  Overall accuracy: {results[f'LSTM-{seq_len}']['test_acc']:.1%}")
        print(f"  Same-phase accuracy: {results[f'LSTM-{seq_len}']['same_acc']:.1%}")
        print(f"  Transition accuracy: {results[f'LSTM-{seq_len}']['trans_acc']:.1%}")

    # ============================================================
    # Visualize comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("Generating comparison visualizations")
    print("=" * 70)

    # Summary comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = list(results.keys())
    overall_acc = [results[m]['test_acc'] for m in models]
    same_acc = [results[m]['same_acc'] for m in models]
    trans_acc = [results[m]['trans_acc'] for m in models]

    x = np.arange(len(models))
    width = 0.6

    # Overall accuracy
    bars = axes[0].bar(x, overall_acc, width, color='#3498db', edgecolor='black')
    axes[0].axhline(y=0.75, color='green', linestyle='--', label='Target (75%)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Overall Test Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    for bar, acc in zip(bars, overall_acc):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Same-phase accuracy
    bars = axes[1].bar(x, same_acc, width, color='#2ecc71', edgecolor='black')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Same-Phase Accuracy\n(predicting "stay in current phase")')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylim(0, 1.05)
    for bar, acc in zip(bars, same_acc):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Transition accuracy (the key metric!)
    bars = axes[2].bar(x, trans_acc, width, color='#e74c3c', edgecolor='black')
    axes[2].axhline(y=0.5, color='gray', linestyle=':', label='Random (50%)')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('TRANSITION Accuracy\n(predicting phase changes - KEY METRIC)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models)
    axes[2].set_ylim(0, 1.05)
    axes[2].legend()
    for bar, acc in zip(bars, trans_acc):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('MLP vs LSTM Comparison: Phase Lookahead Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison.png")

    # Training curves comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'MLP': '#3498db', 'LSTM-4': '#e74c3c', 'LSTM-6': '#2ecc71', 'LSTM-8': '#9b59b6'}

    for model_name, res in results.items():
        hist = res['history']
        epochs = range(1, len(hist['train_loss']) + 1)
        color = colors.get(model_name, 'gray')

        axes[0, 0].plot(epochs, hist['train_loss'], '-', color=color, label=f'{model_name}', linewidth=2)
        axes[0, 1].plot(epochs, hist['val_loss'], '-', color=color, label=f'{model_name}', linewidth=2)
        axes[1, 0].plot(epochs, hist['train_acc'], '-', color=color, label=f'{model_name}', linewidth=2)
        axes[1, 1].plot(epochs, hist['val_acc'], '-', color=color, label=f'{model_name}', linewidth=2)

    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0.5, 1.0)

    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].axhline(y=0.75, color='green', linestyle='--', alpha=0.7, label='Target')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0.5, 1.0)

    plt.suptitle('Training Curves: MLP vs LSTM Variants', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves_comparison.png", dpi=150)
    plt.close()
    print("Saved: training_curves_comparison.png")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'Overall':>10} {'Same-Phase':>12} {'Transition':>12} {'Trans N':>10}")
    print("-" * 60)
    for model_name, res in results.items():
        print(f"{model_name:<12} {res['test_acc']:>10.1%} {res['same_acc']:>12.1%} "
              f"{res['trans_acc']:>12.1%} {res['n_trans']:>10}")

    # Find best model for transitions
    best_trans_model = max(results.keys(), key=lambda m: results[m]['trans_acc'])
    print(f"\nBest for transitions: {best_trans_model} ({results[best_trans_model]['trans_acc']:.1%})")

    # Save best LSTM model if it beats MLP on the metric that matters (transition accuracy)
    if 'LSTM' in best_trans_model and results[best_trans_model]['trans_acc'] > results['MLP']['trans_acc']:
        print(f"\nLSTM improves transition accuracy by {results[best_trans_model]['trans_acc'] - results['MLP']['trans_acc']:.1%}")

        seq_len = int(best_trans_model.split('-')[1])
        lstm_output_dir = Path(__file__).parent.parent / "models" / "phase_lookahead_lstm"
        save_best_model(
            trained_models[best_trans_model], best_trans_model, seq_len,
            results[best_trans_model], n_classes, lstm_output_dir,
        )
    else:
        print(f"\nNo LSTM variant beat the MLP on transition accuracy; not saving an LSTM artifact.")

    # Save results
    summary = {
        model_name: {
            'test_acc': float(res['test_acc']),
            'same_acc': float(res['same_acc']),
            'trans_acc': float(res['trans_acc']),
            'n_trans': int(res['n_trans']),
            'n_same': int(res['n_same']),
        }
        for model_name, res in results.items()
    }
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
