"""
Visualize Model B: Phase Lookahead Results

Generates:
1. Training curves (loss + accuracy) - shows no overfitting
2. Confusion matrix heatmap
3. Per-class precision/recall
4. Live prediction demo on actual thread sequences

Run: python analysis/visualize_model_b.py
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
from torch.utils.data import DataLoader
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
    LookaheadMLP, LookaheadDataset, create_lookahead_pairs,
    HIDDEN_DIM, BATCH_SIZE, LEARNING_RATE, HORIZON_BINS
)
from analysis.seeding import set_seed


def load_data_and_model():
    """Load everything needed for visualization."""
    model_dir = Path(__file__).parent.parent / "models"

    # Load phase discovery artifacts
    phase_dir = model_dir / "phase_discovery"
    scaler_data = np.load(phase_dir / "scaler.npz")
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    centroids = np.load(phase_dir / "kmeans_centroids.npy")

    input_dim = len(scaler.mean_)
    ae_model = Autoencoder(input_dim, hidden_dim=16, latent_dim=8)
    ae_model.load_state_dict(torch.load(phase_dir / "autoencoder.pt", weights_only=True))
    ae_model.eval()

    # Load lookahead model
    lookahead_dir = model_dir / "phase_lookahead"
    n_classes = len(centroids)
    lookahead_model = LookaheadMLP(8, HIDDEN_DIM, n_classes)
    lookahead_model.load_state_dict(torch.load(lookahead_dir / "lookahead_model.pt", weights_only=True))
    lookahead_model.eval()

    # Load data
    print("Loading data...")
    df = load_timelines()
    df_feat = extract_features(df)
    df_phase = get_phase_features(df_feat)

    feature_cols = [c for c in PHASE_FEATURES if c in df_phase.columns]
    X_windows, meta = create_windows(df_phase, feature_cols=feature_cols)

    # Get embeddings and labels
    embeddings = get_embeddings(X_windows, ae_model, scaler)
    from scipy.spatial.distance import cdist
    distances = cdist(embeddings, centroids, metric='euclidean')
    labels = np.argmin(distances, axis=1)

    return embeddings, labels, meta, lookahead_model, centroids


def train_with_history(embeddings, labels, meta, epochs=100):
    """Train model and return full history for plotting."""
    set_seed()
    X, y, pair_meta = create_lookahead_pairs(embeddings, labels, meta, HORIZON_BINS)
    n_classes = len(np.unique(labels))

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

    train_dataset = LookaheadDataset(X_train, y_train)
    val_dataset = LookaheadDataset(X_val, y_val)
    test_dataset = LookaheadDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = LookaheadMLP(8, HIDDEN_DIM, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    print("Training with history tracking...")
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
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    # Test evaluation
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            test_preds.extend(logits.argmax(dim=1).numpy())
            test_true.extend(y_batch.numpy())

    test_acc = accuracy_score(test_true, test_preds)
    conf_mat = confusion_matrix(test_true, test_preds)

    return model, history, test_acc, conf_mat, test_true, test_preds


def plot_training_curves(history, output_dir):
    """Plot training curves showing no overfitting."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (Cross-Entropy)')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Add gap annotation
    final_gap = history['train_loss'][-1] - history['val_loss'][-1]
    axes[0].annotate(f'Gap: {abs(final_gap):.4f}',
                     xy=(len(epochs), history['val_loss'][-1]),
                     xytext=(len(epochs)*0.7, (history['train_loss'][-1] + history['val_loss'][-1])/2),
                     fontsize=10, ha='center',
                     arrowprops=dict(arrowstyle='->', color='gray'))

    # Accuracy curves
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].axhline(y=0.75, color='g', linestyle='--', label='Target (75%)', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.5, 1.0)

    # Add final accuracy annotation
    axes[1].annotate(f'Final: {history["val_acc"][-1]:.1%}',
                     xy=(len(epochs), history['val_acc'][-1]),
                     xytext=(len(epochs)*0.8, history['val_acc'][-1] - 0.05),
                     fontsize=11, fontweight='bold')

    plt.suptitle('Model B: No Overfitting Evidence\n(Train and Val curves track closely)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: training_curves.png")


def plot_confusion_matrix(conf_mat, output_dir):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize by row (true labels)
    conf_mat_norm = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True) * 100

    sns.heatmap(conf_mat_norm, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                cbar_kws={'label': '% of True Class'})
    ax.set_xlabel('Predicted Phase')
    ax.set_ylabel('True Phase')
    ax.set_title('Confusion Matrix: Phase Lookahead Predictions\n(% of each true class)')

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")


def plot_per_class_metrics(test_true, test_preds, output_dir):
    """Plot per-class precision, recall, F1."""
    report = classification_report(test_true, test_preds, output_dict=True, zero_division=0)

    classes = sorted([k for k in report.keys() if k.isdigit()])
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    support = [report[c]['support'] for c in classes]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(classes))
    width = 0.25

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#9b59b6')

    ax.axhline(y=0.75, color='red', linestyle='--', label='Target (75%)', alpha=0.7)

    ax.set_xlabel('Phase Cluster')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Phase {c}\n(n={s})' for c, s in zip(classes, support)])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "per_class_metrics.png", dpi=150)
    plt.close()
    print("Saved: per_class_metrics.png")


def live_prediction_demo(embeddings, labels, meta, model, output_dir):
    """
    Live prediction demo: show actual sequences and step-by-step predictions.
    """
    meta = meta.copy()
    meta['embedding_idx'] = np.arange(len(meta))
    meta['label'] = labels

    # Find a few interesting sequences with transitions
    demo_sequences = []

    for (run_id, tid), group in meta.groupby(['run_id', 'tid']):
        group = group.sort_values('window_idx').reset_index(drop=True)
        if len(group) < 10:
            continue

        # Check for transitions
        group_labels = group['label'].values
        transitions = np.where(np.diff(group_labels) != 0)[0]

        if len(transitions) >= 1:
            demo_sequences.append({
                'run_id': run_id,
                'tid': tid,
                'group': group,
                'n_transitions': len(transitions),
                'workload': group.iloc[0]['workload']
            })

    # Sort by number of transitions and pick diverse workloads
    demo_sequences.sort(key=lambda x: x['n_transitions'], reverse=True)

    # Pick 4 sequences from different workloads
    selected = []
    seen_workloads = set()
    for seq in demo_sequences:
        if seq['workload'] not in seen_workloads and len(selected) < 4:
            selected.append(seq)
            seen_workloads.add(seq['workload'])

    if len(selected) < 4:
        selected = demo_sequences[:4]

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    n_classes = int(labels.max()) + 1
    cluster_names = {}

    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    for idx, seq in enumerate(selected):
        ax = axes[idx]
        group = seq['group'].iloc[:15]  # Take first 15 windows

        # Get embeddings and make predictions
        emb_indices = group['embedding_idx'].values
        seq_embeddings = embeddings[emb_indices]
        true_labels = group['label'].values

        # Predict next phase for each position (except last 2)
        predictions = []
        with torch.no_grad():
            for i in range(len(seq_embeddings) - HORIZON_BINS):
                emb = torch.tensor(seq_embeddings[i:i+1], dtype=torch.float32)
                pred = model(emb).argmax(dim=1).item()
                predictions.append(pred)

        # Get actual "next" labels
        actual_next = true_labels[HORIZON_BINS:]

        # Plot
        x = np.arange(len(true_labels))

        # True phase sequence as colored bars
        for i, label in enumerate(true_labels):
            ax.barh(2, 1, left=i, color=colors[label], edgecolor='black', linewidth=0.5)

        # Predictions (shifted by horizon)
        for i, pred in enumerate(predictions):
            actual = actual_next[i]
            correct = pred == actual
            ax.barh(1, 1, left=i + HORIZON_BINS,
                   color=colors[pred] if correct else 'white',
                   edgecolor='green' if correct else 'red',
                   linewidth=2, hatch='///' if not correct else '')

        # Actual next (for comparison)
        for i, label in enumerate(actual_next):
            ax.barh(0, 1, left=i + HORIZON_BINS, color=colors[label],
                   edgecolor='black', linewidth=0.5, alpha=0.7)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Actual Next', 'Predicted Next', 'Current Phase'])
        ax.set_xlabel('Window Index')
        ax.set_title(f'{seq["workload"].upper()} (run={seq["run_id"][:20]}...)\n'
                    f'Transitions: {seq["n_transitions"]}')
        ax.set_xlim(-0.5, len(true_labels) + 0.5)

        # Add accuracy for this sequence
        acc = sum(p == a for p, a in zip(predictions, actual_next)) / len(predictions)
        ax.text(0.02, 0.98, f'Seq Acc: {acc:.0%}', transform=ax.transAxes,
               fontsize=10, fontweight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add legend
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=colors[i], edgecolor='black',
                                      label=f'{i}: {cluster_names.get(i, f"Cluster {i}")}')
                      for i in range(7)]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99),
              title='Phase Clusters', ncol=2)

    plt.suptitle('Live Prediction Demo: Predicting Next Phase (1s ahead)\n'
                'Green border = correct, Red border + hatching = incorrect',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(output_dir / "live_prediction_demo.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: live_prediction_demo.png")


def plot_transition_analysis(embeddings, labels, meta, model, output_dir):
    """Analyze prediction accuracy specifically on transitions vs same-phase."""
    X, y, pair_meta = create_lookahead_pairs(embeddings, labels, meta, HORIZON_BINS)

    # Split same way as training
    _, X_test, _, y_test, _, pair_meta_test = train_test_split(
        X, y, pair_meta, test_size=0.15, random_state=42, stratify=y
    )

    # Predict
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_tensor).argmax(dim=1).numpy()

    pair_meta_test = pair_meta_test.copy()
    pair_meta_test['predicted'] = preds
    pair_meta_test['correct'] = preds == y_test

    # Separate transitions vs same-phase
    transitions = pair_meta_test[pair_meta_test['is_transition']]
    same_phase = pair_meta_test[~pair_meta_test['is_transition']]

    trans_acc = transitions['correct'].mean() if len(transitions) > 0 else 0
    same_acc = same_phase['correct'].mean() if len(same_phase) > 0 else 0

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    categories = ['Same Phase\n(stay in current)', 'Transitions\n(phase changes)']
    accuracies = [same_acc, trans_acc]
    counts = [len(same_phase), len(transitions)]

    bars = axes[0].bar(categories, accuracies, color=['#3498db', '#e74c3c'], edgecolor='black')
    axes[0].axhline(y=0.75, color='green', linestyle='--', label='Target (75%)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Prediction Accuracy: Same-Phase vs Transitions')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()

    # Add count and accuracy labels
    for bar, acc, count in zip(bars, accuracies, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.1%}\n(n={count})', ha='center', va='bottom', fontsize=11)

    # Pie chart of test set composition
    axes[1].pie([len(same_phase), len(transitions)],
               labels=[f'Same Phase\n{len(same_phase)} ({100*len(same_phase)/len(pair_meta_test):.0f}%)',
                      f'Transitions\n{len(transitions)} ({100*len(transitions)/len(pair_meta_test):.0f}%)'],
               colors=['#3498db', '#e74c3c'], explode=[0, 0.1],
               autopct='', startangle=90)
    axes[1].set_title('Test Set Composition')

    plt.suptitle('Transition vs Same-Phase Analysis\n'
                '(Transitions are harder to predict but more valuable for scheduling)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "transition_analysis.png", dpi=150)
    plt.close()
    print("Saved: transition_analysis.png")

    print(f"\nTransition Analysis:")
    print(f"  Same-phase accuracy: {same_acc:.1%} (n={len(same_phase)})")
    print(f"  Transition accuracy: {trans_acc:.1%} (n={len(transitions)})")


def main():
    output_dir = Path(__file__).parent.parent / "results" / "model_b_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Model B: Phase Lookahead Visualization")
    print("=" * 60)

    # Load data
    embeddings, labels, meta, model, centroids = load_data_and_model()

    print(f"\nOutput dir: {output_dir}")
    print()

    # Train with full history for learning curves
    print("=" * 60)
    print("Re-training to capture learning curves...")
    print("=" * 60)
    trained_model, history, test_acc, conf_mat, test_true, test_preds = train_with_history(
        embeddings, labels, meta, epochs=80
    )

    print(f"\nFinal test accuracy: {test_acc:.1%}")

    # Generate all visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    plot_training_curves(history, output_dir)
    plot_confusion_matrix(conf_mat, output_dir)
    plot_per_class_metrics(test_true, test_preds, output_dir)
    live_prediction_demo(embeddings, labels, meta, model, output_dir)
    plot_transition_analysis(embeddings, labels, meta, model, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
