"""
Model A: Phase Discovery with Autoencoder + Clustering

Uses a small autoencoder to learn a low-dimensional representation of
execution windows, then clusters the embeddings to discover behavioral
phases.

This is deliberately simple so Stage 4/5 can re-implement inference
in pure Python (no PyTorch) for the placement service.

Usage:
    from analysis.model_a import train_autoencoder, cluster_phases
    encoder, scaler = train_autoencoder(X)
    embeddings = encoder(X)
    labels, kmeans = cluster_phases(embeddings)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.seeding import set_seed, DEFAULT_SEED


# Model architecture
HIDDEN_DIM = 16
LATENT_DIM = 8

# Training parameters
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3


class Autoencoder(nn.Module):
    """Simple autoencoder with Linear layers."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 latent_dim: int = LATENT_DIM):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(
    X: np.ndarray,
    hidden_dim: int = HIDDEN_DIM,
    latent_dim: int = LATENT_DIM,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    val_split: float = 0.1,
    verbose: bool = True,
    seed: int = DEFAULT_SEED,
) -> Tuple[nn.Module, StandardScaler]:
    """
    Train autoencoder on window data.

    Args:
        X: Window data, shape (n_windows, n_features * window_size)
        hidden_dim: Hidden layer dimension
        latent_dim: Bottleneck dimension
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        val_split: Fraction of data for validation
        verbose: Print training progress
        seed: Random seed for reproducible weight init, split, and shuffling

    Returns:
        encoder: Trained encoder (callable, returns embeddings)
        scaler: Fitted StandardScaler (needed for inference)
    """
    set_seed(seed)

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Train/val split
    n_val = int(len(X_scaled) * val_split)
    indices = np.random.RandomState(seed).permutation(len(X_scaled))
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    X_train = torch.tensor(X_scaled[train_idx])
    X_val = torch.tensor(X_scaled[val_idx])

    # Model
    input_dim = X.shape[1]
    model = Autoencoder(input_dim, hidden_dim, latent_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        # Shuffle training data
        perm = torch.randperm(len(X_train))
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i + batch_size]
            batch = X_train[batch_idx]

            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_recon = model(X_val)
            val_loss = criterion(val_recon, X_val).item()
        model.train()

        if verbose and (epoch + 1) % 10 == 0:
            train_loss = epoch_loss / n_batches
            print(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    model.eval()
    return model, scaler


def get_embeddings(
    X: np.ndarray,
    model: nn.Module,
    scaler: StandardScaler,
) -> np.ndarray:
    """Get latent embeddings for window data."""
    X_scaled = scaler.transform(X).astype(np.float32)
    X_tensor = torch.tensor(X_scaled)

    with torch.no_grad():
        embeddings = model.encode(X_tensor).numpy()

    return embeddings


def cluster_phases(
    embeddings: np.ndarray,
    k_range: Tuple[int, int] = (4, 12),
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans, dict]:
    """
    Cluster embeddings to discover phases.

    Args:
        embeddings: Latent embeddings from autoencoder
        k_range: Range of k values to try
        random_state: Random seed for reproducibility

    Returns:
        labels: Cluster labels for each window
        best_kmeans: Fitted KMeans model with best k
        scores: Dict of {k: silhouette_score}
    """
    scores = {}

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores[k] = score
        print(f"k={k}: silhouette={score:.4f}")

    # Select best k
    best_k = max(scores, key=scores.get)
    print(f"\nBest k={best_k} with silhouette={scores[best_k]:.4f}")

    best_kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = best_kmeans.fit_predict(embeddings)

    return labels, best_kmeans, scores


def analyze_clusters(
    meta: pd.DataFrame,
    labels: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    window_size: int = 4,
) -> pd.DataFrame:
    """
    Analyze cluster characteristics.

    Returns DataFrame with cluster statistics.
    """
    meta = meta.copy()
    meta["cluster"] = labels

    # Mean features per cluster
    n_features = len(feature_names)
    cluster_stats = []

    for cluster_id in sorted(meta["cluster"].unique()):
        mask = labels == cluster_id
        cluster_X = X[mask]

        # Average across all windows and bins
        mean_features = cluster_X.reshape(-1, window_size, n_features).mean(axis=(0, 1))

        stats = {"cluster": cluster_id, "n_windows": mask.sum()}
        for i, name in enumerate(feature_names):
            stats[f"mean_{name}"] = mean_features[i]

        # Workload distribution
        cluster_meta = meta[meta["cluster"] == cluster_id]
        workload_counts = cluster_meta["workload"].value_counts()
        dominant_workload = workload_counts.idxmax()
        stats["dominant_workload"] = dominant_workload
        stats["workload_purity"] = workload_counts.max() / len(cluster_meta)

        cluster_stats.append(stats)

    return pd.DataFrame(cluster_stats)


def get_cluster_workload_crosstab(meta: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Get cross-tabulation of clusters vs workloads."""
    meta = meta.copy()
    meta["cluster"] = labels
    return pd.crosstab(meta["cluster"], meta["workload"], margins=True)


def save_model(
    model: nn.Module,
    scaler: StandardScaler,
    kmeans: KMeans,
    output_dir: Path,
):
    """Save model artifacts for later inference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save PyTorch model
    torch.save(model.state_dict(), output_dir / "autoencoder.pt")

    # Save scaler parameters (for stdlib reimplementation)
    np.savez(
        output_dir / "scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
    )

    # Save KMeans centroids
    np.save(output_dir / "kmeans_centroids.npy", kmeans.cluster_centers_)

    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    from load_dataset import load_timelines
    from features import extract_features, get_phase_features, PHASE_FEATURES
    from windows import create_windows

    print("=== Loading Data ===")
    df = load_timelines()

    print("\n=== Extracting Features ===")
    df_feat = extract_features(df)
    df_phase = get_phase_features(df_feat)

    feature_cols = [c for c in PHASE_FEATURES if c in df_phase.columns]

    print("\n=== Creating Windows ===")
    X, meta = create_windows(df_phase, feature_cols=feature_cols)

    print("\n=== Training Autoencoder ===")
    model, scaler = train_autoencoder(X, epochs=50)

    print("\n=== Getting Embeddings ===")
    embeddings = get_embeddings(X, model, scaler)
    print(f"Embedding shape: {embeddings.shape}")

    print("\n=== Clustering ===")
    labels, kmeans, scores = cluster_phases(embeddings)

    print("\n=== Cluster Analysis ===")
    stats = analyze_clusters(meta, labels, X, feature_cols)
    print(stats.to_string())

    print("\n=== Cluster × Workload Cross-tab ===")
    crosstab = get_cluster_workload_crosstab(meta, labels)
    print(crosstab)

    # Save model
    print("\n=== Saving Model ===")
    save_model(model, scaler, kmeans, Path("models/phase_discovery"))
