"""
Visualize Stage 2 phase discovery results.

Run: python analysis/visualize_phases.py

Generates:
- Cluster statistics table
- Cluster × workload heatmap
- Feature profiles per cluster (radar/bar charts)
- 2D UMAP projection colored by cluster and workload
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Optional: UMAP for dimensionality reduction visualization
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Note: Install umap-learn for 2D projection visualization")

import torch
from sklearn.cluster import KMeans

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.load_dataset import load_timelines
from analysis.features import extract_features, get_phase_features, PHASE_FEATURES
from analysis.windows import create_windows
from analysis.model_a import Autoencoder, get_embeddings, analyze_clusters, get_cluster_workload_crosstab


def load_model_and_data():
    """Load saved model and recreate embeddings."""
    model_dir = Path(__file__).parent.parent / "models" / "phase_discovery"

    # Load scaler
    scaler_data = np.load(model_dir / "scaler.npz")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    # Load KMeans centroids
    centroids = np.load(model_dir / "kmeans_centroids.npy")
    n_clusters = len(centroids)

    # Load autoencoder
    input_dim = len(scaler.mean_)
    model = Autoencoder(input_dim, hidden_dim=16, latent_dim=8)
    model.load_state_dict(torch.load(model_dir / "autoencoder.pt", weights_only=True))
    model.eval()

    # Load and process data
    print("Loading data...")
    df = load_timelines()
    df_feat = extract_features(df)
    df_phase = get_phase_features(df_feat)

    feature_cols = [c for c in PHASE_FEATURES if c in df_phase.columns]
    X, meta = create_windows(df_phase, feature_cols=feature_cols)

    # Get embeddings
    embeddings = get_embeddings(X, model, scaler)

    # Assign clusters using saved centroids (manual nearest-centroid assignment)
    # This avoids needing to fit KMeans which would require setting internal state
    from scipy.spatial.distance import cdist
    distances = cdist(embeddings, centroids, metric='euclidean')
    labels = np.argmin(distances, axis=1)

    return X, meta, embeddings, labels, feature_cols, n_clusters


def plot_cluster_sizes(meta, labels, output_dir):
    """Bar chart of cluster sizes."""
    meta = meta.copy()
    meta["cluster"] = labels

    cluster_sizes = meta.groupby("cluster").size().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_sizes)))
    bars = ax.bar(range(len(cluster_sizes)), cluster_sizes.values, color=colors)
    ax.set_xticks(range(len(cluster_sizes)))
    ax.set_xticklabels([f"Cluster {i}" for i in cluster_sizes.index])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Windows")
    ax.set_title("Phase Cluster Sizes")

    # Add value labels
    for bar, val in zip(bars, cluster_sizes.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{val:,}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "cluster_sizes.png", dpi=150)
    plt.close()
    print(f"Saved: cluster_sizes.png")


def plot_workload_heatmap(meta, labels, output_dir):
    """Heatmap of cluster × workload distribution."""
    meta = meta.copy()
    meta["cluster"] = labels

    crosstab = pd.crosstab(meta["cluster"], meta["workload"], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(crosstab, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "% of cluster"})
    ax.set_title("Workload Distribution per Cluster (%)")
    ax.set_xlabel("Workload")
    ax.set_ylabel("Cluster")

    plt.tight_layout()
    plt.savefig(output_dir / "workload_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved: workload_heatmap.png")


def plot_feature_profiles(X, labels, feature_cols, window_size, output_dir):
    """Bar chart of mean features per cluster."""
    n_features = len(feature_cols)
    n_clusters = len(np.unique(labels))

    # Compute mean features per cluster
    cluster_means = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_X = X[mask]
        # Average across windows and bins
        mean_features = cluster_X.reshape(-1, window_size, n_features).mean(axis=(0, 1))
        cluster_means.append(mean_features)

    cluster_means = np.array(cluster_means)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(feature_cols))
    width = 0.8 / n_clusters
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        offset = (i - n_clusters/2 + 0.5) * width
        ax.bar(x + offset, cluster_means[i], width, label=f"Cluster {i}", color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(feature_cols, rotation=45, ha='right')
    ax.set_ylabel("Mean Value")
    ax.set_title("Feature Profiles per Cluster")
    ax.legend(loc='upper right', ncol=2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_profiles.png", dpi=150)
    plt.close()
    print(f"Saved: feature_profiles.png")


def plot_umap_projection(embeddings, labels, meta, output_dir):
    """2D UMAP projection colored by cluster and workload."""
    if not HAS_UMAP:
        print("Skipping UMAP (not installed)")
        return

    print("Computing UMAP projection (this may take a minute)...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    proj = reducer.fit_transform(embeddings)

    # Plot by cluster
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: colored by cluster
    n_clusters = len(np.unique(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
    for i in range(n_clusters):
        mask = labels == i
        axes[0].scatter(proj[mask, 0], proj[mask, 1], c=[colors[i]],
                       label=f"Cluster {i}", alpha=0.5, s=10)
    axes[0].set_title("UMAP Projection by Cluster")
    axes[0].legend(markerscale=3)
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")

    # Right: colored by workload
    workloads = meta["workload"].unique()
    workload_colors = plt.cm.tab10(np.linspace(0, 1, len(workloads)))
    for i, wl in enumerate(workloads):
        mask = meta["workload"].values == wl
        axes[1].scatter(proj[mask, 0], proj[mask, 1], c=[workload_colors[i]],
                       label=wl, alpha=0.5, s=10)
    axes[1].set_title("UMAP Projection by Workload")
    axes[1].legend(markerscale=3)
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(output_dir / "umap_projection.png", dpi=150)
    plt.close()
    print(f"Saved: umap_projection.png")


def plot_cluster_summary(X, meta, labels, feature_cols, window_size, output_dir):
    """Summary table as an image."""
    stats = analyze_clusters(meta, labels, X, feature_cols, window_size)

    # Create a nice formatted table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Select key columns for display
    display_cols = ["cluster", "n_windows", "dominant_workload", "workload_purity"]
    feature_display = ["mean_ipc", "mean_retiring", "mean_be_bound", "mean_fe_bound",
                       "mean_bad_spec", "mean_mem_stall_frac", "mean_dram_stall_frac"]

    # Format the data
    table_data = []
    for _, row in stats.iterrows():
        table_row = [
            f"{int(row['cluster'])}",
            f"{int(row['n_windows']):,}",
            row['dominant_workload'],
            f"{row['workload_purity']:.0%}",
        ]
        for col in feature_display:
            if col in row:
                table_row.append(f"{row[col]:.2f}")
        table_data.append(table_row)

    headers = ["Cluster", "Windows", "Dominant", "Purity",
               "IPC", "Retiring", "BE Bound", "FE Bound",
               "Bad Spec", "L1 Miss", "L3 Miss"]

    table = ax.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title("Phase Cluster Summary", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "cluster_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: cluster_summary.png")

    return stats


def main():
    output_dir = Path(__file__).parent.parent / "results" / "stage2_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 2 Phase Discovery Visualization")
    print("=" * 60)

    # Load everything
    X, meta, embeddings, labels, feature_cols, n_clusters = load_model_and_data()
    window_size = 4

    print(f"\nDataset: {len(X):,} windows, {n_clusters} clusters")
    print(f"Features: {feature_cols}")
    print(f"Output dir: {output_dir}")
    print()

    # Generate all plots
    print("Generating visualizations...")
    print("-" * 40)

    stats = plot_cluster_summary(X, meta, labels, feature_cols, window_size, output_dir)
    plot_cluster_sizes(meta, labels, output_dir)
    plot_workload_heatmap(meta, labels, output_dir)
    plot_feature_profiles(X, labels, feature_cols, window_size, output_dir)
    plot_umap_projection(embeddings, labels, meta, output_dir)

    # Print text summary
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)
    print(stats.to_string(index=False))

    print("\n" + "=" * 60)
    print("WORKLOAD × CLUSTER CROSS-TAB")
    print("=" * 60)
    crosstab = get_cluster_workload_crosstab(meta, labels)
    print(crosstab)

    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nFiles:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
