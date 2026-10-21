"""
Dataset catalogue for the progress evaluation: a description of the final Stage 1 dataset.

Reads only results.json files (never modifies data/raw/) from the two final folders:
  stage1_v4      - final collection, 2026-10-02 (its 7-Zip runs all failed: 32-bit binary).
  stage1_v4_7zip - 7-Zip re-collected with the 64-bit binary, 2026-10-04.
The final dataset (used by Stage 2 / Model A / Model B) is the OK runs of both, with 7-Zip taken
from stage1_v4_7zip, matching analysis/load_dataset.py.

Energy caveat: SoC Watch reports one package-level number per run, measured over a window of
max(40 s, expected duration + 15 s) that starts before VTune and includes idle time. Runs are
comparable with each other, but the numbers are not workload-only energy.

Writes to results/presentation/datasets/:
  runs_all.csv            one row per run folder
  dataset_overview.csv/md one row per folder + the final total (the slide table)
  per_config.csv          final set, one row per workload x core x threads (mean over reps)
  per_workload.md         final set, P vs E per workload
  *.png                   coverage, IPC, power and thread-scaling figures

Run:  python analysis/dataset_catalogue.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw" / "FZA633700090_Ultra9-290HX"
OUT = ROOT / "results" / "presentation" / "datasets"

DATASETS = {
    "stage1_v4": RAW / "stage1_v4",
    "stage1_v4_7zip": RAW / "stage1_v4_7zip",
}
DESCRIPTIONS = {
    "stage1_v4": "All 8 workloads (its 7-Zip runs failed: 32-bit exe)",
    "stage1_v4_7zip": "7-Zip re-collected with the 64-bit exe",
}
WORKLOAD_ORDER = ["7zip", "zstd", "x264", "sqlite", "compute", "memory", "mixed", "io"]
REAL_APPS = {"7zip", "zstd", "x264", "sqlite"}
CORE_COLORS = {"p": "#2a78d6", "e": "#eb6834"}
CORE_NAMES = {"p": "P-core", "e": "E-core"}
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _iter_results():
    """Yield (dataset, folder_name, results_dict)."""
    for ds, path in DATASETS.items():
        for f in sorted(path.glob("*/results.json")):
            yield ds, f.parent.name, json.loads(f.read_text(encoding="utf-8"))


def _row(ds: str, folder: str, d: dict) -> dict:
    agg = d.get("aggregate") or {}
    tl = d.get("timeline") or {}
    en = d.get("energy") or {}
    vt = d.get("vtune")
    counters = [k for k in vt if k not in ("thread", "tid")] if isinstance(vt, dict) else []
    status = d.get("status") or ("error" if d.get("vtune_error") else "ok")
    return {
        "dataset": ds,
        "folder": folder,
        "workload": d.get("workload"),
        "core_type": d.get("core_type"),
        "thread_count": d.get("thread_count", 1),
        "rep": d.get("rep"),
        "status": status,
        "timestamp": d.get("timestamp"),
        "requested_duration_sec": d.get("duration_sec"),
        "workload_span_sec": d.get("workload_span_sec"),
        "run_wall_sec": d.get("actual_duration_sec"),
        "ipc": d.get("ipc"),
        "total_instructions": agg.get("total_instructions"),
        "threads_seen": d.get("all_threads", agg.get("total_thread_count")),
        "threads_active": agg.get("active_thread_count"),
        "timeline_bins": tl.get("bins_written"),
        "timeline_counter_columns": tl.get("counter_columns"),
        "vtune_counters": len(counters) or None,
        "avg_power_w": en["avg_power_mw"] / 1000 if en.get("avg_power_mw") else None,
        "energy_j": en["total_energy_mj"] / 1000 if en.get("total_energy_mj") else None,
        "energy_window_sec": en.get("collection_duration_sec"),
        "energy_truncated": d.get("energy_truncated"),
    }


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    rows = [list(df.columns)] + df.astype(str).values.tolist()
    lines = ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    lines.insert(1, "|" + "---|" * len(df.columns))
    return "\n".join(lines)


def _fmt_range(s: pd.Series, fmt="{:.0f}") -> str:
    s = s.dropna()
    if s.empty:
        return "-"
    lo, hi = s.min(), s.max()
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}-{fmt.format(hi)}"


def overview(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in runs.groupby("dataset", sort=False):
        ok = g[g["status"] == "ok"]
        dates = pd.to_datetime(g["timestamp"]).dt.date
        rows.append({
            "dataset": ds,
            "description": DESCRIPTIONS[ds],
            "dates": f"{dates.min()} to {dates.max()}" if dates.min() != dates.max() else str(dates.min()),
            "runs": len(g),
            "ok": len(ok),
            "failed": len(g) - len(ok),
            "workloads": ", ".join(w for w in WORKLOAD_ORDER if w in set(g["workload"])),
            "configs": ok.groupby(["workload", "core_type", "thread_count"]).ngroups,
            "thread_counts": ", ".join(str(t) for t in sorted(g["thread_count"].unique())),
            "reps_per_config": _fmt_range(ok.groupby(["workload", "core_type", "thread_count"]).size()),
            "workload_span_sec": _fmt_range(ok["workload_span_sec"], "{:.1f}"),
            "run_wall_min_total": round(g["run_wall_sec"].sum() / 60, 1),
            "ipc_range": _fmt_range(ok["ipc"], "{:.2f}"),
            "vtune_counters": _fmt_range(ok["vtune_counters"]),
            "timeline_bins_total": int(ok["timeline_bins"].sum()) if ok["timeline_bins"].notna().any() else 0,
            "avg_power_w_mean": round(ok["avg_power_w"].mean(), 2),
            "energy_j_mean": round(ok["energy_j"].mean(), 1),
            "energy_truncated_runs": int((g["energy_truncated"] == True).sum()),  # noqa: E712
        })
    return pd.DataFrame(rows)


def final_set(runs: pd.DataFrame) -> pd.DataFrame:
    """The final dataset: OK runs, 7-Zip from stage1_v4_7zip, everything else from stage1_v4."""
    ok = runs[runs["status"] == "ok"]
    keep = (ok["workload"] == "7zip") == (ok["dataset"] == "stage1_v4_7zip")
    return ok[keep].copy()


def per_config(fs: pd.DataFrame) -> pd.DataFrame:
    g = fs.groupby(["workload", "core_type", "thread_count"])
    out = g.agg(reps=("folder", "size"),
                ipc_mean=("ipc", "mean"), ipc_std=("ipc", "std"),
                workload_span_sec=("workload_span_sec", "mean"),
                threads_active=("threads_active", "mean"), threads_seen=("threads_seen", "mean"),
                timeline_bins=("timeline_bins", "mean"),
                avg_power_w=("avg_power_w", "mean"), energy_j=("energy_j", "mean"),
                energy_window_sec=("energy_window_sec", "mean")).reset_index()
    out["workload"] = pd.Categorical(out["workload"], WORKLOAD_ORDER, ordered=True)
    return out.sort_values(["workload", "core_type", "thread_count"]).round(3)


def per_workload_md(fs: pd.DataFrame) -> str:
    lines = ["# Final Stage 1 dataset (stage1_v4 + stage1_v4_7zip), per workload", "",
             "Mean over all OK runs of that workload on that core type (all thread counts).",
             "P/E IPC ratio > 1 means the workload retires more instructions per cycle on P-cores.",
             "Power/energy: package-level, measured over a window that includes idle time "
             "(comparable between runs, not workload-only).", "",
             "| Workload | Kind | Runs (P/E) | Threads tested (P / E) | Span (s) | IPC P | IPC E | P/E IPC | "
             "Active threads (mean) | Timeline bins / run | Power P (W) | Power E (W) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in WORKLOAD_ORDER:
        g = fs[fs["workload"] == w]
        if g.empty:
            continue
        p, e = g[g["core_type"] == "p"], g[g["core_type"] == "e"]
        tc = lambda x: ",".join(str(t) for t in sorted(x["thread_count"].unique()))
        lines.append(
            f"| {w} | {'real app' if w in REAL_APPS else 'Python micro'} | {len(p)}/{len(e)} | "
            f"{tc(p)} / {tc(e)} | {g['workload_span_sec'].mean():.1f} | {p['ipc'].mean():.2f} | "
            f"{e['ipc'].mean():.2f} | {p['ipc'].mean() / e['ipc'].mean():.2f} | "
            f"{g['threads_active'].mean():.1f} | {g['timeline_bins'].mean():.0f} | "
            f"{p['avg_power_w'].mean():.1f} | {e['avg_power_w'].mean():.1f} |")
    lines += ["", f"Total: {len(fs)} runs, {int(fs['timeline_bins'].sum()):,} timeline bins "
              f"(0.5 s each, per thread rows not counted), "
              f"{fs['total_instructions'].sum() / 1e12:.1f} trillion instructions sampled."]
    return "\n".join(lines) + "\n"


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d0cfcb")
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_coverage(fs: pd.DataFrame, path: Path):
    """Runs per workload x (core, threads) - shows the experiment design at a glance."""
    cols = [(c, t) for c in ("p", "e") for t in sorted(fs[fs["core_type"] == c]["thread_count"].unique())]
    mat = np.array([[((fs["workload"] == w) & (fs["core_type"] == c) & (fs["thread_count"] == t)).sum()
                     for c, t in cols] for w in WORKLOAD_ORDER])
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.imshow(np.where(mat > 0, mat, np.nan), cmap="Blues", vmin=0, vmax=mat.max() * 1.6, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]) if mat[i, j] else "-", ha="center", va="center",
                    color=INK if mat[i, j] else INK_MUTED, fontsize=9)
    ax.set_xticks(range(len(cols)), [f"{CORE_NAMES[c]}\n{t} thr" for c, t in cols], fontsize=8)
    ax.set_yticks(range(len(WORKLOAD_ORDER)), WORKLOAD_ORDER)
    ax.set_title("Final dataset coverage: OK runs per configuration (stage1_v4 + stage1_v4_7zip)",
                 loc="left", color=INK, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pe_bars(fs: pd.DataFrame, col: str, title: str, ylabel: str, path: Path, single_thread=True):
    data = fs[fs["thread_count"] == 1] if single_thread else fs
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    x = np.arange(len(WORKLOAD_ORDER))
    w = 0.38
    for k, c in enumerate(("p", "e")):
        g = data[data["core_type"] == c].groupby("workload")[col]
        m = g.mean().reindex(WORKLOAD_ORDER)
        sd = g.std().reindex(WORKLOAD_ORDER)
        ax.bar(x + (k - 0.5) * w, m, w * 0.94, yerr=sd, color=CORE_COLORS[c], label=CORE_NAMES[c],
               error_kw={"ecolor": INK_MUTED, "elinewidth": 0.8, "capsize": 2})
    ax.set_xticks(x, WORKLOAD_ORDER)
    _style(ax, title, ylabel=ylabel)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scaling(fs: pd.DataFrame, path: Path):
    """IPC vs thread count for the four real applications, P vs E."""
    apps = [w for w in WORKLOAD_ORDER if w in REAL_APPS]
    fig, axes = plt.subplots(1, len(apps), figsize=(11, 3.2), facecolor=SURFACE, sharey=True)
    for ax, w in zip(axes, apps):
        for c in ("p", "e"):
            g = fs[(fs["workload"] == w) & (fs["core_type"] == c)].groupby("thread_count")["ipc"]
            m, sd = g.mean(), g.std()
            ax.errorbar(m.index, m.values, yerr=sd.values, color=CORE_COLORS[c], marker="o",
                        markersize=5, linewidth=2, capsize=2, label=CORE_NAMES[c])
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16], ["1", "4", "8", "16"])
        _style(ax, w, xlabel="threads")
    axes[0].set_ylabel("aggregate IPC", color=INK_MUTED)
    axes[0].legend(frameon=False, labelcolor=INK)
    fig.suptitle("IPC vs thread count (mean ± std over reps)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame([_row(ds, f, d) for ds, f, d in _iter_results()])
    runs.to_csv(OUT / "runs_all.csv", index=False)

    fs = final_set(runs)
    ov = overview(runs)
    total = {"dataset": "**final dataset**", "description": "OK runs of both, 7-Zip from the retry",
             "runs": len(fs), "ok": len(fs), "failed": 0,
             "timeline_bins_total": int(fs["timeline_bins"].sum())}
    ov = pd.concat([ov, pd.DataFrame([total])], ignore_index=True)
    ov = ov.astype({"configs": "Int64", "energy_truncated_runs": "Int64"}).astype(object).fillna("")
    ov.to_csv(OUT / "dataset_overview.csv", index=False)
    md = ["# Stage 1 final dataset", "",
          "Machine: Intel Core Ultra 9 290HX (8 P-cores, 16 E-cores), 32 GB DDR5-6400, Windows 11 "
          "(build 26200). Counters from Intel VTune (uarch-exploration); energy from Intel SoC Watch.", "",
          md_table(ov), "",
          "Notes:",
          "- `ok`/`failed` come from each run's `status` field. Counters are recorded every 0.5 s "
          "(the per-run timeline used by Stage 2).",
          "- Energy is one package-level number per run over a window that includes idle time "
          "(at least 40 s), so compare runs with each other, not with workload-only energy.",
          "- stage1_v4 failures: 35 of 35 first-pass 7-Zip runs (32-bit `7z.exe`, which VTune "
          "refuses) plus 4 others; 7-Zip was re-collected in stage1_v4_7zip. 7-Zip's high "
          "active-thread count is real: its benchmark spawns many short-lived worker threads.", ""]
    (OUT / "dataset_overview.md").write_text("\n".join(md), encoding="utf-8")

    per_config(fs).to_csv(OUT / "per_config.csv", index=False)
    (OUT / "per_workload.md").write_text(per_workload_md(fs), encoding="utf-8")

    plot_coverage(fs, OUT / "coverage.png")
    plot_pe_bars(fs, "ipc", "IPC by workload, single thread: P-core vs E-core", "IPC", OUT / "ipc_p_vs_e.png")
    plot_pe_bars(fs, "avg_power_w", "Average package power, single thread: P-core vs E-core",
                 "watts (window incl. idle)", OUT / "power_p_vs_e.png")
    plot_scaling(fs, OUT / "ipc_scaling.png")

    print(ov.to_string(index=False))
    print(f"\nFinal set: {len(fs)} OK runs. Outputs in {OUT}")


if __name__ == "__main__":
    main()
