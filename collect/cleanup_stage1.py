"""
One-shot cleanup for Stage 1 collected data.

The collection harness leaves two big raw artifacts in every ``run_*/``:

  - ``vtune_result/``   -- VTune's per-sample .pb protobufs, 200-500 MB per run.
                          Everything Stage 2 needs was already exported into
                          ``vtune_hw_events.csv``; the raw folder is only useful
                          for opening a run in the VTune GUI for debugging.
  - ``energy/*.etl``    -- SoC Watch raw Event Trace Logs, tens of MB per run.
                          The parsed ``energy.csv`` sitting next to them is
                          what ``energy.parse_output_dir()`` actually reads.

A full sweep produces ~50+ GB of these two things combined. This script walks
every run folder, verifies the PARSED artifacts we depend on are intact, and
(with ``--apply``) drops the raw folders / .etl files.

Safety rails:
  - Dry-run by default. Prints everything it would do; nothing is deleted.
    Re-run with ``--apply`` to actually reclaim disk.
  - Never deletes anything from a run that failed validation. Broken runs
    are only reported so you can decide by hand.
  - Never deletes ``results.json``, ``vtune_hw_events.csv``, ``energy.csv``,
    ``manifest.json``, or ``fingerprint.json`` -- those are the Stage 2 inputs.
  - Writes a ``cleanup_report.json`` at the root of the walked tree so the
    per-run verdicts survive after the console output scrolls away.

Usage:
    # From repo root, dry run over everything under data/raw/:
    python collect/cleanup_stage1.py

    # Only look at one subtree:
    python collect/cleanup_stage1.py --root data/raw/FZA633700090_Ultra9-290HX/stage1_v2

    # Actually delete:
    python collect/cleanup_stage1.py --apply

Standard library only (this may run on the test machine where pip is off).
"""

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


# ---------- helpers: validate the artifacts Stage 2 will actually consume ----

def _size_bytes(path: Path) -> int:
    """Total size of a file or a directory tree, in bytes. Missing == 0."""
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def _validate_results_json(path: Path) -> tuple[bool, str]:
    """Return (ok, reason). ok=True means results.json is present and parseable."""
    if not path.exists():
        return False, "missing results.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return False, f"results.json unreadable: {e}"
    # v1 runs (pre-thread-count-sweep) recorded only workload + core_type;
    # v2 also has thread_count. Both are valid; only the shared keys are
    # required. Their absence means the file was hand-edited or written by
    # a much older harness we don't understand -- treat as broken.
    for k in ("workload", "core_type"):
        if k not in data:
            return False, f"results.json missing '{k}'"
    return True, ""


def _validate_vtune_csv(path: Path) -> tuple[bool, str]:
    """Return (ok, reason). Needs header + at least one data row that parses
    like vtune.parse_csv() would want it to."""
    if not path.exists():
        return False, "missing vtune_hw_events.csv"
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return False, "vtune csv has no header"
            # Cheap check that the header looks like VTune's hw-events report.
            has_thread_col = any(
                (fn or "").strip().lower() == "thread" for fn in reader.fieldnames
            )
            has_counter = any(
                (fn or "").startswith("Hardware Event Count:")
                for fn in reader.fieldnames
            )
            if not (has_thread_col and has_counter):
                return False, "vtune csv header shape unexpected"
            # Need at least one row of data.
            rows_seen = 0
            for _ in reader:
                rows_seen += 1
                if rows_seen >= 1:
                    break
            if rows_seen == 0:
                return False, "vtune csv has no data rows"
    except (OSError, csv.Error) as e:
        return False, f"vtune csv unreadable: {e}"
    return True, ""


def _find_energy_csv(energy_dir: Path) -> Path | None:
    """Match the candidate list in energy.parse_output_dir()."""
    if not energy_dir.is_dir():
        return None
    for name in ("energy.csv", "SoCWatchOutput.csv"):
        p = energy_dir / name
        if p.exists():
            return p
    # Fall back to *any* .csv in the folder -- SoC Watch versions vary.
    for p in energy_dir.glob("*.csv"):
        return p
    return None


def _validate_energy_csv(energy_dir: Path) -> tuple[bool, str, Path | None]:
    """Return (ok, reason, csv_path).

    ok=True means we can find a CSV with a "Package ... Power" row -- what
    energy.parse_output_dir() actually reads. If the run was collected with
    ``--no-energy`` the whole folder is absent, and we count that as OK
    (we don't delete .etl that don't exist).
    """
    if not energy_dir.exists():
        return True, "no energy dir (collected with --no-energy?)", None
    csv_path = _find_energy_csv(energy_dir)
    if csv_path is None:
        return False, "energy dir has no csv", None
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 4 and "Package" in row[0] and "Power" in row[1]:
                    return True, "", csv_path
    except OSError as e:
        return False, f"energy csv unreadable: {e}", csv_path
    return False, "energy csv has no Package/Power row", csv_path


# ---------- per-run analysis ------------------------------------------------

def analyze_run(run_dir: Path) -> dict:
    """Look at one ``run_*/`` folder and decide what (if anything) is safe
    to drop. Returns a verdict dict; no filesystem mutation here."""
    verdict = {
        "path": str(run_dir),
        "status": "unknown",
        "issues": [],
        "deletable": [],       # list of (path, bytes) tuples
        "reclaim_bytes": 0,
    }

    ok_json, reason = _validate_results_json(run_dir / "results.json")
    if not ok_json:
        verdict["status"] = "BROKEN"
        verdict["issues"].append(reason)
        return verdict

    ok_vtune, reason = _validate_vtune_csv(run_dir / "vtune_hw_events.csv")
    if not ok_vtune:
        # Raw vtune_result may still be here -- keep it, VTune can re-export
        # from it if we ever want to. Just report.
        verdict["status"] = "MISSING_VTUNE_CSV"
        verdict["issues"].append(reason)
        return verdict

    ok_energy, energy_reason, _ = _validate_energy_csv(run_dir / "energy")
    if not ok_energy:
        verdict["status"] = "MISSING_ENERGY_CSV"
        verdict["issues"].append(energy_reason)
        # Don't touch .etl in this case -- user may want to re-parse.
        # We *could* still drop vtune_result/, but keeping the whole run
        # untouched until it's diagnosed is safer.
        return verdict

    # Fully valid run -- safe to drop the raw stuff.
    verdict["status"] = "GOOD"

    vtune_raw = run_dir / "vtune_result"
    if vtune_raw.exists():
        size = _size_bytes(vtune_raw)
        verdict["deletable"].append((str(vtune_raw), size, "dir"))
        verdict["reclaim_bytes"] += size

    energy_dir = run_dir / "energy"
    if energy_dir.is_dir():
        for etl in energy_dir.glob("*.etl"):
            size = _size_bytes(etl)
            verdict["deletable"].append((str(etl), size, "file"))
            verdict["reclaim_bytes"] += size

    return verdict


# ---------- driver ----------------------------------------------------------

def walk_runs(root: Path) -> list[Path]:
    """Find every ``run_*/`` directory under ``root``. Recursive so we catch
    both ``data/raw/<fp>/run_*`` (v1 layout, no subdir) and
    ``data/raw/<fp>/stage1_v2/run_*`` (v2 layout, with --subdir)."""
    if not root.exists():
        return []
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.startswith("run_"):
                out.append(Path(dirpath) / d)
    return sorted(out)


def check_manifests(root: Path) -> list[dict]:
    """Sanity-check every parent dir that contains run_*/ folders -- it
    should have manifest.json + fingerprint.json alongside. Missing them
    doesn't stop cleanup, but is worth flagging in the report."""
    issues = []
    seen_parents = set()
    for run in walk_runs(root):
        parent = run.parent
        if parent in seen_parents:
            continue
        seen_parents.add(parent)
        for name in ("manifest.json", "fingerprint.json"):
            if not (parent / name).exists():
                issues.append({
                    "parent": str(parent),
                    "missing": name,
                })
    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Purge raw Stage 1 collection artifacts (vtune_result/, "
                    ".etl) whose parsed outputs are intact."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "raw"),
        help="Directory to walk for run_*/ folders "
             "(default: data/raw/ under the repo root)."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without this flag the script only reports "
             "what it would do."
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Where to write the JSON verdict report. "
             "Default: <root>/cleanup_report.json"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        sys.exit(2)

    print(f"Walking: {root}")
    runs = walk_runs(root)
    print(f"Found {len(runs)} run_*/ folder(s)")
    if not runs:
        print("Nothing to do.")
        return

    manifest_issues = check_manifests(root)
    if manifest_issues:
        print(f"WARNING: {len(manifest_issues)} manifest/fingerprint file(s) missing:")
        for m in manifest_issues[:5]:
            print(f"  - {m['parent']}: missing {m['missing']}")
        if len(manifest_issues) > 5:
            print(f"  ... and {len(manifest_issues) - 5} more (see report)")
        print()

    # Analyze all runs (no deletion yet)
    verdicts = [analyze_run(r) for r in runs]

    # Categorize
    buckets = {"GOOD": [], "BROKEN": [], "MISSING_VTUNE_CSV": [],
               "MISSING_ENERGY_CSV": []}
    for v in verdicts:
        buckets.setdefault(v["status"], []).append(v)

    total_reclaim = sum(v["reclaim_bytes"] for v in verdicts)

    print("=" * 60)
    print("Summary:")
    print(f"  GOOD (safe to clean):      {len(buckets['GOOD']):4d} runs")
    print(f"  MISSING_ENERGY_CSV:        {len(buckets['MISSING_ENERGY_CSV']):4d} runs (kept intact)")
    print(f"  MISSING_VTUNE_CSV:         {len(buckets['MISSING_VTUNE_CSV']):4d} runs (kept intact)")
    print(f"  BROKEN (no results.json):  {len(buckets['BROKEN']):4d} runs (kept intact)")
    print(f"  Reclaimable disk:          {_fmt_bytes(total_reclaim)}")
    print("=" * 60)

    # Detailed listing of problem runs
    for status in ("BROKEN", "MISSING_VTUNE_CSV", "MISSING_ENERGY_CSV"):
        items = buckets[status]
        if not items:
            continue
        print(f"\n{status}:")
        for v in items[:20]:
            issues = "; ".join(v["issues"]) or "(no detail)"
            print(f"  - {v['path']}  --  {issues}")
        if len(items) > 20:
            print(f"  ... and {len(items) - 20} more (see report)")

    # Apply / dry-run
    print()
    if args.apply:
        print(">>> --apply set: deleting now.")
        actually_freed = 0
        delete_errors = []
        for v in buckets["GOOD"]:
            for target, size, kind in v["deletable"]:
                p = Path(target)
                try:
                    if kind == "dir":
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                    actually_freed += size
                except OSError as e:
                    delete_errors.append({"path": target, "error": str(e)})
        print(f"Freed: {_fmt_bytes(actually_freed)} "
              f"({len(delete_errors)} delete error(s))")
        if delete_errors:
            print("First few errors:")
            for de in delete_errors[:5]:
                print(f"  - {de['path']}: {de['error']}")
    else:
        print(">>> DRY RUN. Re-run with --apply to actually delete.")
        actually_freed = 0
        delete_errors = []

    # Write report
    report_path = Path(args.report) if args.report else (root / "cleanup_report.json")
    report = {
        "timestamp": datetime.now().isoformat(),
        "root": str(root),
        "applied": bool(args.apply),
        "totals": {
            "runs": len(verdicts),
            "good": len(buckets["GOOD"]),
            "broken": len(buckets["BROKEN"]),
            "missing_vtune_csv": len(buckets["MISSING_VTUNE_CSV"]),
            "missing_energy_csv": len(buckets["MISSING_ENERGY_CSV"]),
            "reclaimable_bytes": total_reclaim,
            "actually_freed_bytes": actually_freed,
        },
        "manifest_issues": manifest_issues,
        "delete_errors": delete_errors,
        "runs": verdicts,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
