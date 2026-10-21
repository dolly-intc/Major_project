"""
Phase 0 environment probes -- run these on the TEST machine before the final
Stage 1 campaign.

Every probe answers a question the harness fixes depend on, and each one is
something we currently have no evidence for. The campaign is ~9-12 hours and
must be the last one, so the cost of guessing wrong here is another two weeks.

  0.1  Can we run several `vtune -report` slice calls concurrently?
       The reporter is single-threaded and clock-bound, so ~85% of the
       campaign's wall time is serial slicing. If N=4 works this is ~11h -> ~6h.
       PASS = identical CSV content to serial, and speedup >= 2.0x at some N.

  0.2  Does VTune attribute a GRANDCHILD process?
       Every timeline CSV collected so far contains exactly one PID, so there
       is no evidence either way. The planned x264 stdin feeder makes x264.exe
       a grandchild of VTune (vtune -> python feeder -> x264). If VTune does
       not follow it, the feeder silently profiles the feeder and 35 runs are
       wasted. PASS = the report contains the grandchild's PID.

  0.3  What is the real workload span at --duration 30, per workload and
       thread count? Sets max_bins, validates the time budget, and catches the
       `zstd -i30` blowup (-i is a MINIMUM evaluation time and -b3 runs both
       compress and decompress, so the span may be 37-67s rather than 30s).
       PASS = every span <= --span-budget (default 45s).

  0.4  x264 span at slower presets. Only needed if 0.2 FAILS -- it picks the
       preset for the fallback one-line fix.

  0.5  Free disk vs projected campaign footprint.

  0.6  Does `socwatch -p "<cmd>"` bracket the child's lifetime, and does `-n`
       change the sample interval? This designs the clean-energy pass. Also
       reports whether the CSV grows a time-series section ABOVE the summary,
       which would break energy.parse_csv's first-match-wins scan.

Standard library only (pip does not work on the test machine).

Usage:
    # Everything (~20-25 min, mostly probe 0.3):
    python collect/probe_env.py

    # Just the two that gate the code changes:
    python collect/probe_env.py --only 0.1 0.2

    # Reuse an existing finalized VTune result for 0.1 instead of collecting:
    python collect/probe_env.py --only 0.1 --result-dir path/to/vtune_result

Writes probe_report.json next to the console output so the verdicts survive.
"""

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import core_topology  # noqa: E402
import energy  # noqa: E402
import vtune  # noqa: E402
from runner import _build_workload_command  # noqa: E402
from workloads import WORKLOADS  # noqa: E402

COLLECT_DIR = Path(__file__).parent

# Probe 0.1 tries these worker counts. 1 is the serial baseline.
PARALLEL_LEVELS = [1, 2, 4, 6]

# Probe 0.1 slices this many bins at each level. Enough to average out noise
# without spending minutes -- at ~2.1 s/slice serial that is ~25 s for N=1.
PARALLEL_BINS = 12


# ---------- small helpers ---------------------------------------------------

def _hr(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _free_gb(path):
    """Free space in GB on the volume holding `path`."""
    return shutil.disk_usage(str(path)).free / (1024 ** 3)


def _slice_cmd(result_dir, t0, t1, out_csv, group_by="thread"):
    return [
        vtune.VTUNE_PATH,
        "-report", "hw-events",
        "-r", str(result_dir),
        "-group-by", group_by,
        "-time-filter", f"{t0}:{t1}",
        "-format=csv",
        "-csv-delimiter=comma",
        "-report-output", str(out_csv),
    ]


def _run_slice(result_dir, i, bin_size, scratch, tag):
    """One slice call. Returns (index, returncode, bytes_written, path)."""
    t0 = round(i * bin_size, 3)
    t1 = round(t0 + bin_size, 3)
    out = Path(scratch) / f"{tag}_{i:04d}.csv"
    try:
        p = subprocess.run(
            _slice_cmd(result_dir, t0, t1, out),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=120,
        )
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    size = out.stat().st_size if out.exists() else 0
    return i, rc, size, str(out)


def _collect_reference_result(dest_dir, duration=12):
    """Collect one short uarch run so the slicing probes have a real,
    finalized result directory to work against."""
    result_dir = Path(dest_dir) / "probe_ref_result"
    if result_dir.exists():
        shutil.rmtree(result_dir, ignore_errors=True)
    cmd = [
        sys.executable, "workloads.py", "compute",
        "--duration", str(duration), "--quiet",
    ]
    print(f"  collecting reference result ({duration}s compute)...")
    t0 = time.perf_counter()
    vtune.collect(cmd, str(result_dir), duration_sec=duration + 5,
                  cwd=str(COLLECT_DIR))
    wall = time.perf_counter() - t0
    print(f"  done in {wall:.1f}s -> {result_dir}")
    return result_dir, wall


# ---------- probe 0.1: parallel slicing -------------------------------------

def probe_parallel_slicing(result_dir, levels=PARALLEL_LEVELS,
                           n_bins=PARALLEL_BINS, bin_size=0.5):
    """Slice the same bins serially and concurrently; compare wall time and
    verify the concurrent output is byte-identical to the serial output.

    Byte-identity matters more than speed here: VTune's reporter may use a
    shared lock file or scratch area inside the result dir, in which case
    concurrent calls could interleave and silently corrupt each other.
    """
    out = {"levels": {}, "verdict": "FAIL", "notes": []}
    baseline_sizes = None

    with tempfile.TemporaryDirectory(prefix="probe_par_") as scratch:
        for n in levels:
            tag = f"n{n}"
            t0 = time.perf_counter()
            if n == 1:
                results = [_run_slice(result_dir, i, bin_size, scratch, tag)
                           for i in range(n_bins)]
            else:
                with ThreadPoolExecutor(max_workers=n) as pool:
                    futs = [pool.submit(_run_slice, result_dir, i, bin_size,
                                        scratch, tag)
                            for i in range(n_bins)]
                    results = [f.result() for f in futs]
            wall = time.perf_counter() - t0

            results.sort()
            rcs = [r[1] for r in results]
            sizes = [r[2] for r in results]
            bad = [r for r in results if r[1] != 0]

            entry = {
                "workers": n,
                "wall_sec": round(wall, 2),
                "sec_per_slice": round(wall / n_bins, 3),
                "nonzero_exits": len(bad),
                "exit_codes": sorted(set(str(c) for c in rcs)),
                "total_bytes": sum(sizes),
                "empty_outputs": sum(1 for s in sizes if s == 0),
            }

            if n == 1:
                baseline_sizes = sizes
                entry["matches_serial"] = True
                entry["speedup"] = 1.0
            else:
                entry["matches_serial"] = (sizes == baseline_sizes)
                base = out["levels"]["1"]["wall_sec"]
                entry["speedup"] = round(base / wall, 2) if wall else None

            out["levels"][str(n)] = entry
            print(f"  workers={n:<2} wall={entry['wall_sec']:>6.2f}s  "
                  f"{entry['sec_per_slice']:.2f}s/slice  "
                  f"speedup={entry['speedup']}  "
                  f"errors={entry['nonzero_exits']}  "
                  f"identical={entry['matches_serial']}")

    # Pick the best level that was both correct and faster.
    safe = [e for k, e in out["levels"].items()
            if k != "1" and e["nonzero_exits"] == 0 and e["matches_serial"]]
    if not safe:
        out["notes"].append(
            "No concurrency level produced correct output. Keep max_workers=1; "
            "the campaign stays on the ~11h serial estimate."
        )
        out["recommended_max_workers"] = 1
    else:
        best = max(safe, key=lambda e: e["speedup"] or 0)
        out["recommended_max_workers"] = best["workers"]
        if (best["speedup"] or 0) >= 2.0:
            out["verdict"] = "PASS"
            out["notes"].append(
                f"Use max_workers={best['workers']} "
                f"({best['speedup']}x, {best['sec_per_slice']}s/slice)."
            )
        else:
            out["verdict"] = "WEAK"
            out["notes"].append(
                f"Best speedup only {best['speedup']}x at "
                f"{best['workers']} workers -- not worth the added failure "
                f"modes. Keep max_workers=1."
            )
            out["recommended_max_workers"] = 1

    # A level that ran clean but produced DIFFERENT bytes is the dangerous
    # outcome: it would have silently corrupted the dataset.
    corrupt = [e for k, e in out["levels"].items()
               if k != "1" and e["nonzero_exits"] == 0
               and not e["matches_serial"]]
    if corrupt:
        out["notes"].append(
            "WARNING: at " +
            ", ".join(str(e["workers"]) for e in corrupt) +
            " workers VTune exited 0 but wrote different bytes than serial. "
            "Concurrent slicing is NOT safe on this machine."
        )
        out["verdict"] = "FAIL"
        out["recommended_max_workers"] = 1

    return out


# ---------- probe 0.2: grandchild attribution -------------------------------

def probe_grandchild(dest_dir, duration=10):
    """Profile `python -c "subprocess.run([7z.exe, b])"` and count PIDs.

    VTune is told to profile the python launcher; 7z.exe is its child, so from
    VTune's point of view it is a grandchild of the vtune.exe process. If the
    resulting report mentions only the launcher's PID, then process-tree
    following is off and the x264 stdin-feeder design cannot work.
    """
    out = {"verdict": "FAIL", "notes": []}

    sevenzip = COLLECT_DIR / "workloads_bin" / "7z.exe"
    if not sevenzip.exists():
        out["notes"].append(f"missing {sevenzip}; cannot run this probe")
        out["verdict"] = "SKIP"
        return out

    result_dir = Path(dest_dir) / "probe_grandchild_result"
    if result_dir.exists():
        shutil.rmtree(result_dir, ignore_errors=True)

    # The launcher prints both PIDs so we can tell them apart in the report.
    launcher = (
        "import os,subprocess,sys;"
        "print('LAUNCHER_PID',os.getpid(),flush=True);"
        f"p=subprocess.Popen([r'{sevenzip}','b','-mmt2']);"
        "print('CHILD_PID',p.pid,flush=True);"
        "p.wait()"
    )
    cmd = [sys.executable, "-c", launcher]

    print("  collecting (python launcher -> 7z.exe child)...")
    proc = subprocess.run(
        [vtune.VTUNE_PATH, "-collect", "uarch-exploration",
         "-knob", "enable-stack-collection=false",
         "-r", str(result_dir), "-duration", str(duration), "--"] + cmd,
        capture_output=True, text=True, cwd=str(COLLECT_DIR),
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    launcher_pid = child_pid = None
    for line in combined.splitlines():
        if line.startswith("LAUNCHER_PID"):
            launcher_pid = line.split()[1].strip()
        elif line.startswith("CHILD_PID"):
            child_pid = line.split()[1].strip()
    out["launcher_pid"] = launcher_pid
    out["child_pid"] = child_pid
    out["collect_returncode"] = proc.returncode

    if not result_dir.exists():
        out["notes"].append("collection produced no result dir; see stderr")
        out["collect_stderr_tail"] = combined[-800:]
        return out

    # Group by process so PIDs are unambiguous in the report.
    report_csv = Path(dest_dir) / "probe_grandchild_report.csv"
    rep = subprocess.run(
        [vtune.VTUNE_PATH, "-report", "hw-events", "-r", str(result_dir),
         "-group-by", "process", "-format=csv", "-csv-delimiter=comma",
         "-report-output", str(report_csv)],
        capture_output=True, text=True,
    )
    out["report_returncode"] = rep.returncode
    if not report_csv.exists():
        out["notes"].append("report produced no CSV")
        out["report_stderr_tail"] = ((rep.stdout or "") + (rep.stderr or ""))[-800:]
        return out

    text = report_csv.read_text(encoding="utf-8", errors="replace")
    processes = []
    for line in text.splitlines()[1:]:
        if line.strip():
            processes.append(line.split(",")[0].strip())
    out["processes_in_report"] = processes[:40]
    out["n_processes"] = len(processes)

    saw_7z = any("7z" in p.lower() for p in processes)
    saw_child_pid = bool(child_pid) and any(child_pid in p for p in processes)
    out["grandchild_attributed"] = bool(saw_7z or saw_child_pid)

    if out["grandchild_attributed"]:
        out["verdict"] = "PASS"
        out["notes"].append(
            "VTune follows the process tree -- the x264 stdin feeder is viable."
        )
    else:
        out["notes"].append(
            "Only the launcher appears in the report. The x264 feeder would "
            "profile the feeder, not the encoder. Use the preset fallback "
            "(probe 0.4) instead."
        )
    return out


# ---------- probe 0.3: real workload spans ----------------------------------

def probe_spans(duration=30.0, workloads=None, span_budget=45.0):
    """Time every workload WITHOUT VTune, at its fastest and slowest thread
    count. This is the ground truth for max_bins and the time budget."""
    out = {"duration_requested": duration, "runs": [], "verdict": "PASS",
           "notes": []}
    names = workloads or list(WORKLOADS.keys())

    for name in names:
        spec = WORKLOADS[name]
        tcs = spec.get("thread_counts", [1])
        probe_tcs = sorted({tcs[0], tcs[-1]})
        for tc in probe_tcs:
            try:
                cmd, _expected, missing = _build_workload_command(
                    name, tc, duration
                )
            except Exception as e:
                out["runs"].append({"workload": name, "thread_count": tc,
                                    "error": f"build failed: {e}"})
                continue
            if missing:
                out["runs"].append({"workload": name, "thread_count": tc,
                                    "error": "missing: " + ", ".join(missing)})
                continue

            t0 = time.perf_counter()
            try:
                p = subprocess.run(cmd, cwd=str(COLLECT_DIR),
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   timeout=max(300.0, duration * 8))
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            span = time.perf_counter() - t0

            entry = {
                "workload": name,
                "thread_count": tc,
                "span_sec": round(span, 2),
                "returncode": rc,
                "bins_at_0.5s": int(span / 0.5) + 1,
                "over_budget": span > span_budget,
            }
            out["runs"].append(entry)
            flag = "  <-- OVER BUDGET" if entry["over_budget"] else ""
            print(f"  {name:<8} t{tc:<3} span={span:>6.1f}s  "
                  f"bins={entry['bins_at_0.5s']:>3}  rc={rc}{flag}")

    spans = [r["span_sec"] for r in out["runs"] if "span_sec" in r]
    if spans:
        out["max_span_sec"] = max(spans)
        # +8 bins of headroom so a slow run is never clipped by max_bins.
        out["recommended_max_bins"] = int(max(spans) / 0.5) + 8
    over = [r for r in out["runs"] if r.get("over_budget")]
    if over:
        out["verdict"] = "WARN"
        for r in over:
            out["notes"].append(
                f"{r['workload']} t{r['thread_count']} ran "
                f"{r['span_sec']}s (> {span_budget}s budget) -- clamp it or "
                f"the campaign overruns."
            )
    errs = [r for r in out["runs"] if "error" in r]
    if errs:
        out["verdict"] = "FAIL"
        for r in errs:
            out["notes"].append(f"{r['workload']} t{r['thread_count']}: "
                                f"{r['error']}")
    return out


# ---------- probe 0.4: x264 presets -----------------------------------------

def probe_x264_presets(presets=("medium", "slow", "veryslow", "placebo"),
                       thread_counts=(1, 16), target=30.0):
    """Only relevant if 0.2 fails. Finds the preset whose span is closest to
    `target` at the HIGHEST thread count (the fastest case)."""
    out = {"runs": [], "verdict": "PASS", "notes": []}
    exe = COLLECT_DIR / "workloads_bin" / "x264.exe"
    clip = COLLECT_DIR / "workloads_data" / "sample.y4m"
    if not exe.exists() or not clip.exists():
        out["verdict"] = "SKIP"
        out["notes"].append("x264.exe or sample.y4m missing")
        return out

    for preset in presets:
        for tc in thread_counts:
            cmd = [str(exe), "--threads", str(tc), "--preset", preset,
                   "-o", "NUL", str(clip)]
            t0 = time.perf_counter()
            try:
                p = subprocess.run(cmd, cwd=str(COLLECT_DIR),
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=600)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            span = time.perf_counter() - t0
            out["runs"].append({"preset": preset, "thread_count": tc,
                                "span_sec": round(span, 2), "returncode": rc})
            print(f"  preset={preset:<9} t{tc:<3} span={span:>6.1f}s rc={rc}")

    fastest_tc = max(thread_counts)
    cands = [r for r in out["runs"]
             if r["thread_count"] == fastest_tc and r["returncode"] == 0]
    if cands:
        best = min(cands, key=lambda r: abs(r["span_sec"] - target))
        out["recommended_preset"] = best["preset"]
        out["notes"].append(
            f"At t{fastest_tc}, --preset {best['preset']} gives "
            f"{best['span_sec']}s (target {target}s)."
        )
    return out


# ---------- probe 0.5: disk -------------------------------------------------

def probe_disk(output_root, n_runs=180, mb_per_run=40, energy_runs=180,
               mb_per_energy_run=75):
    """Free space vs projected footprint.

    mb_per_run is CSVs only (the campaign runs --no-energy and deletes
    vtune_result/). mb_per_energy_run is the clean-energy pass's .etl, which
    the harness now deletes per run -- so the peak is one run's worth, but we
    budget for a handful in case a delete fails.
    """
    free = _free_gb(output_root)
    campaign_gb = (n_runs * mb_per_run) / 1024
    energy_peak_gb = (5 * mb_per_energy_run) / 1024
    needed = campaign_gb + energy_peak_gb
    out = {
        "path": str(output_root),
        "free_gb": round(free, 1),
        "projected_campaign_gb": round(campaign_gb, 1),
        "energy_peak_gb": round(energy_peak_gb, 1),
        "needed_gb": round(needed, 1),
        "headroom_gb": round(free - needed, 1),
        "verdict": "PASS" if free > needed * 2 else "WARN",
        "notes": [],
    }
    if free <= needed * 2:
        out["notes"].append(
            f"Only {free:.1f} GB free against a {needed:.1f} GB need. "
            f"Free space or point --output at another volume."
        )
    print(f"  {output_root}: {free:.1f} GB free, need ~{needed:.1f} GB "
          f"-> {out['verdict']}")
    return out


# ---------- probe 0.6: socwatch -p / -n -------------------------------------

def probe_socwatch(dest_dir, child_sec=8, window_sec=25):
    """Does `-p "<cmd>"` stop collection when the child exits, and does `-n`
    change the sample interval?

    If `-p` brackets the child, the clean-energy pass can measure exactly the
    workload with no idle padding -- which is the whole point, since the
    in-VTune energy numbers were ~84% overhead.
    """
    out = {"verdict": "FAIL", "notes": []}
    if not os.path.exists(energy.SOCWATCH_PATH):
        out["verdict"] = "SKIP"
        out["notes"].append(f"SoC Watch not found at {energy.SOCWATCH_PATH}")
        return out

    dest = Path(dest_dir) / "probe_socwatch"
    dest.mkdir(parents=True, exist_ok=True)

    # A child that runs for a known, short time -- much shorter than -t.
    child = (f'"{sys.executable}" -c "import time;time.sleep({child_sec})"')
    prefix = str(dest / "with_p")
    cmd = [energy.SOCWATCH_PATH, "-f", "pkg-pwr", "-m",
           "-n", "100", "-t", str(window_sec), "-o", prefix, "-p", child]
    print(f"  running: socwatch -t {window_sec} -p <{child_sec}s child>")
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=window_sec + 90)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    wall = time.perf_counter() - t0
    out["wall_sec"] = round(wall, 1)
    out["returncode"] = rc
    out["child_sec"] = child_sec
    out["window_sec"] = window_sec

    # If -p brackets the child, wall time tracks the child, not -t.
    midpoint = (child_sec + window_sec) / 2
    out["p_flag_brackets_child"] = wall < midpoint
    print(f"  wall={wall:.1f}s (child {child_sec}s, -t {window_sec}s) "
          f"-> brackets={out['p_flag_brackets_child']}")

    csv_path = None
    for cand in sorted(dest.glob("*.csv")):
        csv_path = cand
        break
    if csv_path is None:
        out["notes"].append("no CSV produced; check SoC Watch driver/elevation")
        return out
    out["csv"] = str(csv_path)

    text = csv_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Where does the summary section start, and is there anything that looks
    # like a Package/Power row BEFORE it? That is the case that would break
    # energy.parse_csv's first-match-wins scan.
    summary_idx = next(
        (i for i, ln in enumerate(lines) if "Package Power Summary" in ln),
        None,
    )
    first_pkg_row = None
    for i, ln in enumerate(lines):
        parts = [c.strip() for c in ln.split(",")]
        if len(parts) >= 4 and "Package" in parts[0] and "Power" in parts[1]:
            first_pkg_row = i
            break
    out["summary_header_line"] = summary_idx
    out["first_package_power_row"] = first_pkg_row
    out["timeseries_before_summary"] = bool(
        summary_idx is not None and first_pkg_row is not None
        and first_pkg_row < summary_idx
    )

    for key in ("Total # of samples", "Avg sampling interval (msec)",
                "Collection duration"):
        for ln in lines[:60]:
            if key in ln:
                out[key] = ln.strip()
                break

    try:
        parsed = energy.parse_csv(str(csv_path))
        out["parsed"] = parsed
    except Exception as e:
        out["parse_error"] = str(e)
        out["notes"].append(f"energy.parse_csv failed on this CSV: {e}")

    if out["timeseries_before_summary"]:
        out["notes"].append(
            "A Package/Power row appears BEFORE the summary header -- "
            "energy.parse_csv would read the wrong row. Anchor the parser to "
            "the 'Package Power Summary' section (Phase 4)."
        )
    if out.get("parsed") and out["p_flag_brackets_child"]:
        out["verdict"] = "PASS"
        out["notes"].append(
            "Use `-p \"<workload>\"` for the clean-energy pass: the window "
            "tracks the workload with no idle padding."
        )
    elif out.get("parsed"):
        out["verdict"] = "WARN"
        out["notes"].append(
            "CSV parses but -p did not shorten the window. Fall back to "
            "-t sized from the measured span, and subtract the idle baseline."
        )
    return out


# ---------- driver ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None,
                    help="Subset of probe ids to run, e.g. --only 0.1 0.3")
    ap.add_argument("--out", default=str(Path(__file__).parent.parent /
                                         "results" / "probe_env"),
                    help="Directory for probe artifacts and probe_report.json")
    ap.add_argument("--result-dir", default=None,
                    help="Existing finalized VTune result dir for probe 0.1 "
                         "(skips the reference collection)")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="Duration passed to workloads in probe 0.3")
    ap.add_argument("--span-budget", type=float, default=45.0,
                    help="Probe 0.3 flags any workload slower than this")
    ap.add_argument("--campaign-output", default=None,
                    help="Path probe 0.5 checks for free space "
                         "(default: --out)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.only) if args.only else None

    def want(pid):
        return selected is None or pid in selected

    report = {
        "timestamp": datetime.now().isoformat(),
        "machine": None,
        "python": sys.version.split()[0],
        "vtune_path": vtune.VTUNE_PATH,
        "vtune_present": os.path.exists(vtune.VTUNE_PATH),
        "socwatch_path": energy.SOCWATCH_PATH,
        "socwatch_present": os.path.exists(energy.SOCWATCH_PATH),
        "probes": {},
    }
    try:
        report["machine"] = os.environ.get("COMPUTERNAME")
        report["topology"] = core_topology.get_topology()
    except Exception as e:
        report["topology_error"] = str(e)

    try:
        report["elevated"] = bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        report["elevated"] = None
    if report["elevated"] is False:
        print("WARNING: not running elevated. VTune collection will fail.")

    ref_result = Path(args.result_dir) if args.result_dir else None

    if want("0.1"):
        _hr("Probe 0.1 -- parallel VTune slicing (decides ~11h vs ~6h)")
        try:
            if ref_result is None:
                ref_result, _ = _collect_reference_result(out_dir)
            report["probes"]["0.1"] = probe_parallel_slicing(ref_result)
        except Exception as e:
            report["probes"]["0.1"] = {"verdict": "ERROR", "error": str(e)}
            print(f"  ERROR: {e}")

    if want("0.2"):
        _hr("Probe 0.2 -- does VTune attribute a grandchild process?")
        try:
            report["probes"]["0.2"] = probe_grandchild(out_dir)
        except Exception as e:
            report["probes"]["0.2"] = {"verdict": "ERROR", "error": str(e)}
            print(f"  ERROR: {e}")

    if want("0.3"):
        _hr(f"Probe 0.3 -- real workload spans at --duration {args.duration}")
        try:
            report["probes"]["0.3"] = probe_spans(
                duration=args.duration, span_budget=args.span_budget
            )
        except Exception as e:
            report["probes"]["0.3"] = {"verdict": "ERROR", "error": str(e)}
            print(f"  ERROR: {e}")

    if want("0.4"):
        _hr("Probe 0.4 -- x264 preset sweep (fallback if 0.2 failed)")
        try:
            report["probes"]["0.4"] = probe_x264_presets(target=args.duration)
        except Exception as e:
            report["probes"]["0.4"] = {"verdict": "ERROR", "error": str(e)}
            print(f"  ERROR: {e}")

    if want("0.5"):
        _hr("Probe 0.5 -- disk headroom")
        target = Path(args.campaign_output) if args.campaign_output else out_dir
        try:
            report["probes"]["0.5"] = probe_disk(target)
        except Exception as e:
            report["probes"]["0.5"] = {"verdict": "ERROR", "error": str(e)}
            print(f"  ERROR: {e}")

    if want("0.6"):
        _hr("Probe 0.6 -- socwatch -p / -n behaviour")
        try:
            report["probes"]["0.6"] = probe_socwatch(out_dir)
        except Exception as e:
            report["probes"]["0.6"] = {"verdict": "ERROR", "error": str(e)}
            print(f"  ERROR: {e}")

    # ---- verdict summary ----
    _hr("VERDICTS")
    blocking = {"0.1", "0.2", "0.3"}
    blocked = []
    for pid in sorted(report["probes"]):
        p = report["probes"][pid]
        v = p.get("verdict", "?")
        mark = "BLOCKING" if pid in blocking else "        "
        print(f"  {mark}  {pid}: {v}")
        for n in p.get("notes", []):
            print(f"              - {n}")
        if p.get("error"):
            print(f"              - error: {p['error']}")
        if pid in blocking and v in ("FAIL", "ERROR"):
            blocked.append(pid)

    report_path = out_dir / "probe_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {report_path}")

    if blocked:
        print(f"\n>>> BLOCKING probes failed: {', '.join(blocked)}. "
              f"Do not start the campaign.")
        sys.exit(1)
    print("\n>>> No blocking failures.")


if __name__ == "__main__":
    main()
