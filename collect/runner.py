"""
Data collection harness: orchestrates workload × core-type experiments.

This is the main script for Stage 1 data collection. It runs all combinations
of workloads and core types in randomized order, collecting hardware counters
(via VTune) and energy measurements (via SoC Watch) for each run.

Standard library only, so it runs on the restricted test machine.

Output structure:
    data/raw/<fingerprint>/
    ├── fingerprint.json                  # Machine identity
    ├── manifest.json                     # All runs, rewritten after each one
    ├── manifest_<timestamp>.json         # Never-overwritten snapshot
    ├── compute_t01_p_r00/                # <workload>_t<threads>_<class>_r<rep>
    │   ├── vtune_hw_events.csv           # Per-thread hardware counters
    │   ├── vtune_hw_events_timeline.csv  # Per-thread × per-500ms bucket
    │   ├── energy.csv                    # Package power (if --no-energy off)
    │   └── results.json                  # Parsed + derived metrics
    ├── memory_t01_e_r00/
    │   └── ...
    └── noise_study/                      # (optional) many reps of one config

Run ids are derived from the configuration, not the execution index, so they
are stable across a --resume relaunch even though the order is shuffled.

Usage (elevated shell on test machine):
    # Review the plan and projected wall time without collecting anything
    python runner.py --dry-run --reps 5 --duration 30

    # The Stage 1 campaign
    python runner.py --no-energy --duration 30 --reps 5 --subdir stage1_v4 \\
        --campaign-id stage1_v4 --seed 1337

    # Resume it after an interruption (SAME campaign id, or everything redoes)
    python runner.py --no-energy --duration 30 --reps 5 --subdir stage1_v4 \\
        --campaign-id stage1_v4 --seed 1337 --resume

    # Quick smoke: single workload, single core type, 1 rep
    python runner.py --workloads compute --core-types p --reps 1

    # Custom output location (keep it off OneDrive during measurement)
    python runner.py --output C:\\data\\experiment_001
"""

import argparse
import ctypes
import json
import os
import random
import shutil
import sys
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

# Local imports (same directory)
from core_topology import get_topology
from fingerprint import get_fingerprint, fingerprint_id, save_fingerprint
from workloads import WORKLOADS, DEFAULT_DURATION
import vtune
import energy

# Windows API for process affinity.
# DWORD_PTR is 64-bit on 64-bit Windows; c_size_t matches that.
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.GetProcessAffinityMask.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
kernel32.SetProcessAffinityMask.restype = wintypes.BOOL

# Bumped when results.json gains or changes fields that --resume relies on.
# A run written by an older harness fails the resume predicate and is redone
# rather than silently mixed into a new campaign.
#   1 -- v1-v3 campaigns
#   2 -- adds rep/run_id/campaign_id/status/workload_span_sec, self-terminating
#        timeline, affinity reset before the slice loop
RESULTS_SCHEMA_VERSION = 2


def _atomic_write_json(path, payload):
    """Write JSON so a crash or a Ctrl-C can never leave a half-file.

    The manifest is now rewritten after every run, and results.json is written
    up to three times per run, so a torn write during a 12-hour unattended
    campaign is a real risk rather than a theoretical one. Write to a sibling
    temp file, flush to the platter, then os.replace (atomic on Windows).
    """
    path = str(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _finalize_status(result: dict) -> dict:
    """Stamp a single ``status`` field derived from every *_error key.

    Callers used to check three specific error keys by name, which quietly
    counted a run that failed to pin -- or failed its timeline export -- as a
    success. Deriving the status from the presence of ANY *_error key means a
    new failure mode is caught the day it is added.
    """
    errors = sorted(k for k in result if k.endswith("_error"))
    result["errors_present"] = errors
    result["status"] = "ok" if not errors else "error"
    result["schema_version"] = RESULTS_SCHEMA_VERSION
    return result


def _delete_etl_files(energy_dir) -> int:
    """Delete SoC Watch .etl traces once energy.csv has been parsed.

    Returns the number of files removed. Never raises: a locked .etl is an
    annoyance, not a reason to fail an otherwise-good run.
    """
    removed = 0
    try:
        for name in os.listdir(str(energy_dir)):
            if name.lower().endswith(".etl"):
                try:
                    os.remove(os.path.join(str(energy_dir), name))
                    removed += 1
                except OSError:
                    pass
    except OSError:
        pass
    return removed


def _write_results(output_dir, result: dict) -> dict:
    """Stamp the status and write results.json atomically.

    Single choke point for all three former write sites (two setup-error
    early returns and the happy path), so ``status`` can never be stale
    relative to the error keys actually present.
    """
    _finalize_status(result)
    _atomic_write_json(os.path.join(str(output_dir), "results.json"), result)
    return result


def select_cpus_for_class(core_type: str, topology: dict,
                          thread_count: int = 1) -> tuple:
    """Choose which logical CPUs to pin to, spreading across physical cores.

    Returns (cpu_list, physical_cores_used).

    Why not just take the first N ids of the class: ``p_cores`` is a flat
    sorted list of LOGICAL cpu ids, so on a hyperthreaded P-cluster the first
    four ids are two physical cores with both siblings each. A "4-thread"
    run would then share two cores' execution resources and look far slower
    than a genuine 4-core run -- a measurement artifact indistinguishable, in
    the data, from a real scheduling effect.

    So allocate breadth-first: one sibling from each physical core, and only
    start handing out second siblings once every core in the class is in use.
    """
    if core_type == "p":
        want_performance = True
    elif core_type == "e":
        want_performance = False
    else:
        raise ValueError(
            f"Invalid core_type: {core_type}. Use 'p', 'e', or 'default'"
        )

    cores = topology.get("cores") or []
    if cores:
        classes = sorted({eff for eff, _ in cores})
        performance_class = max(classes)
        selected = [
            sorted(cpus) for eff, cpus in cores
            if (eff == performance_class) == want_performance
        ]
        # Stable, topology-order allocation: lowest cpu id first.
        selected.sort(key=lambda cpus: cpus[0] if cpus else 0)
    else:
        # Fall back to the flat lists if this topology dict predates "cores".
        flat = topology["p_cores"] if want_performance else topology["e_cores"]
        selected = [[c] for c in flat]

    if not selected:
        raise ValueError(f"No {core_type}-cores found in topology")

    if thread_count < 1:
        thread_count = 1

    chosen = []
    used_cores = []
    depth = 0
    while len(chosen) < thread_count:
        progressed = False
        for idx, cpus in enumerate(selected):
            if depth >= len(cpus):
                continue
            progressed = True
            chosen.append(cpus[depth])
            if idx not in used_cores:
                used_cores.append(idx)
            if len(chosen) >= thread_count:
                break
        if not progressed:
            break  # every logical CPU in the class is already in the mask
        depth += 1

    return sorted(chosen), [selected[i] for i in used_cores]


def pin_to_core_type(core_type: str, topology: dict, thread_count: int = 1) -> list:
    """Pin this process to ``thread_count`` CPUs of the given core class.

    Windows: child processes inherit their parent's affinity mask at creation.
    Setting the mask on the runner process here propagates through the
    ``vtune.exe`` child and its ``workload`` grandchild, so the workload's
    threads land on the intended cluster.

    For thread_count == 1 (Python micro-benchmarks) this behaves like the
    previous single-CPU pin. For thread_count > 1 (real multi-threaded apps
    like ``7z b -mmt8``) the mask spreads across distinct physical cores
    first -- see :func:`select_cpus_for_class` -- so all N worker threads run
    on the intended class, on N separate cores, and no thread bleeds onto the
    other cluster.

    Args:
        core_type: 'p' for P-core cluster, 'e' for E-core cluster,
                   'default' for no pinning (and clears any existing mask).
        topology: Result from get_topology().
        thread_count: Number of CPUs to include in the mask. Capped to the
                      class size (e.g. 16 threads on a P-cluster of 8 CPUs
                      pins to all 8).

    Returns:
        List of CPU numbers in the mask (empty list for 'default').

    Raises:
        ValueError: If core_type is invalid or no cores of that type exist.
        ctypes.WinError: If SetProcessAffinityMask fails.
    """
    if core_type == "default":
        # Clear any mask left over from a previous run rather than inheriting
        # it. Returning early without this is how an unpinned run could end up
        # silently pinned to whatever the run before it used.
        reset_affinity()
        return []

    chosen, _physical = select_cpus_for_class(core_type, topology, thread_count)
    mask = 0
    for c in chosen:
        mask |= 1 << c

    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
        raise ctypes.WinError(ctypes.get_last_error())

    return chosen


def reset_affinity():
    """Reset process affinity to allow all cores."""
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
        kernel32.GetCurrentProcess(),
        ctypes.byref(process_mask),
        ctypes.byref(system_mask),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    # Pass the c_size_t object directly; ctypes handles the conversion.
    # Passing system_mask.value (a Python int) can overflow on large masks.
    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), system_mask):
        raise ctypes.WinError(ctypes.get_last_error())


def _build_workload_command(workload: str, thread_count: int,
                            duration_sec: float) -> tuple:
    """Turn a workload spec into an argv list runner.py can pass to VTune.

    Returns (workload_cmd, expected_duration_sec, missing_inputs).

    ``missing_inputs`` is a list of any declared ``input_files`` (relative to
    the collect/ folder) that don't exist on disk — the caller must fail fast
    if it's non-empty. External binaries themselves are also checked here.
    """
    spec = WORKLOADS[workload]
    collect_dir = Path(__file__).parent

    if spec["kind"] == "python":
        cmd = [
            sys.executable,
            str(collect_dir / "workloads.py"),
            workload,
            "--duration", str(duration_sec),
            "--quiet",
        ]
        return cmd, float(duration_sec), []

    # kind == 'external'
    args = spec["args_builder"](thread_count, duration_sec)

    exe_rel = spec.get("exe")
    if exe_rel is not None:
        exe_path = collect_dir / exe_rel
        cmd = [str(exe_path)] + list(args)
        missing = [] if exe_path.exists() else [exe_rel]
    else:
        # args already starts with the interpreter (SQLite wrapper case).
        # Nothing to check for the exe itself; the wrapper script lives in
        # collect/ and is guaranteed to be present alongside runner.py.
        cmd = list(args)
        missing = []

    for f in spec.get("input_files", []):
        if not (collect_dir / f).exists():
            missing.append(f)

    return cmd, float(spec["expected_duration_sec"]), missing


def run_single(
    workload: str,
    core_type: str,
    duration_sec: float,
    output_dir: str,
    topology: dict,
    collect_energy: bool = True,
    thread_count: int = 1,
    bin_size_sec: float = 0.5,
    keep_vtune_raw: bool = False,
    rep: int = 0,
    run_id: str = None,
    campaign_id: str = None,
    max_bins: int = 120,
    max_workers: int = 1,
    min_free_gb: float = 5.0,
) -> dict:
    """Run a single experiment: one workload, one core type, one thread count.

    Args:
        workload: Name from WORKLOADS dict
        core_type: 'p', 'e', or 'default'
        duration_sec: Workload duration hint (Python micros honour it directly;
                      external workloads use the spec's expected_duration_sec
                      to size the SoC Watch window)
        output_dir: Where to store results for this run
        topology: Core topology dict
        collect_energy: Whether to run SoC Watch (needs separate elevation)
        thread_count: How many worker threads the workload should launch, and
                      also how many cores get included in the affinity mask.
        bin_size_sec: Bucket width for the per-thread time-series export. The
                      collection itself is unchanged; after it finishes we
                      call VTune once per bucket with ``-time-filter`` and
                      stitch the outputs into vtune_hw_events_timeline.csv.
                      Default 0.5s ≈ 30 buckets per 15-sec workload.
        keep_vtune_raw: If False (default) the ``vtune_result/`` folder is
                        deleted after the CSVs are successfully produced —
                        the raw artifacts blow up disk (~200 MB per run × 240
                        runs = ~50 GB) and Stage 2+ only needs the CSVs.
                        Set True to keep them for VTune GUI inspection.
        rep: Which repetition of this configuration this is. Recorded so
             Model C can join the P and E run of the SAME repetition instead
             of pairing them by position.
        run_id: Identifier recorded inside results.json. Must match the folder
                name; --resume checks both.
        campaign_id: Tags every run of one campaign so --resume never adopts a
                     leftover folder from a previous sweep.
        max_bins: Ceiling on timeline buckets (see vtune.export_time_sliced_csv).
        max_workers: Concurrent VTune report processes for the slice loop.
        min_free_gb: Refuse to start if the output volume has less than this
                     much free space.

    Returns:
        dict with run results (see keys below).

    Raises:
        OSError: if free disk is below ``min_free_gb``. This one propagates
                 deliberately -- degrading it into a result field would let a
                 12-hour campaign keep "succeeding" with empty runs after the
                 volume filled up.
    """
    os.makedirs(output_dir, exist_ok=True)

    free_gb = shutil.disk_usage(output_dir).free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise OSError(
            f"Only {free_gb:.1f} GB free on the volume holding {output_dir}; "
            f"need at least {min_free_gb} GB. Stopping before the campaign "
            f"starts producing truncated runs."
        )

    result = {
        "run_id": run_id,
        "campaign_id": campaign_id,
        "schema_version": RESULTS_SCHEMA_VERSION,
        "workload": workload,
        "core_type": core_type,
        "thread_count": thread_count,
        "rep": rep,
        "cpu_pinned": [],
        "physical_cores": [],
        "duration_sec": duration_sec,
        "actual_duration_sec": None,
        "collect_wall_sec": None,
        "slice_wall_sec": None,
        "workload_span_sec": None,
        "free_gb_at_start": round(free_gb, 2),
        "timestamp": datetime.now().isoformat(),
        "vtune": None,
        "aggregate": None,
        "timeline": None,
        "energy": None,
        "energy_truncated": None,
        "ipc": None,
    }

    # Build the workload command up front so we can fail fast on a missing
    # binary or input file before starting VTune/SoC Watch.
    try:
        workload_cmd, expected_duration, missing = _build_workload_command(
            workload, thread_count, duration_sec
        )
    except Exception as e:
        result["setup_error"] = f"Failed to build workload command: {e}"
        _write_results(output_dir, result)
        return result

    if missing:
        result["setup_error"] = (
            "Missing required files (run collect/workload_setup.py on the dev "
            "machine and copy the results over): " + ", ".join(missing)
        )
        _write_results(output_dir, result)
        return result

    # Pin to the requested cluster with a mask sized to thread_count.
    try:
        result["cpu_pinned"] = pin_to_core_type(core_type, topology, thread_count)
        if core_type in ("p", "e"):
            _cpus, physical = select_cpus_for_class(
                core_type, topology, thread_count
            )
            result["physical_cores"] = physical
    except Exception as e:
        result["pin_error"] = str(e)

    # VTune result directory
    vtune_dir = os.path.join(output_dir, "vtune_result")
    vtune_csv = os.path.join(output_dir, "vtune_hw_events.csv")
    vtune_timeline_csv = os.path.join(output_dir, "vtune_hw_events_timeline.csv")

    # Energy collection directory
    energy_dir = os.path.join(output_dir, "energy")
    energy_proc = None
    energy_window_sec = None

    # Start energy collection in background BEFORE VTune (if requested).
    # SoC Watch window = max(30, 3x expected + 5). The 3x margin covers
    # slow runs (thermal throttling, cold caches on first repetition, etc.)
    # without leaving the window ridiculously long for short workloads.
    if collect_energy:
        try:
            # Tight window: workload span is ~30s, VTune collection is ~68s.
            # A 45s SoC Watch window finishes before VTune → zero extra time.
            # The old formula (3×expected+5 = 95s) caused 84% idle overhead.
            energy_window_sec = max(40, int(expected_duration) + 15)
            energy_proc = energy.collect_background(energy_window_sec, energy_dir)
            time.sleep(0.5)  # Brief pause to let SoC Watch initialize
        except Exception as e:
            result["energy_error"] = f"Failed to start: {e}"
            energy_proc = None

    # Run VTune collection (this also runs the workload)
    start_time = time.perf_counter()
    try:
        # cwd = collect/ so the external workloads' relative paths
        # ("workloads_data/silesia.tar", "workloads_sqlite.py", etc.) resolve
        # correctly regardless of where runner.py was invoked from.
        vtune.collect(workload_cmd, vtune_dir, cwd=str(Path(__file__).parent))
        # Wall time of the workload+VTune step. This is NOT workload time --
        # it is dominated by VTune's own startup and finalize (48-95s observed
        # for workloads spanning 7-35s), which is exactly why it must not be
        # used to size the bin grid. Recorded for diagnostics only.
        result["collect_wall_sec"] = round(time.perf_counter() - start_time, 3)

        # Measurement is over the moment collect() returns, so drop the
        # affinity mask here rather than at the end of the function. VTune's
        # report tool is single-threaded and clock-bound: leaving the mask on
        # through the slice loop cost 2.67 s/slice on E-cores vs 1.90 s
        # unpinned, ~1.3 h across a 180-run campaign.
        try:
            reset_affinity()
            result["affinity_reset_before_slicing"] = True
        except Exception as e:
            result["affinity_reset_error"] = str(e)

        vtune.export_csv(vtune_dir, vtune_csv)

        # Per-thread × per-time-bucket export. Isolated from the aggregate
        # export so a failure here doesn't lose the aggregate CSV (which is
        # the historical output every earlier stage of code depends on).
        #
        # duration_sec is deliberately NOT passed: the exporter discovers the
        # real span by slicing until it sees consecutive empty buckets. Passing
        # collect_wall_sec here is the v3 root-cause defect (85% wasted slice
        # calls, 17 runs whose timeline was discarded by a late crash).
        slice_start = time.perf_counter()
        try:
            result["timeline"] = vtune.export_time_sliced_csv(
                vtune_dir, vtune_timeline_csv,
                duration_sec=None,
                bin_size_sec=bin_size_sec,
                max_bins=max_bins,
                max_workers=max_workers,
            )
            result["workload_span_sec"] = result["timeline"].get("max_bin_end_sec")
        except Exception as e:
            result["timeline_error"] = str(e)
        result["slice_wall_sec"] = round(time.perf_counter() - slice_start, 3)

        rows = vtune.parse_csv(vtune_csv)
        rows = vtune.compute_derived_metrics(rows)

        # Preserve v1's primary-thread block for backward compatibility with
        # existing analysis code, and add an aggregate block so multi-threaded
        # workloads get a meaningful cluster-wide IPC.
        primary = vtune.get_primary_thread(rows)
        aggregate = vtune.aggregate_workload_threads(rows)

        result["vtune"] = primary
        result["aggregate"] = aggregate
        # Prefer aggregate IPC as the "run's IPC" for multi-threaded workloads;
        # for single-threaded workloads it collapses to the primary thread's IPC.
        result["ipc"] = aggregate.get("aggregate_ipc") or primary.get("ipc")
        result["all_threads"] = len(rows)

    except Exception as e:
        result["vtune_error"] = str(e)

    result["actual_duration_sec"] = time.perf_counter() - start_time

    # Auto-cleanup: the ~200 MB vtune_result/ folder is only needed for VTune
    # GUI inspection. Once both CSVs are on disk the raw artifact is dead
    # weight — 240 runs × ~200 MB = ~50 GB, which blew up the disk last
    # campaign. Only delete if the aggregate CSV was actually produced;
    # otherwise leave the raw folder so a failed run can be debugged.
    if (not keep_vtune_raw
            and os.path.exists(vtune_csv)
            and "vtune_error" not in result
            and os.path.isdir(vtune_dir)):
        try:
            shutil.rmtree(vtune_dir)
            result["vtune_raw_cleaned"] = True
        except Exception as e:
            result["vtune_raw_cleanup_error"] = str(e)

    # Flag energy_truncated BEFORE waiting on SoC Watch. The comparison is
    # against workload_span_sec, not actual_duration_sec: the latter includes
    # the whole slice loop (~2 s x bins), which made 183 of 185 v3 runs claim
    # truncation that never happened. None means "span unknown", which is
    # genuinely different from False and must not be conflated with it.
    if energy_window_sec is not None:
        span = result.get("workload_span_sec")
        if span is None:
            result["energy_truncated"] = None
        else:
            result["energy_truncated"] = bool(span > energy_window_sec)

    # Wait for energy collection to complete and parse results
    if energy_proc is not None:
        try:
            # Wait up to the SoC Watch window plus generous headroom.
            energy_proc.wait(timeout=(energy_window_sec or 30) + 30)
            if energy_proc.returncode == 0:
                result["energy"] = energy.parse_output_dir(energy_dir)
                # .etl traces are ~75 MB/run -- 558 files / 12.5 GB last
                # campaign, ~97% of everything collected. Once energy.csv has
                # parsed, the trace is dead weight. Delete per run rather than
                # in a cleanup pass so the volume never fills mid-campaign.
                result["etl_deleted"] = _delete_etl_files(energy_dir)
            else:
                result["energy_error"] = f"SoC Watch exited with code {energy_proc.returncode}"
        except Exception as e:
            result["energy_error"] = f"Failed to collect: {e}"
            try:
                energy_proc.kill()
            except Exception:
                pass

    # Belt-and-braces: if the pre-slicing reset failed or was never reached
    # (e.g. vtune.collect raised), make sure the next run doesn't inherit
    # this run's mask.
    try:
        reset_affinity()
    except Exception:
        pass

    return _write_results(output_dir, result)


def make_run_id(workload: str, thread_count: int, core_type: str,
                rep: int) -> str:
    """Deterministic folder name for one cell of the experiment matrix.

    The v3 scheme was ``run_{index:03d}_{workload}_t{tc}_{ct}`` over the
    SHUFFLED list, so the same configuration got a different id on every
    campaign and two different configurations could share an index prefix
    (``run_001_io_t01_e`` and ``run_001_zstd_t16_e`` both existed). Keying on
    the configuration instead makes --resume possible at all, and makes ids
    collision-free by construction.
    """
    return f"{workload}_t{thread_count:02d}_{core_type}_r{rep:02d}"


def _csv_has_content(path, min_bytes: int = 64) -> bool:
    """True if ``path`` is a CSV with more than just a header in it."""
    try:
        return os.path.getsize(str(path)) >= min_bytes
    except OSError:
        return False


def _run_is_complete(run_dir, workload: str, core_type: str,
                     thread_count: int, rep: int,
                     campaign_id: str = None) -> tuple:
    """Strict --resume predicate. Returns (complete, reason).

    "results.json exists" is NOT sufficient: it is also written on the
    setup_error path, and a v3 folder can hold CSVs from one date beside an
    energy/ directory from another. Every one of these checks exists because
    something in the audit could have slipped past a weaker one.
    """
    run_dir = Path(run_dir)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return False, "no results.json"

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        return False, f"unreadable results.json ({e})"

    if not isinstance(data, dict):
        return False, "results.json is not an object"
    if data.get("status") != "ok":
        return False, f"status={data.get('status')!r}"
    if data.get("schema_version") != RESULTS_SCHEMA_VERSION:
        return False, (f"schema_version={data.get('schema_version')!r}, "
                       f"want {RESULTS_SCHEMA_VERSION}")
    if campaign_id is not None and data.get("campaign_id") != campaign_id:
        return False, f"campaign_id={data.get('campaign_id')!r}"

    want = (workload, core_type, int(thread_count), int(rep))
    try:
        got = (data.get("workload"), data.get("core_type"),
               int(data.get("thread_count")), int(data.get("rep")))
    except (TypeError, ValueError):
        return False, "missing/invalid config fields"
    if got != want:
        return False, f"config mismatch: {got} != {want}"

    for name in ("vtune_hw_events.csv", "vtune_hw_events_timeline.csv"):
        if not _csv_has_content(run_dir / name):
            return False, f"{name} missing or empty"

    return True, "complete"


def _thread_counts_for(workload: str, override: list = None) -> list:
    """Return the thread counts to sweep for one workload.

    Uses the spec's ``thread_counts`` unless the caller supplied an override
    (e.g. ``--thread-counts 1 4 8``). Threading is single-threaded (``[1]``)
    for Python micro-benchmarks whatever the override says — running a
    ``compute_bound`` Python loop with N threads still executes serially due
    to the GIL, so sweeping it would just add duplicate runs.
    """
    spec = WORKLOADS[workload]
    if spec["kind"] == "python":
        return spec["thread_counts"]  # always [1]
    return override if override else spec["thread_counts"]


def run_experiment(
    workloads: list,
    core_types: list,
    reps: int,
    output_base: str,
    duration_sec: float = DEFAULT_DURATION,
    randomize: bool = True,
    collect_energy: bool = True,
    noise_study: bool = False,
    thread_counts: list = None,
    output_subdir: str = None,
    bin_size_sec: float = 0.5,
    keep_vtune_raw: bool = False,
    seed: int = 1337,
    resume: bool = False,
    campaign_id: str = None,
    max_bins: int = 120,
    max_workers: int = 1,
    min_free_gb: float = 5.0,
    dry_run: bool = False,
    sec_per_run_estimate: float = None,
    assume_class_sizes: tuple = None,
) -> dict:
    """Run the full experiment matrix.

    Args:
        workloads: List of workload names from WORKLOADS
        core_types: List of core types ('p', 'e', 'default')
        reps: Number of repetitions per combination
        output_base: Base directory for output
        duration_sec: Duration per workload run
        randomize: Shuffle run order to reduce systematic bias
        collect_energy: Whether to run SoC Watch
        noise_study: If True, run one config many times for variance analysis
        seed: Seeds a local random.Random so the shuffled order is
              reproducible -- required for --resume to be meaningful, and so a
              campaign can be described exactly in the thesis.
        resume: Skip runs whose folder passes :func:`_run_is_complete`. Folders
                that exist but FAIL the predicate are renamed aside rather than
                written into.
        campaign_id: Tag stamped into every results.json and checked by
                     --resume. Defaults to a timestamp.
        max_bins: Ceiling on timeline buckets per run.
        max_workers: Concurrent VTune report processes per slice loop.
        min_free_gb: Abort the campaign if the volume drops below this.
        dry_run: Print the plan and the projected wall time, run nothing.
        sec_per_run_estimate: Per-run seconds used for the dry-run projection
                              (from probe 0.3). Defaults to a conservative 180.
        assume_class_sizes: ``(n_p, n_e)`` used ONLY when dry_run is set and
                            the machine has no hybrid topology, so the plan
                            can be reviewed on the (non-hybrid) dev machine.
                            Never consulted on a real collection run.

    Returns:
        dict with experiment summary:
            fingerprint: machine fingerprint
            runs: list of all run results
            total_time_sec: wall-clock time
            errors: count of failed runs
            complete: False if the campaign was interrupted
    """
    # Get machine fingerprint
    fp = get_fingerprint()
    fp_id = fingerprint_id(fp)
    if campaign_id is None:
        campaign_id = datetime.now().strftime("campaign_%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = Path(output_base) / fp_id
    if noise_study:
        output_dir = output_dir / "noise_study"
    elif output_subdir:
        output_dir = output_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save fingerprint
    save_fingerprint(fp, str(output_dir / "fingerprint.json"))

    # Get core topology
    assumed_topology = False
    try:
        topology = get_topology()
    except RuntimeError:
        # A non-hybrid machine can still plan a campaign, which is how the
        # run list gets reviewed on the dev machine before moving over. It can
        # never COLLECT one, so this fallback is gated on dry_run.
        if not (dry_run and assume_class_sizes):
            raise
        n_p, n_e = int(assume_class_sizes[0]), int(assume_class_sizes[1])
        topology = {
            "p_cores": list(range(n_p)),
            "e_cores": list(range(n_p, n_p + n_e)),
            "cores": [],
            "assumed": True,
        }
        assumed_topology = True

    class_size = {"p": len(topology["p_cores"]), "e": len(topology["e_cores"])}

    # Build run list. Each entry is (workload, core_type, thread_count, rep).
    if noise_study:
        # Noise study: single config, many reps. Threading defaults to the
        # spec's first thread-count (1 for python micros, 1 for externals too
        # since that's the calibration point).
        workload = workloads[0] if workloads else "compute"
        core_type = core_types[0] if core_types else "p"
        tc = _thread_counts_for(workload, thread_counts)[0]
        runs_planned = [(workload, core_type, tc, i) for i in range(reps)]
    else:
        runs_planned = []
        for w in workloads:
            for tc in _thread_counts_for(w, thread_counts):
                for ct in core_types:
                    # Skip (thread_count > cluster size) — pinning would cap the
                    # mask silently and the workload would end up over-subscribed
                    # on the intended cluster.
                    if ct in class_size and tc > class_size[ct]:
                        continue
                    for rep in range(reps):
                        runs_planned.append((w, ct, tc, rep))

    # Deterministic shuffle: a local Random instance, so seeding here can't be
    # perturbed by any other module's use of the global random state.
    if randomize:
        random.Random(seed).shuffle(runs_planned)

    # Run ids are keyed on the configuration, so they're stable across a
    # --resume relaunch even though the execution order is shuffled.
    planned = [
        (make_run_id(w, tc, ct, rep), w, ct, tc, rep)
        for (w, ct, tc, rep) in runs_planned
    ]
    duplicate_ids = sorted(
        {rid for rid in (p[0] for p in planned)
         if [q[0] for q in planned].count(rid) > 1}
    )
    if duplicate_ids:
        raise ValueError(
            "Run id collision in the plan, which would make two runs share a "
            f"folder: {duplicate_ids}"
        )

    print(f"Machine: {fp_id}")
    print(f"Topology: {len(topology['p_cores'])} P-cores, "
          f"{len(topology['e_cores'])} E-cores"
          + ("  *** ASSUMED, not measured ***" if assumed_topology else ""))
    print(f"Campaign: {campaign_id}  (seed {seed})")
    print(f"Planned runs: {len(planned)}")
    print(f"Output: {output_dir}")

    if dry_run:
        per_run = sec_per_run_estimate if sec_per_run_estimate else 180.0
        projected = len(planned) * per_run
        print()
        for rid, w, ct, tc, rep in planned:
            print(f"  {rid:<34} {w} t={tc} {ct}-cluster rep={rep}")
        print()
        print(f"Projected wall time: {projected / 3600.0:.2f} h "
              f"({per_run:.0f}s/run x {len(planned)} runs)")
        try:
            free_gb = shutil.disk_usage(str(output_dir)).free / (1024 ** 3)
            print(f"Free disk: {free_gb:.1f} GB "
                  f"(need >= {min_free_gb} GB throughout)")
        except OSError:
            pass
        print("Dry run: nothing executed.")
        return {
            "campaign_id": campaign_id,
            "dry_run": True,
            "planned_run_ids": [p[0] for p in planned],
            "projected_wall_sec": projected,
        }
    print()

    # Run experiments
    results = []
    errors = 0
    skipped = 0
    stale_renamed = []
    interrupted = False
    start_time = time.perf_counter()

    manifest_path = output_dir / "manifest.json"

    def _build_manifest(complete: bool) -> dict:
        return {
            "campaign_id": campaign_id,
            "schema_version": RESULTS_SCHEMA_VERSION,
            "seed": seed,
            "complete": complete,
            "interrupted": interrupted,
            "fingerprint": fp,
            "fingerprint_id": fp_id,
            "topology": topology,
            "workloads": workloads,
            "core_types": core_types,
            "reps": reps,
            "duration_sec": duration_sec,
            "randomized": randomize,
            "noise_study": noise_study,
            "timeline_bin_size_sec": bin_size_sec,
            "max_bins": max_bins,
            "max_workers": max_workers,
            "collect_energy": collect_energy,
            "keep_vtune_raw": keep_vtune_raw,
            "planned_runs": len(planned),
            "total_runs": len(results),
            "skipped_complete": skipped,
            "stale_renamed": stale_renamed,
            "errors": errors,
            "total_time_sec": time.perf_counter() - start_time,
            "timestamp": datetime.now().isoformat(),
            "runs": [
                {
                    "run_id": r.get("run_id"),
                    "workload": r.get("workload"),
                    "core_type": r.get("core_type"),
                    "thread_count": r.get("thread_count"),
                    "rep": r.get("rep"),
                    "status": r.get("status"),
                    "errors_present": r.get("errors_present"),
                    "ipc": r.get("ipc"),
                    "energy_mj": (r.get("energy") or {}).get("total_energy_mj"),
                    "energy_truncated": r.get("energy_truncated"),
                    "collect_wall_sec": r.get("collect_wall_sec"),
                    "slice_wall_sec": r.get("slice_wall_sec"),
                    "workload_span_sec": r.get("workload_span_sec"),
                    "timeline_bins": (r.get("timeline") or {}).get("bins_written"),
                    "timeline_bins_requested": (r.get("timeline") or {}).get("bins_requested"),
                    "timeline_bins_failed": (r.get("timeline") or {}).get("bins_failed"),
                    "timeline_stopped_early": (r.get("timeline") or {}).get("stopped_early"),
                    "timeline_stop_reason": (r.get("timeline") or {}).get("stop_reason"),
                    "timeline_rows": (r.get("timeline") or {}).get("rows_written"),
                }
                for r in results
            ],
        }

    def _save_manifest(complete: bool) -> dict:
        m = _build_manifest(complete)
        # Atomic: the manifest is now rewritten after every run, so a torn
        # write during a 12-hour campaign is a real risk. This is also what
        # makes Ctrl-C at hour 9 survivable.
        _atomic_write_json(manifest_path, m)
        return m

    # Write the plan to disk BEFORE the first run, so a campaign that dies on
    # run 1 -- or gets killed by a reboot -- still leaves a record of what it
    # intended to do and which seed produced the order.
    manifest = _save_manifest(False)
    try:
        for i, (run_id, workload, core_type, tc, rep) in enumerate(planned):
            run_dir = output_dir / run_id
            label = f"[{i+1}/{len(planned)}] {run_id}"

            if run_dir.exists():
                complete, reason = _run_is_complete(
                    run_dir, workload, core_type, tc, rep, campaign_id
                )
                if complete and resume:
                    skipped += 1
                    print(f"{label} ... skip (already complete)")
                    continue

                # Never write into a pre-existing folder -- not one that
                # failed the predicate, and not a complete one we were asked
                # to redo. Mixing a half-finished or foreign run with a new
                # one is how run_003_compute_p ended up holding Aug-31 CSVs
                # beside a Sep-27 energy/ directory: data that looks valid
                # and isn't. os.makedirs(exist_ok=True) made that silent.
                if complete:
                    reason = "complete, but re-running without --resume"
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                aside = output_dir / f"{run_id}__stale_{ts}"
                try:
                    os.replace(str(run_dir), str(aside))
                    stale_renamed.append(
                        {"run_id": run_id, "moved_to": aside.name,
                         "reason": reason}
                    )
                    print(f"{label} ... existing folder set aside "
                          f"({reason}) -> {aside.name}")
                except OSError as e:
                    print(f"{label} ... WARNING could not set aside existing "
                          f"folder ({e}); skipping to avoid mixing data")
                    errors += 1
                    continue

            print(f"{label} {workload} t={tc} on {core_type}-cluster... ",
                  end="", flush=True)

            try:
                result = run_single(
                    workload=workload,
                    core_type=core_type,
                    duration_sec=duration_sec,
                    output_dir=str(run_dir),
                    topology=topology,
                    collect_energy=collect_energy,
                    thread_count=tc,
                    bin_size_sec=bin_size_sec,
                    keep_vtune_raw=keep_vtune_raw,
                    rep=rep,
                    run_id=run_id,
                    campaign_id=campaign_id,
                    max_bins=max_bins,
                    max_workers=max_workers,
                    min_free_gb=min_free_gb,
                )
                results.append(result)

                ipc = result.get("ipc")
                span = result.get("workload_span_sec")
                bits = []
                bits.append(f"IPC={ipc:.2f}" if ipc else "no IPC")
                if span is not None:
                    bits.append(f"span={span:.1f}s")
                tl = result.get("timeline") or {}
                if tl:
                    bits.append(f"bins={tl.get('bins_written')}"
                                f"/{tl.get('bins_requested')}")
                    if tl.get("bins_failed"):
                        bits.append(f"FAILED_BINS={tl['bins_failed']}")
                print("  ".join(bits))

                # Count ANY error key. The old check named three keys, so a
                # run that failed to pin -- or lost its timeline -- was
                # reported as a success.
                if result.get("status") != "ok":
                    errors += 1

            except OSError as e:
                # Disk preflight. Not recoverable, and continuing would just
                # produce a queue of truncated runs.
                print(f"ABORT: {e}")
                errors += 1
                raise
            except Exception as e:
                print(f"FAILED: {e}")
                errors += 1
                results.append({
                    "run_id": run_id, "workload": workload,
                    "core_type": core_type, "thread_count": tc, "rep": rep,
                    "status": "error", "errors_present": ["run_error"],
                    "run_error": str(e),
                })

            # Write the manifest after EVERY run, not just at the end. v3 lost
            # a 185-run manifest to a later 40-run sweep; a campaign that dies
            # at run 170 should still describe its first 169.
            manifest = _save_manifest(False)

    except KeyboardInterrupt:
        # KeyboardInterrupt is not an Exception, so the old handler missed it
        # entirely and the manifest was never written. Nine hours of a 12-hour
        # campaign were one Ctrl-C from being undescribed.
        interrupted = True
        print()
        print("Interrupted by user; writing manifest for completed runs.")
    finally:
        try:
            reset_affinity()
        except Exception:
            pass
        complete = (not interrupted
                    and (len(results) + skipped) == len(planned))
        manifest = _save_manifest(complete)
        # Never-overwritten snapshot. manifest.json itself is rewritten by the
        # next sweep into the same subdir; this is what preserved the 140-run
        # manifest last time. One-second timestamp granularity is enough for
        # real campaigns but not for tests or aborted relaunches, so the name
        # is made unique rather than assumed unique.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = output_dir / f"manifest_{ts}.json"
        suffix = 1
        while snapshot_path.exists():
            snapshot_path = output_dir / f"manifest_{ts}_{suffix:02d}.json"
            suffix += 1
        try:
            _atomic_write_json(snapshot_path, manifest)
        except OSError:
            snapshot_path = None

        total_time = manifest["total_time_sec"]
        print()
        print(f"Runs executed: {len(results)} of {len(planned)} planned "
              f"in {total_time:.1f}s")
        if skipped:
            print(f"Skipped (already complete): {skipped}")
        if stale_renamed:
            print(f"Folders set aside: {len(stale_renamed)}")
        print(f"Errors: {errors}")
        print(f"Campaign complete: {manifest['complete']}")
        print(f"Manifest: {manifest_path}")
        if snapshot_path:
            print(f"Snapshot: {snapshot_path}")

    return manifest


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run data collection experiments for hybrid CPU scheduling research."
    )
    parser.add_argument(
        "--workloads", "-w",
        nargs="+",
        choices=list(WORKLOADS.keys()),
        default=list(WORKLOADS.keys()),
        help="Which workloads to run (default: all)"
    )
    parser.add_argument(
        "--core-types", "-c",
        nargs="+",
        choices=["p", "e", "default"],
        default=["p", "e"],
        help="Which core types to test (default: p e)"
    )
    parser.add_argument(
        "--reps", "-r",
        type=int,
        default=7,
        help="Repetitions per configuration (default: 7)"
    )
    parser.add_argument(
        "--thread-counts",
        nargs="+",
        type=int,
        default=None,
        help="Override the thread-count sweep for external workloads "
             "(e.g. --thread-counts 1 4 8). Python micros are always [1]."
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Optional subdirectory under data/raw/<fingerprint>/ to keep this "
             "collection separate from earlier runs (e.g. --subdir stage1_v2)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Workload duration in seconds (default: {DEFAULT_DURATION})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "raw"),
        help="Output base directory (default: data/raw/)"
    )
    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help="Run in fixed order instead of randomized"
    )
    parser.add_argument(
        "--no-energy",
        action="store_true",
        help="Skip SoC Watch energy collection"
    )
    parser.add_argument(
        "--noise-study",
        action="store_true",
        help="Run one config many times for variance analysis"
    )
    parser.add_argument(
        "--bin-size-sec",
        type=float,
        default=0.5,
        help="Bucket width for the per-thread time-series export "
             "(default: 0.5s = 30 buckets per 15-sec workload, ~63s of "
             "post-processing per run at ~2.1s per VTune call)"
    )
    parser.add_argument(
        "--keep-vtune-raw",
        action="store_true",
        help="Keep the ~200MB vtune_result/ folder for each run. Off by "
             "default because 240 runs × 200 MB = ~50 GB (blew up disk last "
             "campaign). Only turn on for a small run you want to inspect "
             "in the VTune GUI."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed for the run-order shuffle (default: 1337). The same seed "
             "reproduces the same order, which is what makes --resume and the "
             "thesis's methodology section exact."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose folder already holds a complete, matching, "
             "same-campaign result. Folders that exist but fail that check "
             "are renamed to <run_id>__stale_<timestamp> instead of being "
             "written into."
    )
    parser.add_argument(
        "--campaign-id",
        type=str,
        default=None,
        help="Tag stamped into every results.json and required to match by "
             "--resume (default: campaign_<timestamp>). Pass the SAME id when "
             "resuming an interrupted campaign, or every run will be redone."
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=120,
        help="Ceiling on timeline buckets per run (default: 120 = 60s at "
             "0.5s bins). The exporter stops on its own when the workload "
             "ends; this is only a backstop."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Concurrent VTune report processes in the slice loop (default: "
             "1). Only raise this if probe 0.1 confirmed byte-identical "
             "output at that level -- see collect/probe_env.py."
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="Abort rather than start a run when the output volume has less "
             "than this much free space (default: 5.0)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run-id list and the projected wall time, then exit "
             "without collecting anything. This is the gate that catches a "
             "14-hour plan before it starts."
    )
    parser.add_argument(
        "--sec-per-run",
        type=float,
        default=None,
        help="Per-run seconds for the --dry-run projection, from probe 0.3 "
             "(default: a conservative 180)."
    )
    parser.add_argument(
        "--assume-class-sizes",
        nargs=2,
        type=int,
        metavar=("N_P", "N_E"),
        default=None,
        help="With --dry-run only: plan as if the machine had N_P P-cores and "
             "N_E E-cores, so the run list can be reviewed on the non-hybrid "
             "dev machine (e.g. --assume-class-sizes 8 16). Ignored for real "
             "collection, which always measures the topology."
    )

    args = parser.parse_args()

    if args.resume and args.campaign_id is None:
        print("ERROR: --resume needs --campaign-id <id of the campaign being "
              "resumed>.")
        print("A fresh timestamp id would fail the resume check on every run "
              "and silently redo all of them.")
        print("The id is in the campaign's manifest.json under "
              "\"campaign_id\".")
        sys.exit(1)

    if args.max_workers < 1:
        print("ERROR: --max-workers must be >= 1")
        sys.exit(1)

    # Check for VTune. Skipped for --dry-run so the plan can be reviewed on
    # the dev machine before moving to the test machine.
    if not args.dry_run and not os.path.exists(vtune.VTUNE_PATH):
        print(f"ERROR: VTune not found at {vtune.VTUNE_PATH}")
        print("This script must run on the test machine with VTune installed.")
        sys.exit(1)

    # Check for administrator
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        is_admin = False

    if not is_admin and not args.dry_run:
        print("WARNING: Not running as Administrator. VTune/SoC Watch may fail.")
        print("Consider: Run this script from an elevated command prompt.")
        print()

    run_experiment(
        workloads=args.workloads,
        core_types=args.core_types,
        reps=args.reps,
        output_base=args.output,
        duration_sec=args.duration,
        randomize=not args.no_randomize,
        collect_energy=not args.no_energy,
        noise_study=args.noise_study,
        thread_counts=args.thread_counts,
        output_subdir=args.subdir,
        bin_size_sec=args.bin_size_sec,
        keep_vtune_raw=args.keep_vtune_raw,
        seed=args.seed,
        resume=args.resume,
        campaign_id=args.campaign_id,
        max_bins=args.max_bins,
        max_workers=args.max_workers,
        min_free_gb=args.min_free_gb,
        dry_run=args.dry_run,
        sec_per_run_estimate=args.sec_per_run,
        assume_class_sizes=args.assume_class_sizes,
    )


if __name__ == "__main__":
    main()
