"""
Built-in micro-benchmarks for the data collection harness.

These workloads are designed to show measurable differences between P-cores and
E-cores, providing the training signal for the ML models. Each workload runs for
a specified duration (default 5 seconds), allowing consistent collection windows.

Standard library only, so it runs on the restricted test machine.

Workload characteristics:
  compute_bound: Tight arithmetic loop, stays in registers/L1.
                 Large P-core advantage (~1.26x in Stage 0 feasibility test).
  memory_bound:  Pointer-chasing over large working set (>L3 cache).
                 Small P-core advantage (~1.11x).
  mixed_phases:  Alternates between compute and memory phases.
                 Tests the model's ability to detect phase transitions.

Usage:
    # Run from Python
    from workloads import WORKLOADS
    WORKLOADS['compute']()

    # Run as subprocess (what the harness does)
    python workloads.py compute
    python workloads.py memory
    python workloads.py mixed

    # Run with custom duration
    python workloads.py compute --duration 10
"""

import argparse
import json
import os
import random
import sys
import time

# Working set size for memory-bound workloads.
# 4M elements * 8 bytes/int ≈ 32 MB, roughly matches L3 cache.
# To guarantee cache misses, we use a random shuffle so accesses are unpredictable.
# The Core Ultra 9 290HX Plus has 36 MB L3, so this should spill frequently.
CHASE_ELEMENTS = 4_000_000

# Default workload duration in seconds
DEFAULT_DURATION = 5.0


def compute_bound(duration_sec: float = DEFAULT_DURATION) -> int:
    """Tight arithmetic loop, stays in registers/L1. Large P-core advantage.

    This workload is entirely compute-bound. The loop is simple enough that:
    - All data fits in registers or L1 cache
    - No memory stalls occur
    - The P-core's higher clock and wider execution resources dominate

    Returns the final computed value (prevents optimizer from removing the loop).
    """
    end_time = time.perf_counter() + duration_sec
    total = 0
    i = 0
    while time.perf_counter() < end_time:
        # Batch 10000 iterations between time checks to reduce overhead
        for _ in range(10000):
            total += i * i
            i += 1
    return total


def _build_chase(n: int = CHASE_ELEMENTS) -> list:
    """Build a random permutation for pointer-chasing.

    Following p = chase[p] walks a cycle of dependent, unpredictable accesses.
    This defeats hardware prefetching and forces memory latency to dominate.

    Args:
        n: Number of elements (working set size = n * 8 bytes for Python int).

    Returns:
        List where chase[i] points to the next index in the random walk.
    """
    indices = list(range(n))
    random.shuffle(indices)
    return indices


# Pre-built chase array. Building this takes ~2 seconds, so we do it once at
# module load time rather than on each workload invocation.
_CHASE_ARRAY = None


def _get_chase():
    """Lazily initialize the chase array."""
    global _CHASE_ARRAY
    if _CHASE_ARRAY is None:
        _CHASE_ARRAY = _build_chase()
    return _CHASE_ARRAY


def memory_bound(duration_sec: float = DEFAULT_DURATION) -> int:
    """Pointer-chasing over large working set. Small P-core advantage.

    This workload is entirely memory-bound. Each access depends on the previous,
    creating a chain of dependent loads that cannot be parallelized or prefetched.
    The core spends most of its time waiting on DRAM, so the P-core's extra
    execution resources sit idle — both core types wait equally for memory.

    Returns the final pointer value (prevents optimizer from removing the loop).
    """
    chase = _get_chase()
    end_time = time.perf_counter() + duration_sec
    p = 0
    while time.perf_counter() < end_time:
        # Batch accesses between time checks
        for _ in range(10000):
            p = chase[p]
    return p


def mixed_phases(duration_sec: float = DEFAULT_DURATION) -> int:
    """Alternates between compute and memory phases. Tests phase detection.

    This workload switches between compute-intensive and memory-intensive phases
    every 0.5 seconds. It tests whether the ML model can:
    1. Detect phase transitions in the hardware counters
    2. Predict the correct placement for each phase

    Returns the sum of compute and memory results.
    """
    chase = _get_chase()
    end_time = time.perf_counter() + duration_sec
    phase_duration = 0.5  # seconds per phase

    compute_total = 0
    memory_total = 0
    i = 0
    p = 0
    phase_is_compute = True
    phase_end = time.perf_counter() + phase_duration

    while time.perf_counter() < end_time:
        if time.perf_counter() >= phase_end:
            # Switch phase
            phase_is_compute = not phase_is_compute
            phase_end = time.perf_counter() + phase_duration

        if phase_is_compute:
            # Compute phase
            for _ in range(1000):
                compute_total += i * i
                i += 1
        else:
            # Memory phase
            for _ in range(1000):
                p = chase[p]
            memory_total += p

    return compute_total + memory_total


def io_bound(duration_sec: float = DEFAULT_DURATION) -> int:
    """Simulated I/O-bound workload with periodic sleeps.

    This workload represents applications that spend most of their time waiting
    for I/O (network, disk). Neither P-cores nor E-cores help much here — the
    thread is blocked waiting for external events.

    Useful for training the model to recognize "don't care" workloads where
    placement doesn't matter for performance but does matter for power.

    Returns a counter of work units completed.
    """
    end_time = time.perf_counter() + duration_sec
    work_units = 0
    total = 0

    while time.perf_counter() < end_time:
        # Do a small burst of compute
        for i in range(1000):
            total += i * i
        work_units += 1

        # Simulate waiting for I/O
        time.sleep(0.01)  # 10ms sleep, like waiting for network or disk

    return work_units


# -----------------------------------------------------------------------------
# Workload registry
# -----------------------------------------------------------------------------
# Every workload — whether an in-process Python function or an external Windows
# executable — is described by the SAME dict shape below, so runner.py can
# handle them uniformly.
#
# Required keys per spec:
#   kind:                   'python' | 'external'
#   thread_counts:          list[int] of thread counts to sweep for this workload
#   expected_duration_sec:  rough runtime, used to size the SoC Watch window
#
# For kind == 'python':
#   func:                   callable(duration_sec) -> result (in-process)
#
# For kind == 'external':
#   exe:                    path (relative to collect/) to the .exe, or None
#                           if the args_builder returns a full argv starting
#                           with the interpreter (SQLite wrapper case).
#   args_builder:           callable(thread_count, duration_sec) -> list[str]
#                           of arguments (NOT including the exe itself).
#   input_files:            list of relative paths that must exist before the
#                           run starts (runner fails fast if missing).
#
# The Python micros keep thread_counts=[1] because they are single-threaded
# by design — they are the calibration control.
# -----------------------------------------------------------------------------

# -- Tunables set by collect/probe_env.py ------------------------------------
# These exist as named constants rather than literals inside the registry
# because each one is the answer to a specific probe, and getting one wrong
# costs a whole campaign.

# Probe 0.4. sample.y4m is 640x360 / 720 frames, so at --preset medium x264
# finishes in 1.5-3.5s and contributed 29 of v3's 2760 windows. A slower preset
# buys span without a new module or a pipe. The span still varies ~8x across
# thread counts, which the self-terminating exporter handles.
X264_PRESET = "veryslow"

# Probe 0.2. False encodes the file once at X264_PRESET. True switches to the
# duration-driven stdin feeder in workloads_x264.py.
#
# **Probe 0.2 PASSED 2026-10-01:** VTune attributes grandchild processes (both
# python.exe and 7z.exe appeared in the report). The feeder is now enabled.
X264_USE_FEEDER = True

# Probe 0.3. zstd's -i is a MINIMUM per-level evaluation time and -b3 runs
# both compression and decompression, so -i30 can mean a ~60s+ span. Cap the
# per-level time so the total lands near the 30s target.
ZSTD_ITER_CAP_SEC = 15


def _x264_args(thread_count: int, duration_sec: float) -> list:
    """Build x264's argv, either the single-encode or the feeder form."""
    if X264_USE_FEEDER:
        # Full argv starting with the interpreter, like the sqlite wrapper.
        return [
            sys.executable,
            "workloads_x264.py",
            "--threads", str(thread_count),
            "--duration", str(max(5, int(duration_sec))),
            "--preset", X264_PRESET,
            "--input", "workloads_data/sample.y4m",
            "--x264", "workloads_bin/x264.exe",
        ]
    return [
        "--threads", str(thread_count),
        "--preset", X264_PRESET,
        "-o", "NUL",
        "workloads_data/sample.y4m",
    ]


WORKLOADS = {
    # -- Python micro-benchmarks (calibration controls) -----------------------
    "compute": {
        "kind": "python",
        "func": compute_bound,
        "thread_counts": [1],
        "expected_duration_sec": DEFAULT_DURATION,
    },
    "memory": {
        "kind": "python",
        "func": memory_bound,
        "thread_counts": [1],
        "expected_duration_sec": DEFAULT_DURATION,
    },
    "mixed": {
        "kind": "python",
        "func": mixed_phases,
        "thread_counts": [1],
        "expected_duration_sec": DEFAULT_DURATION,
    },
    "io": {
        "kind": "python",
        "func": io_bound,
        "thread_counts": [1],
        "expected_duration_sec": DEFAULT_DURATION,
    },

    # -- Real applications ----------------------------------------------------
    # 7-Zip's built-in benchmark. `-mmt<N>` sets thread count. It runs a fixed
    # amount of work per iteration and prints a summary at the end; wall time
    # is roughly 30 s for the default work size.
    "7zip": {
        "kind": "external",
        "exe": "workloads_bin/7z.exe",
        "args_builder": lambda t, d: ["b", "-mmt" + str(t)],
        "thread_counts": [1, 4, 8, 16],
        "expected_duration_sec": 30.0,
        "input_files": [],
    },

    # zstd benchmark mode: -T<N> threads, -b3 uses compression level 3
    # (default), -i<sec> sets a per-level iteration time. The corpus is the
    # standard Silesia dataset.
    "zstd": {
        "kind": "external",
        "exe": "workloads_bin/zstd.exe",
        "args_builder": lambda t, d: [
            "-T" + str(t),
            "-b3",
            "-i" + str(max(3, min(int(d), ZSTD_ITER_CAP_SEC))),
            "workloads_data/silesia.tar",
        ],
        "thread_counts": [1, 4, 8, 16],
        "expected_duration_sec": 30.0,
        "input_files": ["workloads_data/silesia.tar"],
    },

    # x264 encoding. `--threads <N>` sets thread count; the preset and whether
    # a stdin feeder is used are decided by probes 0.4 and 0.2 -- see
    # X264_PRESET / X264_USE_FEEDER above. Output goes to NUL so there is no
    # disk contention. x264.exe is listed in input_files as well as exe so it
    # is still checked in the feeder form, where exe is None.
    "x264": {
        "kind": "external",
        "exe": None if X264_USE_FEEDER else "workloads_bin/x264.exe",
        "args_builder": _x264_args,
        "thread_counts": [1, 4, 8, 16],
        "expected_duration_sec": 30.0,
        "input_files": ["workloads_data/sample.y4m", "workloads_bin/x264.exe"],
    },

    # SQLite is driven by a small Python wrapper (workloads_sqlite.py) that
    # opens the pre-built DB, launches N worker threads, and each thread runs
    # mixed SELECT/UPDATE queries from a fixed pool until --duration elapses.
    # sqlite3.exe alone has no built-in benchmark and doesn't thread queries.
    "sqlite": {
        "kind": "external",
        "exe": None,  # argv builder emits a full command starting with python
        "args_builder": lambda t, d: [
            sys.executable,
            "workloads_sqlite.py",
            "--threads", str(t),
            "--duration", str(max(5, int(d))),
            "--db", "workloads_data/test.sqlite",
        ],
        "thread_counts": [1, 4, 8, 16],
        # Honours --duration directly, so this tracks the campaign's 30s.
        "expected_duration_sec": 30.0,
        "input_files": ["workloads_data/test.sqlite"],
    },
}


def run_workload(name: str, duration_sec: float = DEFAULT_DURATION) -> tuple:
    """Run a Python micro-benchmark and return (name, duration, result).

    This is the in-process entry point used by ``python workloads.py <name>`` —
    only supports ``kind == 'python'`` workloads. External workloads are
    launched by ``runner.py`` directly and never come through this path.

    Args:
        name: Key from WORKLOADS dict (must be a python-kind workload)
        duration_sec: How long to run

    Returns:
        Tuple of (name, actual_duration, result_value)

    Raises:
        KeyError: If workload name not found
        ValueError: If workload is not a python-kind workload
    """
    if name not in WORKLOADS:
        raise KeyError(f"Unknown workload: {name}. Available: {list(WORKLOADS.keys())}")

    spec = WORKLOADS[name]
    if spec["kind"] != "python":
        raise ValueError(
            f"Workload '{name}' is kind={spec['kind']!r}; only python-kind "
            "workloads can run in-process. Launch it via runner.py instead."
        )

    fn = spec["func"]
    start = time.perf_counter()
    result = fn(duration_sec)
    elapsed = time.perf_counter() - start

    return (name, elapsed, result)


def python_workload_names() -> list:
    """Return the names of the python-kind workloads (for the CLI choices)."""
    return [name for name, spec in WORKLOADS.items() if spec["kind"] == "python"]


# Workloads whose setup cost lands inside the profiled window unless it is
# forced to happen first. _get_chase() builds and shuffles a 4-million-element
# list -- seconds of allocation and random access that look nothing like the
# workload itself -- and it is called lazily from inside memory_bound and
# mixed_phases. In a 5-second run that was most of the run.
NEEDS_CHASE = ("memory", "mixed")


def prewarm(name: str) -> float:
    """Do a workload's setup up front and return how long it took.

    The setup still happens inside VTune's collection window -- VTune launches
    the process, so there is nowhere earlier to put it. The point is that its
    cost becomes a MEASURED offset the consumer can drop, instead of an
    unknown amount of allocation noise smeared across the first few bins.
    """
    start = time.perf_counter()
    if name in NEEDS_CHASE:
        _get_chase()
    return time.perf_counter() - start


def main():
    """CLI entry point for running workloads."""
    parser = argparse.ArgumentParser(
        description="Run micro-benchmarks for data collection."
    )
    # This CLI only runs the in-process Python micros. External workloads
    # (7zip, zstd, x264, sqlite) are launched by runner.py directly and would
    # blow up here — the choices list excludes them.
    parser.add_argument(
        "workload",
        choices=python_workload_names(),
        help="Which python-kind workload to run"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Duration in seconds (default: {DEFAULT_DURATION})"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output (for subprocess invocation)"
    )
    parser.add_argument(
        "--phase-file",
        type=str,
        default=None,
        help="Write a JSON file recording when setup finished and when the "
             "workload proper started and ended, as offsets from process "
             "start. analysis/windows.py drops the leading timeline bins that "
             "fall inside setup, which for 'memory' and 'mixed' is the "
             "multi-second build of the 4M-element pointer-chase array."
    )

    args = parser.parse_args()

    t0 = time.perf_counter()
    started_unix = time.time()
    setup_sec = prewarm(args.workload)
    workload_start = time.perf_counter() - t0

    name, elapsed, result = run_workload(args.workload, args.duration)
    workload_end = time.perf_counter() - t0

    if args.phase_file:
        phases = {
            "workload": name,
            "pid": os.getpid(),
            "process_start_unix": started_unix,
            "setup_sec": round(setup_sec, 4),
            # Offsets are from PROCESS start, not from VTune's collection
            # start; VTune launches the process, so its own startup sits
            # before offset 0. Treat these as a lower bound on how many
            # leading bins to drop.
            "setup_done_offset_sec": round(workload_start, 4),
            "workload_start_offset_sec": round(workload_start, 4),
            "workload_end_offset_sec": round(workload_end, 4),
            "duration_requested_sec": args.duration,
            "duration_actual_sec": round(elapsed, 4),
        }
        with open(args.phase_file, "w", encoding="utf-8") as f:
            json.dump(phases, f, indent=2)

    if not args.quiet:
        print(f"Workload: {name}")
        print(f"Setup:    {setup_sec:.3f}s")
        print(f"Duration: {elapsed:.3f}s (requested {args.duration}s)")
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
