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
import random
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


# Registry of available workloads
WORKLOADS = {
    'compute': compute_bound,
    'memory': memory_bound,
    'mixed': mixed_phases,
    'io': io_bound,
}


def run_workload(name: str, duration_sec: float = DEFAULT_DURATION) -> tuple:
    """Run a named workload and return (name, duration, result).

    Args:
        name: Key from WORKLOADS dict
        duration_sec: How long to run

    Returns:
        Tuple of (name, actual_duration, result_value)

    Raises:
        KeyError: If workload name not found
    """
    if name not in WORKLOADS:
        raise KeyError(f"Unknown workload: {name}. Available: {list(WORKLOADS.keys())}")

    fn = WORKLOADS[name]
    start = time.perf_counter()
    result = fn(duration_sec)
    elapsed = time.perf_counter() - start

    return (name, elapsed, result)


def main():
    """CLI entry point for running workloads."""
    parser = argparse.ArgumentParser(
        description="Run micro-benchmarks for data collection."
    )
    parser.add_argument(
        "workload",
        choices=list(WORKLOADS.keys()),
        help="Which workload to run"
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

    args = parser.parse_args()

    name, elapsed, result = run_workload(args.workload, args.duration)

    if not args.quiet:
        print(f"Workload: {name}")
        print(f"Duration: {elapsed:.3f}s (requested {args.duration}s)")
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
