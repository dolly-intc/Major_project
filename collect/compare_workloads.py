"""
Core-type sensitivity test: does the P-core advantage depend on workload type?

Runs two contrasting workloads on a P-core and an E-core and reports the
E/P time ratio for each.

  compute-bound  -- tight arithmetic loop, stays in registers/L1.
                    Should show a LARGE P-core advantage.
  memory-bound   -- dependent random access over a working set far larger
                    than the 36 MB L3 cache. The core spends most of its
                    time waiting on DRAM, so the P-core's extra execution
                    resources sit idle. Should show a SMALL advantage.

The gap between those two ratios is the signal the ML models learn to
predict. If it exists, workload-aware placement has something to work with.

Standard library only -- runs on the restricted test machine.
"""

import ctypes
import random
import statistics
import time
from ctypes import wintypes

from core_topology import get_topology

REPS = 7
CHASE_ELEMENTS = 4_000_000   # ~150 MB working set, far exceeds 36 MB L3
CHASE_STEPS = 1_000_000
COMPUTE_ITERATIONS = 5_000_000

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
kernel32.SetProcessAffinityMask.restype = wintypes.BOOL


def pin_to(cpu):
    """Restrict this process to a single logical CPU."""
    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 1 << cpu):
        raise ctypes.WinError(ctypes.get_last_error())


def compute_workload():
    total = 0
    for i in range(COMPUTE_ITERATIONS):
        total += i * i
    return total


def build_chase(n=CHASE_ELEMENTS):
    """A random permutation. Following p = chase[p] walks a cycle of
    dependent, unpredictable accesses, which defeats hardware prefetching."""
    indices = list(range(n))
    random.shuffle(indices)
    return indices


def memory_workload(chase):
    p = 0
    for _ in range(CHASE_STEPS):
        p = chase[p]
    return p


def time_median(fn, reps=REPS):
    fn()  # warm up: page in memory, settle turbo
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main():
    topo = get_topology()
    p_core = topo["p_cores"][0]
    e_core = topo["e_cores"][0]

    print(f"P-cores: {topo['p_cores']}")
    print(f"E-cores: {topo['e_cores']}")
    print(f"\nComparing CPU {p_core} (P) against CPU {e_core} (E), "
          f"median of {REPS} runs\n")

    print("Building memory working set...", flush=True)
    chase = build_chase()

    workloads = [
        ("compute-bound", compute_workload),
        ("memory-bound", lambda: memory_workload(chase)),
    ]

    ratios = {}
    for name, fn in workloads:
        pin_to(p_core)
        p_time = time_median(fn)
        pin_to(e_core)
        e_time = time_median(fn)
        ratios[name] = e_time / p_time
        print(f"{name:>14s}   P={p_time:6.3f}s   E={e_time:6.3f}s   "
              f"E/P = {ratios[name]:.2f}x")

    spread = ratios["compute-bound"] - ratios["memory-bound"]
    print(f"\nSensitivity spread: {spread:+.2f}x")
    if spread > 0.10:
        print("Core-type sensitivity varies by workload -- there is a signal to learn.")
    else:
        print("Little variation detected. Re-run with the RDP session disconnected "
              "before drawing conclusions.")


if __name__ == "__main__":
    main()
