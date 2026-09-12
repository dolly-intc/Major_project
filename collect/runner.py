"""
Data collection harness: orchestrates workload × core-type experiments.

This is the main script for Stage 1 data collection. It runs all combinations
of workloads and core types in randomized order, collecting hardware counters
(via VTune) and energy measurements (via SoC Watch) for each run.

Standard library only, so it runs on the restricted test machine.

Output structure:
    data/raw/<fingerprint>/
    ├── fingerprint.json           # Machine identity
    ├── manifest.json              # All runs with metadata
    ├── run_001_compute_p/
    │   ├── vtune_hw_events.csv    # Per-thread hardware counters
    │   ├── energy.csv             # Package power
    │   └── results.json           # Parsed + derived metrics
    ├── run_002_memory_e/
    │   └── ...
    └── noise_study/               # (optional) 30 runs of one config for variance
        ├── run_001/
        └── ...

Usage (elevated shell on test machine):
    # Full experiment: all workloads × all core types × 7 reps
    python runner.py

    # Quick test: single workload, single core type, 1 rep
    python runner.py --workloads compute --core-types p --reps 1

    # Noise study: 30 reps of one config for variance analysis
    python runner.py --noise-study --reps 30

    # Custom output location
    python runner.py --output C:\\data\\experiment_001
"""

import argparse
import ctypes
import json
import os
import random
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

# Windows API for process affinity
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
kernel32.SetProcessAffinityMask.restype = wintypes.BOOL


def pin_to_core_type(core_type: str, topology: dict) -> int:
    """Pin this process to a core of the specified type.

    Args:
        core_type: 'p' for P-core, 'e' for E-core, 'default' for no pinning
        topology: Result from get_topology()

    Returns:
        The CPU number pinned to, or -1 for default

    Raises:
        ValueError: If core_type is invalid
        ctypes.WinError: If SetProcessAffinityMask fails
    """
    if core_type == "default":
        # Clear affinity mask to allow all cores
        # On Windows, setting mask to 0 means "use all available"
        # But we need to get the process's original affinity first
        # For simplicity, just skip pinning
        return -1

    if core_type == "p":
        cores = topology["p_cores"]
    elif core_type == "e":
        cores = topology["e_cores"]
    else:
        raise ValueError(f"Invalid core_type: {core_type}. Use 'p', 'e', or 'default'")

    if not cores:
        raise ValueError(f"No {core_type}-cores found in topology")

    # Pin to the first core of the requested type
    cpu = cores[0]
    mask = 1 << cpu

    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
        raise ctypes.WinError(ctypes.get_last_error())

    return cpu


def reset_affinity():
    """Reset process affinity to allow all cores."""
    # Get system affinity mask
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    kernel32.GetProcessAffinityMask(
        kernel32.GetCurrentProcess(),
        ctypes.byref(process_mask),
        ctypes.byref(system_mask)
    )
    kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), system_mask.value)


def run_single(
    workload: str,
    core_type: str,
    duration_sec: float,
    output_dir: str,
    topology: dict,
    collect_energy: bool = True,
) -> dict:
    """Run a single experiment: one workload, one core type.

    Args:
        workload: Name from WORKLOADS dict
        core_type: 'p', 'e', or 'default'
        duration_sec: Workload duration
        output_dir: Where to store results for this run
        topology: Core topology dict
        collect_energy: Whether to run SoC Watch (needs separate elevation)

    Returns:
        dict with run results:
            workload: workload name
            core_type: 'p', 'e', or 'default'
            cpu_pinned: CPU number or -1
            duration_sec: requested duration
            actual_duration_sec: measured duration
            vtune: parsed VTune results for primary thread
            energy: parsed SoC Watch results (if collected)
            ipc: computed IPC
            timestamp: ISO format timestamp
    """
    os.makedirs(output_dir, exist_ok=True)

    result = {
        "workload": workload,
        "core_type": core_type,
        "cpu_pinned": -1,
        "duration_sec": duration_sec,
        "actual_duration_sec": None,
        "timestamp": datetime.now().isoformat(),
        "vtune": None,
        "energy": None,
        "ipc": None,
    }

    # Pin to requested core type
    try:
        result["cpu_pinned"] = pin_to_core_type(core_type, topology)
    except Exception as e:
        result["pin_error"] = str(e)

    # Build the workload command
    # We run workloads.py as a subprocess so VTune profiles the child
    workload_cmd = [
        sys.executable,
        str(Path(__file__).parent / "workloads.py"),
        workload,
        "--duration", str(duration_sec),
        "--quiet",
    ]

    # VTune result directory
    vtune_dir = os.path.join(output_dir, "vtune_result")
    vtune_csv = os.path.join(output_dir, "vtune_hw_events.csv")

    # Energy collection directory
    energy_dir = os.path.join(output_dir, "energy")
    energy_proc = None

    # Start energy collection in background BEFORE VTune (if requested)
    # This ensures energy is measured during the entire workload
    if collect_energy:
        try:
            # Add 2 second margin: 1s before VTune starts + 1s after it ends
            energy_duration = int(duration_sec) + 2
            energy_proc = energy.collect_background(energy_duration, energy_dir)
            time.sleep(0.5)  # Brief pause to let SoC Watch initialize
        except Exception as e:
            result["energy_error"] = f"Failed to start: {e}"
            energy_proc = None

    # Run VTune collection (this also runs the workload)
    start_time = time.perf_counter()
    try:
        vtune.collect(workload_cmd, vtune_dir)
        vtune.export_csv(vtune_dir, vtune_csv)

        # Parse VTune results
        rows = vtune.parse_csv(vtune_csv)
        rows = vtune.compute_derived_metrics(rows)
        primary = vtune.get_primary_thread(rows)

        result["vtune"] = primary
        result["ipc"] = primary.get("ipc")
        result["all_threads"] = len(rows)

    except Exception as e:
        result["vtune_error"] = str(e)

    result["actual_duration_sec"] = time.perf_counter() - start_time

    # Wait for energy collection to complete and parse results
    if energy_proc is not None:
        try:
            energy_proc.wait(timeout=duration_sec + 10)  # Allow extra time
            if energy_proc.returncode == 0:
                result["energy"] = energy.parse_output_dir(energy_dir)
            else:
                result["energy_error"] = f"SoC Watch exited with code {energy_proc.returncode}"
        except Exception as e:
            result["energy_error"] = f"Failed to collect: {e}"
            try:
                energy_proc.kill()
            except Exception:
                pass

    # Reset affinity for next run
    try:
        reset_affinity()
    except Exception:
        pass

    # Save results to JSON
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def run_experiment(
    workloads: list,
    core_types: list,
    reps: int,
    output_base: str,
    duration_sec: float = DEFAULT_DURATION,
    randomize: bool = True,
    collect_energy: bool = True,
    noise_study: bool = False,
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

    Returns:
        dict with experiment summary:
            fingerprint: machine fingerprint
            runs: list of all run results
            total_time_sec: wall-clock time
            errors: count of failed runs
    """
    # Get machine fingerprint
    fp = get_fingerprint()
    fp_id = fingerprint_id(fp)

    # Create output directory
    output_dir = Path(output_base) / fp_id if not noise_study else Path(output_base) / fp_id / "noise_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save fingerprint
    save_fingerprint(fp, str(output_dir / "fingerprint.json"))

    # Get core topology
    topology = get_topology()

    # Build run list
    if noise_study:
        # Noise study: single config, many reps
        workload = workloads[0] if workloads else "compute"
        core_type = core_types[0] if core_types else "p"
        runs_planned = [(workload, core_type, i) for i in range(reps)]
    else:
        # Full experiment: all combinations
        runs_planned = [
            (w, ct, rep)
            for w in workloads
            for ct in core_types
            for rep in range(reps)
        ]

    if randomize:
        random.shuffle(runs_planned)

    print(f"Machine: {fp_id}")
    print(f"Topology: {len(topology['p_cores'])} P-cores, {len(topology['e_cores'])} E-cores")
    print(f"Planned runs: {len(runs_planned)}")
    print(f"Output: {output_dir}")
    print()

    # Run experiments
    results = []
    errors = 0
    start_time = time.perf_counter()

    for i, (workload, core_type, rep) in enumerate(runs_planned):
        run_id = f"run_{i+1:03d}_{workload}_{core_type}"
        run_dir = output_dir / run_id

        print(f"[{i+1}/{len(runs_planned)}] {workload} on {core_type}-core... ", end="", flush=True)

        try:
            result = run_single(
                workload=workload,
                core_type=core_type,
                duration_sec=duration_sec,
                output_dir=str(run_dir),
                topology=topology,
                collect_energy=collect_energy,
            )
            results.append(result)

            ipc = result.get("ipc")
            if ipc:
                print(f"IPC={ipc:.2f}")
            else:
                print("done (no IPC)")

            if "vtune_error" in result or "energy_error" in result:
                errors += 1

        except Exception as e:
            print(f"FAILED: {e}")
            errors += 1
            results.append({"workload": workload, "core_type": core_type, "error": str(e)})

    total_time = time.perf_counter() - start_time

    # Save manifest
    manifest = {
        "fingerprint": fp,
        "fingerprint_id": fp_id,
        "topology": topology,
        "workloads": workloads,
        "core_types": core_types,
        "reps": reps,
        "duration_sec": duration_sec,
        "randomized": randomize,
        "noise_study": noise_study,
        "total_runs": len(results),
        "errors": errors,
        "total_time_sec": total_time,
        "timestamp": datetime.now().isoformat(),
        "runs": [
            {
                "workload": r.get("workload"),
                "core_type": r.get("core_type"),
                "ipc": r.get("ipc"),
                "energy_mj": r.get("energy", {}).get("total_energy_mj") if r.get("energy") else None,
            }
            for r in results
        ],
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print()
    print(f"Complete: {len(results)} runs in {total_time:.1f}s")
    print(f"Errors: {errors}")
    print(f"Manifest: {manifest_path}")

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

    args = parser.parse_args()

    # Check for VTune
    if not os.path.exists(vtune.VTUNE_PATH):
        print(f"ERROR: VTune not found at {vtune.VTUNE_PATH}")
        print("This script must run on the test machine with VTune installed.")
        sys.exit(1)

    # Check for administrator
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        is_admin = False

    if not is_admin:
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
    )


if __name__ == "__main__":
    main()
