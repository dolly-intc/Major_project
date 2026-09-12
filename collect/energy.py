"""
Intel SoC Watch wrapper for package energy measurement.

Drives SoC Watch from Python to collect package power during workload execution.
Standard library only, so it runs on the restricted test machine.

SoC Watch is bundled with Intel VTune and measures CPU package energy using the
Intel RAPL (Running Average Power Limit) interface. The pkg-pwr feature reports:
  - Average power (mW)
  - Total energy (mJ)

Usage:
    from energy import collect, parse_csv

    # Run collection with a workload (needs test machine + elevated shell)
    collect(duration_sec=5, output_dir="run_001", workload_cmd=["python", "workloads.py", "compute"])

    # Parse results
    result = parse_csv("run_001/energy.csv")
    print(f"Average power: {result['avg_power_mw']} mW")
    print(f"Total energy: {result['total_energy_mj']} mJ")
"""

import csv
import os
import subprocess
from pathlib import Path

# SoC Watch installation path (inside VTune folder, under 64/ subdirectory)
# Verified 2026-08-31 on test machine
SOCWATCH_PATH = r"C:\Program Files (x86)\Intel\oneAPI\vtune\latest\socwatch\64\socwatch.exe"

# The feature we use: package power via RAPL
FEATURE = "pkg-pwr"


def collect(
    duration_sec: int,
    output_dir: str,
    sample_interval_ms: int = 100,
) -> subprocess.CompletedProcess:
    """Run SoC Watch pkg-pwr collection for a fixed duration.

    Args:
        duration_sec: Collection duration in seconds
        output_dir: Directory to store output files (created if needed)
        sample_interval_ms: Sampling interval in milliseconds (default 100)

    Returns:
        CompletedProcess from subprocess.run

    Raises:
        subprocess.CalledProcessError if SoC Watch fails
        FileNotFoundError if SoC Watch is not installed

    Note: Requires Administrator privileges on Windows.

    Output files:
        <output_dir>/energy_SoCWatchOutput.csv - main output with power data
    """
    if not os.path.exists(SOCWATCH_PATH):
        raise FileNotFoundError(
            f"SoC Watch not found at {SOCWATCH_PATH}. "
            "Is Intel VTune installed with SoC Watch bundled?"
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build command
    # -f pkg-pwr: collect package power
    # -m: output as CSV (machine-readable)
    # -t: duration in seconds
    # -o: output file prefix
    output_prefix = os.path.join(output_dir, "energy")

    cmd = [
        SOCWATCH_PATH,
        "-f", FEATURE,
        "-m",  # CSV output
        "-t", str(duration_sec),
        "-o", output_prefix,
    ]

    return subprocess.run(cmd, check=True)


def collect_background(
    duration_sec: int,
    output_dir: str,
) -> subprocess.Popen:
    """Start SoC Watch collection in the background.

    Use this to collect energy in parallel with a workload. The caller is
    responsible for waiting on the returned Popen object.

    Args:
        duration_sec: Collection duration in seconds
        output_dir: Directory to store output files

    Returns:
        Popen object - call .wait() to block until collection completes

    Example:
        proc = collect_background(10, "run_001/energy")
        run_my_workload()  # runs in parallel
        proc.wait()  # wait for energy collection to finish
        result = parse_output_dir("run_001/energy")
    """
    if not os.path.exists(SOCWATCH_PATH):
        raise FileNotFoundError(
            f"SoC Watch not found at {SOCWATCH_PATH}. "
            "Is Intel VTune installed with SoC Watch bundled?"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.join(output_dir, "energy")

    cmd = [
        SOCWATCH_PATH,
        "-f", FEATURE,
        "-m",
        "-t", str(duration_sec),
        "-o", output_prefix,
    ]

    # Start in background, suppress output
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def parse_csv(csv_path: str) -> dict:
    """Parse SoC Watch output CSV for package energy.

    Expected format (verified 2026-08-31):
        Component    , Metric Type, Average Rate (mW), Total (mJ)
        CPU/Package_0, Power      , 5311.30          , 53611.08

    Args:
        csv_path: Path to SoC Watch CSV output file

    Returns:
        dict with keys:
            avg_power_mw: Average power in milliwatts
            total_energy_mj: Total energy in millijoules
            component: Component name (e.g., "CPU/Package_0")

    Raises:
        FileNotFoundError if csv_path doesn't exist
        ValueError if CSV format is unexpected
    """
    # Try to find the CSV file - SoC Watch adds a suffix
    csv_path = Path(csv_path)

    # SoC Watch may create files like energy.csv or SoCWatchOutput.csv
    candidates = [
        csv_path,
        csv_path.parent / "SoCWatchOutput.csv",
        csv_path.parent / f"{csv_path.stem}.csv",
        csv_path.parent / f"{csv_path.stem}_SoCWatchOutput.csv",
    ]

    actual_path = None
    for candidate in candidates:
        if candidate.exists():
            actual_path = candidate
            break

    if actual_path is None:
        raise FileNotFoundError(
            f"SoC Watch output not found. Tried: {[str(c) for c in candidates]}"
        )

    result = {
        "avg_power_mw": None,
        "total_energy_mj": None,
        "component": None,
        "source_file": str(actual_path),
    }

    with open(actual_path, "r", encoding="utf-8") as f:
        # SoC Watch CSV may have header rows before the data
        reader = csv.reader(f)

        for row in reader:
            if not row or len(row) < 4:
                continue

            # Strip whitespace from all cells
            row = [cell.strip() for cell in row]

            # Look for package power row
            # Format: Component, Metric Type, Average Rate (mW), Total (mJ)
            if "Package" in row[0] and "Power" in row[1]:
                result["component"] = row[0]
                try:
                    result["avg_power_mw"] = float(row[2])
                except (ValueError, IndexError):
                    pass
                try:
                    result["total_energy_mj"] = float(row[3])
                except (ValueError, IndexError):
                    pass
                break

    if result["avg_power_mw"] is None:
        raise ValueError(
            f"Could not find package power data in {actual_path}. "
            "Expected a row with 'Package' in Component and 'Power' in Metric Type."
        )

    return result


def parse_output_dir(output_dir: str) -> dict:
    """Parse SoC Watch output from a collection directory.

    Convenience wrapper that finds the CSV in the output directory.

    Args:
        output_dir: Directory passed to collect()

    Returns:
        Same as parse_csv()
    """
    output_dir = Path(output_dir)

    # SoC Watch creates output files with the prefix we specified
    # Look for any CSV in the directory
    csv_files = list(output_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {output_dir}")

    # Prefer energy*.csv (our prefix) over others
    for csv_file in csv_files:
        if csv_file.stem.startswith("energy"):
            return parse_csv(str(csv_file))

    # Fall back to first CSV found
    return parse_csv(str(csv_files[0]))


def get_power_summary(result: dict) -> str:
    """Format a human-readable power summary.

    Args:
        result: Output from parse_csv() or parse_output_dir()

    Returns:
        Formatted string like "5311 mW avg, 53611 mJ total"
    """
    avg = result.get("avg_power_mw")
    total = result.get("total_energy_mj")

    parts = []
    if avg is not None:
        parts.append(f"{avg:.0f} mW avg")
    if total is not None:
        parts.append(f"{total:.0f} mJ total")

    return ", ".join(parts) if parts else "No power data"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python energy.py collect <duration_sec> <output_dir> [workload_cmd...]")
        print("  python energy.py parse <output_dir>")
        print()
        print("Examples:")
        print("  python energy.py collect 5 run_001 python workloads.py compute")
        print("  python energy.py parse run_001")
        sys.exit(1)

    command = sys.argv[1]

    if command == "collect":
        if len(sys.argv) < 4:
            print("Usage: python energy.py collect <duration_sec> <output_dir> [workload_cmd...]")
            sys.exit(1)

        duration = int(sys.argv[2])
        output_dir = sys.argv[3]
        workload = sys.argv[4:] if len(sys.argv) > 4 else None

        print(f"Collecting for {duration}s into {output_dir}")
        if workload:
            print(f"Running workload: {' '.join(workload)}")

        collect(duration, output_dir, workload)
        print("Collection complete.")

    elif command == "parse":
        if len(sys.argv) < 3:
            print("Usage: python energy.py parse <output_dir>")
            sys.exit(1)

        output_dir = sys.argv[2]
        result = parse_output_dir(output_dir)
        print(f"Component: {result['component']}")
        print(f"Average power: {result['avg_power_mw']:.2f} mW")
        print(f"Total energy: {result['total_energy_mj']:.2f} mJ")
        print(f"Source: {result['source_file']}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
