"""
Intel VTune Profiler wrapper for hardware event collection.

Drives VTune from Python to collect per-thread hardware performance counters,
then exports and parses the results. Standard library only, so it runs on the
restricted test machine.

The harness uses the uarch-exploration analysis type, which collects the
Top-Down Microarchitecture metrics (Front-End Bound, Bad Speculation,
Back-End Bound, Retiring) along with raw hardware event counts.

IPC (instructions per cycle) is not a VTune column — it is computed by this
module from INST_RETIRED.ANY / CPU_CLK_UNHALTED.THREAD.

Usage:
    from vtune import collect, export_csv, parse_csv, compute_derived_metrics

    # Run collection (needs test machine with VTune + elevated shell)
    collect(["python", "workloads.py", "compute"], "r001")

    # Export to CSV
    export_csv("r001", "r001_threads.csv")

    # Parse (works on dev machine with fixture)
    rows = parse_csv("r001_threads.csv")
    rows = compute_derived_metrics(rows)
    print(rows[0]["ipc"])
"""

import csv
import os
import re
import subprocess
from pathlib import Path

# VTune installation path on the test machine.
# Verified 2026-08-31: runs by absolute path, no setvars.bat needed.
VTUNE_PATH = r"C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe"

# Prefix that VTune adds to all hardware event counter columns.
# Must be stripped to get clean event names like "INST_RETIRED.ANY".
COUNTER_PREFIX = "Hardware Event Count:"

# Key events for IPC calculation
EVENT_INSTRUCTIONS = "INST_RETIRED.ANY"
EVENT_CYCLES = "CPU_CLK_UNHALTED.THREAD"


def collect(
    workload_cmd: list,
    result_dir: str,
    analysis_type: str = "uarch-exploration",
    duration_sec: int = None,
    sampling_interval_ms: float = None,
) -> subprocess.CompletedProcess:
    """Run VTune collection on a workload.

    Args:
        workload_cmd: Command to run, e.g. ["python", "workloads.py", "compute"]
        result_dir: Directory to store VTune results (will be created)
        analysis_type: VTune analysis type (default: uarch-exploration)
        duration_sec: Optional collection duration limit
        sampling_interval_ms: Optional sampling interval (0.01-1000, default 1)

    Returns:
        CompletedProcess from subprocess.run

    Raises:
        subprocess.CalledProcessError if VTune fails
        FileNotFoundError if VTune is not installed

    Note: Requires Administrator privileges on Windows.
    """
    if not os.path.exists(VTUNE_PATH):
        raise FileNotFoundError(
            f"VTune not found at {VTUNE_PATH}. "
            "Is Intel oneAPI/VTune installed on this machine?"
        )

    cmd = [VTUNE_PATH, "-collect", analysis_type, "-r", result_dir]

    # Disable stack collection — we only need hardware counters, not call stacks.
    # This suppresses "Cannot locate debugging information" warnings for system DLLs.
    cmd.extend(["-knob", "enable-stack-collection=false"])

    if duration_sec is not None:
        cmd.extend(["-duration", str(duration_sec)])

    if sampling_interval_ms is not None:
        cmd.extend(["-knob", f"sampling-interval={sampling_interval_ms}"])

    # The workload command follows "--"
    cmd.append("--")
    cmd.extend(workload_cmd)

    return subprocess.run(cmd, check=True)


def export_csv(
    result_dir: str,
    output_path: str,
    group_by: str = "thread",
) -> subprocess.CompletedProcess:
    """Export VTune results to CSV.

    Args:
        result_dir: VTune result directory from collect()
        output_path: Path to write CSV file
        group_by: Grouping for the report (default: thread)

    Returns:
        CompletedProcess from subprocess.run

    Raises:
        subprocess.CalledProcessError if VTune fails
    """
    if not os.path.exists(VTUNE_PATH):
        raise FileNotFoundError(f"VTune not found at {VTUNE_PATH}")

    cmd = [
        VTUNE_PATH,
        "-report", "hw-events",
        "-r", result_dir,
        "-group-by", group_by,
        "-format=csv",
        "-csv-delimiter=comma",
        "-report-output", output_path,
    ]

    return subprocess.run(cmd, check=True)


def parse_csv(csv_path: str) -> list:
    """Parse VTune hw-events CSV into a list of per-thread counter dicts.

    The CSV format (from VTune 2026.4.0):
    - First column: "Thread" with values like "Thread (TID: 4044)"
    - Counter columns: "Hardware Event Count:EVENT_NAME"
    - Last two columns: "PID", "TID"

    Returns:
        List of dicts, one per thread:
        [
            {
                'thread': 'Thread (TID: 4044)',
                'tid': 4044,
                'pid': 5520,
                'INST_RETIRED.ANY': 150528000,
                'CPU_CLK_UNHALTED.THREAD': 98304000,
                ...
            },
            ...
        ]

    Raises:
        FileNotFoundError if csv_path doesn't exist
        ValueError if CSV format is unexpected
    """
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            parsed = {}

            for key, value in row.items():
                # Skip empty keys (can happen with trailing commas)
                if not key:
                    continue

                # Clean the column name
                clean_key = key.strip()

                # Strip the "Hardware Event Count:" prefix from counter columns
                if clean_key.startswith(COUNTER_PREFIX):
                    clean_key = clean_key[len(COUNTER_PREFIX):]

                # Parse the value
                value = value.strip()

                if clean_key == "Thread":
                    # Keep the original thread string
                    parsed["thread"] = value
                    # Extract TID from "Thread (TID: 4044)"
                    tid_match = re.search(r"TID:\s*(\d+)", value)
                    if tid_match:
                        parsed["tid"] = int(tid_match.group(1))
                elif clean_key in ("PID", "TID"):
                    # These are integer columns
                    try:
                        parsed[clean_key.lower()] = int(value)
                    except ValueError:
                        parsed[clean_key.lower()] = None
                else:
                    # Hardware event counts are integers
                    try:
                        parsed[clean_key] = int(value)
                    except ValueError:
                        # Some columns might be empty or non-numeric
                        parsed[clean_key] = value if value else 0

            if parsed:
                rows.append(parsed)

    return rows


def compute_derived_metrics(rows: list) -> list:
    """Add derived metrics (IPC, cache miss rates, etc.) to parsed rows.

    Modifies rows in place and returns them.

    Added fields:
        ipc: Instructions per cycle (INST_RETIRED.ANY / CPU_CLK_UNHALTED.THREAD)
        l1_miss_rate: L1 cache miss rate (if counters available)
        l3_miss_rate: L3 cache miss rate (if counters available)

    Returns:
        The same list with additional fields added to each dict.
    """
    for row in rows:
        instructions = row.get(EVENT_INSTRUCTIONS, 0)
        cycles = row.get(EVENT_CYCLES, 0)

        # IPC
        if cycles > 0:
            row["ipc"] = instructions / cycles
        else:
            row["ipc"] = 0.0

        # L1 miss rate (if we have load counts)
        l1_miss = row.get("MEM_LOAD_RETIRED.L1_MISS", 0)
        l1_hit = row.get("MEM_LOAD_RETIRED.L1_HIT", 0)
        total_l1 = l1_miss + l1_hit
        if total_l1 > 0:
            row["l1_miss_rate"] = l1_miss / total_l1
        else:
            row["l1_miss_rate"] = None

        # L3 miss rate
        l3_miss = row.get("MEM_LOAD_RETIRED.L3_MISS", 0)
        l3_hit = row.get("MEM_LOAD_RETIRED.L3_HIT", 0)
        total_l3 = l3_miss + l3_hit
        if total_l3 > 0:
            row["l3_miss_rate"] = l3_miss / total_l3
        else:
            row["l3_miss_rate"] = None

    return rows


def get_primary_thread(rows: list) -> dict:
    """Return the thread with the most instructions (usually the main workload thread).

    Useful when a workload spawns multiple threads but you want to analyze just
    the primary one.
    """
    if not rows:
        return {}

    return max(rows, key=lambda r: r.get(EVENT_INSTRUCTIONS, 0))


if __name__ == "__main__":
    import sys

    # Quick test: parse fixture if it exists, otherwise show usage
    fixture_path = Path(__file__).parent.parent / "tests" / "fixtures" / "vtune_hw_events.csv"

    if fixture_path.exists():
        print(f"Parsing fixture: {fixture_path}")
        rows = parse_csv(str(fixture_path))
        rows = compute_derived_metrics(rows)

        print(f"\nFound {len(rows)} thread(s):")
        for row in rows:
            print(f"  TID {row.get('tid')}: IPC={row.get('ipc', 0):.2f}, "
                  f"instructions={row.get(EVENT_INSTRUCTIONS, 0):,}")

        primary = get_primary_thread(rows)
        print(f"\nPrimary thread: TID {primary.get('tid')}")
    else:
        print("Usage:")
        print("  On test machine (elevated shell):")
        print('    python vtune.py collect "python workloads.py compute" r001')
        print("    python vtune.py export r001 r001_threads.csv")
        print()
        print("  On dev machine:")
        print(f"    Copy {fixture_path} from test machine")
        print("    python vtune.py  # parses fixture")
