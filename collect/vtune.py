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
import math
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
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

# Bumped whenever the timeline CSV's columns or the summary dict's keys change,
# so the loader and runner.py's --resume predicate can reject stale runs
# instead of silently mixing two schemas in one dataset.
#   1 -- original: caller-supplied duration drove the bin list (v1-v3 data)
#   2 -- self-terminating: open-ended bins + empty/fail streak stops
TIMELINE_SCHEMA_VERSION = 2


def collect(
    workload_cmd: list,
    result_dir: str,
    analysis_type: str = "uarch-exploration",
    duration_sec: int = None,
    sampling_interval_ms: float = None,
    cwd: str = None,
) -> subprocess.CompletedProcess:
    """Run VTune collection on a workload.

    Args:
        workload_cmd: Command to run, e.g. ["python", "workloads.py", "compute"]
        result_dir: Directory to store VTune results (will be created)
        analysis_type: VTune analysis type (default: uarch-exploration)
        duration_sec: Optional collection duration limit
        sampling_interval_ms: Optional sampling interval (0.01-1000, default 1)
        cwd: Working directory for the VTune subprocess (and therefore for the
             workload it spawns). Relative paths in workload_cmd -- e.g. the
             external workloads' "workloads_data/silesia.tar" -- resolve
             against this directory. Defaults to the caller's cwd, which is
             usually wrong when runner.py is invoked from the repo root; pass
             Path(__file__).parent from collect/ code.

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

    return subprocess.run(cmd, check=True, cwd=cwd)


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


def _slice_one_bin(
    result_dir: str,
    scratch: str,
    index: int,
    bin_size_sec: float,
    group_by: str,
    slice_timeout_sec: float,
) -> dict:
    """Export one time bucket. Retries once, then gives up.

    Returns a dict with:
        index, t0, t1
        ok:       True if VTune exited 0 and we parsed its output (possibly
                  to zero rows -- an empty bucket is a legitimate result)
        rows:     parsed + derived rows, [] when the bucket was empty
        attempts: 1 or 2
        error:    None on success, else a short reason

    The ok/rows split is the important part. A bucket VTune reported as empty
    and a bucket VTune crashed on look identical from the caller's side unless
    they are distinguished here, and conflating them lets a burst of transient
    crashes masquerade as "the workload ended" -- silently truncating a run.
    """
    t0 = round(index * bin_size_sec, 6)
    t1 = round(t0 + bin_size_sec, 6)
    slice_csv = os.path.join(scratch, f"slice_{index:05d}.csv")
    cmd = [
        VTUNE_PATH,
        "-report", "hw-events",
        "-r", result_dir,
        "-group-by", group_by,
        "-time-filter", f"{t0}:{t1}",
        "-format=csv",
        "-csv-delimiter=comma",
        "-report-output", slice_csv,
    ]

    error = None
    for attempt in (1, 2):
        # A partial file from the previous attempt must never be re-parsed.
        try:
            if os.path.exists(slice_csv):
                os.remove(slice_csv)
        except OSError:
            pass

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=slice_timeout_sec,
            )
        except subprocess.TimeoutExpired:
            error = f"timeout after {slice_timeout_sec}s"
            continue
        except OSError as e:
            error = f"spawn failed: {e}"
            continue

        if proc.returncode != 0:
            # 3221225477 == 0xC0000005 ACCESS_VIOLATION, which VTune raises on
            # some out-of-range -time-filter windows. Retrying is cheap and
            # sometimes works; a genuine out-of-range bucket fails twice and
            # the empty-streak stop is what ends the run.
            error = f"exit {proc.returncode}"
            continue

        if not os.path.exists(slice_csv):
            # Exit 0 with no file: VTune had nothing to report for this
            # window. That is an empty bucket, not a failure.
            return {"index": index, "t0": t0, "t1": t1, "ok": True,
                    "rows": [], "attempts": attempt, "error": None}

        try:
            rows = compute_derived_metrics(parse_csv(slice_csv))
        except (OSError, csv.Error, ValueError) as e:
            error = f"parse failed: {e}"
            continue
        finally:
            try:
                os.remove(slice_csv)
            except OSError:
                pass

        return {"index": index, "t0": t0, "t1": t1, "ok": True,
                "rows": rows, "attempts": attempt, "error": None}

    return {"index": index, "t0": t0, "t1": t1, "ok": False,
            "rows": [], "attempts": 2, "error": error}


def export_time_sliced_csv(
    result_dir: str,
    output_path: str,
    duration_sec: float = None,
    bin_size_sec: float = 0.5,
    group_by: str = "thread",
    max_bins: int = 120,
    empty_streak_stop: int = 4,
    fail_streak_abort: int = 8,
    slice_timeout_sec: float = 60.0,
    max_workers: int = 1,
) -> dict:
    """Export a per-thread × per-time-bucket hardware counter time series.

    Why this exists:
        VTune's ``timeline`` report only surfaces a small set of pre-aggregated
        metrics (CPUTime, TotalLatency) and doesn't expose raw hardware event
        counts. To get counters over time we call ``hw-events`` repeatedly
        with ``-time-filter <t0>:<t1>``, once per bucket, and stitch the
        outputs into one long-format CSV.

    How it decides when to stop:
        There is no reliable way to ask a finalized result how long the
        workload actually ran -- and the obvious proxy, the wall time of the
        ``collect`` call, is dominated by VTune's own startup and finalize
        (it overestimated by 3-10x on the v3 dataset, so ~85% of all slice
        calls were spent on buckets past the end of the workload).

        So this walks buckets outward from t=0 and stops when the data says
        to: ``empty_streak_stop`` consecutive buckets in which *no thread at
        all* reported. Across all 168 timeline CSVs collected so far, no run
        has a single run-level gap, so 4 is generous. ``max_bins`` is the
        backstop.

    Output CSV columns (all counter columns have the ``Hardware Event Count:``
    prefix stripped, matching :func:`parse_csv`):
        ``bin_start_sec, bin_end_sec, thread, tid, pid,
        INST_RETIRED.ANY, CPU_CLK_UNHALTED.THREAD, ... (all counters),
        ipc, l1_miss_rate, l3_miss_rate``

    One row per (bin, thread) pair. Threads with no activity in a bin are
    dropped by VTune itself; we don't inject zero rows.

    Args:
        result_dir: VTune result directory from collect().
        output_path: Path to write the stitched long-format CSV.
        duration_sec: Advisory upper bound on the workload length, in seconds.
                      When given, it caps ``max_bins``; it never sets the
                      bucket count on its own. Pass None (preferred) to let
                      the empty-streak stop find the end.
        bin_size_sec: Bucket width in seconds. Default 0.5.
        group_by: Grouping for the per-bin report (default: thread).
        max_bins: Hard ceiling on buckets attempted. Default 120 = 60 s.
        empty_streak_stop: Stop after this many consecutive buckets with no
                      rows from any thread.
        fail_streak_abort: Stop after this many consecutive buckets that
                      *failed* (non-zero exit, timeout, unparseable). Kept
                      separate from the empty streak and deliberately larger,
                      so a burst of transient crashes is reported rather than
                      mistaken for the end of the workload.
        slice_timeout_sec: Per-bucket subprocess timeout.
        max_workers: Concurrent VTune report processes. Buckets are issued in
                      chunks of this size and the stop conditions are checked
                      between chunks, so the run can overshoot by at most
                      ``max_workers - 1`` buckets. Leave at 1 unless probe 0.1
                      showed concurrent slicing is both faster and byte-identical.

    Returns:
        Dict with:
            bins_requested:   buckets actually attempted
            bins_written:     buckets that produced at least one row
            bins_empty:       buckets VTune reported as having no activity
            bins_failed:      buckets that failed twice
            failed_bins:      up to 20 (index, reason) pairs for diagnosis
            rows_written:     total (bin, thread) rows in the output CSV
            counter_columns:  number of counter columns in the union header
            bin_size_sec:     echo of the input
            max_bin_end_sec:  end time of the last non-empty bucket. This is
                              the best available estimate of the workload's
                              real span -- use it, not the collect wall time.
            trailing_empty_bins: empty buckets at the tail (the stop streak)
            stopped_early:    True unless we ran into max_bins
            stop_reason:      'empty_streak' | 'fail_streak' | 'max_bins'
            schema_version:   TIMELINE_SCHEMA_VERSION

    The output CSV is written even when the run aborts, so partial data
    survives. Only a missing VTune installation raises.

    Raises:
        FileNotFoundError if VTune is not installed.
    """
    if not os.path.exists(VTUNE_PATH):
        raise FileNotFoundError(f"VTune not found at {VTUNE_PATH}")

    if duration_sec is not None and duration_sec > 0:
        # Advisory: never *extend* past the hint, but still stop early when
        # the workload turns out to be shorter than the caller believed.
        max_bins = min(max_bins, int(math.ceil(duration_sec / bin_size_sec)))
    max_bins = max(1, int(max_bins))
    workers = max(1, int(max_workers))

    all_rows = []  # list of (bin_start, bin_end, parsed_row_dict)
    counter_keys_union = set()

    def _flush():
        """Write everything collected so far. Called from a finally block, so
        a run that aborts at bucket 40 still keeps buckets 0-39 -- losing them
        is exactly what cost 17 runs their timelines in the v3 dataset."""
        identity_cols = ["bin_start_sec", "bin_end_sec", "thread", "tid", "pid"]
        # Strip identity/derived keys out of the counter union so they don't
        # get double-listed. compute_derived_metrics adds ipc/l1_miss_rate/
        # l3_miss_rate — keep them at the end for readability.
        derived_cols = ["ipc", "l1_miss_rate", "l3_miss_rate"]
        identity_keys = set(identity_cols) | set(derived_cols)
        counter_cols = sorted(
            k for k in counter_keys_union if k not in identity_keys
        )
        header = identity_cols + counter_cols + derived_cols

        written = 0
        seen = set()
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t0, t1, row in all_rows:
                key = (t0, t1)
                if key not in seen:
                    seen.add(key)
                    written += 1
                out = [
                    t0,
                    t1,
                    row.get("thread", ""),
                    row.get("tid", ""),
                    row.get("pid", ""),
                ]
                for k in counter_cols:
                    out.append(row.get(k, 0))
                for k in derived_cols:
                    # None (missing L1/L3 counters) becomes blank, not "None"
                    v = row.get(k)
                    out.append("" if v is None else v)
                writer.writerow(out)
        return written, len(counter_cols)

    bins_requested = 0
    bins_empty = 0
    bins_failed = 0
    failed_bins = []
    consecutive_empty = 0
    consecutive_failed = 0
    max_bin_end_sec = 0.0
    stop_reason = "max_bins"

    try:
        with tempfile.TemporaryDirectory(prefix="vtune_slice_") as scratch:
            next_bin = 0
            while next_bin < max_bins:
                chunk = list(range(next_bin, min(next_bin + workers, max_bins)))
                next_bin = chunk[-1] + 1

                if workers > 1 and len(chunk) > 1:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        outcomes = list(pool.map(
                            lambda i: _slice_one_bin(
                                result_dir, scratch, i, bin_size_sec,
                                group_by, slice_timeout_sec),
                            chunk,
                        ))
                else:
                    outcomes = [
                        _slice_one_bin(result_dir, scratch, i, bin_size_sec,
                                       group_by, slice_timeout_sec)
                        for i in chunk
                    ]

                # Process in bucket order so the streak counters mean what
                # they say even when the chunk ran concurrently.
                for o in outcomes:
                    bins_requested += 1
                    if not o["ok"]:
                        bins_failed += 1
                        consecutive_failed += 1
                        if len(failed_bins) < 20:
                            failed_bins.append({"index": o["index"],
                                                "reason": o["error"]})
                        continue
                    consecutive_failed = 0
                    if o["rows"]:
                        consecutive_empty = 0
                        if o["t1"] > max_bin_end_sec:
                            max_bin_end_sec = o["t1"]
                        for row in o["rows"]:
                            counter_keys_union.update(row.keys())
                            all_rows.append((o["t0"], o["t1"], row))
                    else:
                        bins_empty += 1
                        consecutive_empty += 1

                if consecutive_failed >= fail_streak_abort:
                    stop_reason = "fail_streak"
                    break
                if consecutive_empty >= empty_streak_stop:
                    stop_reason = "empty_streak"
                    break
    finally:
        bins_written, counter_columns = _flush()

    return {
        "bins_requested": bins_requested,
        "bins_written": bins_written,
        "bins_empty": bins_empty,
        "bins_failed": bins_failed,
        "failed_bins": failed_bins,
        "rows_written": len(all_rows),
        "counter_columns": counter_columns,
        "bin_size_sec": bin_size_sec,
        "max_bin_end_sec": max_bin_end_sec,
        "trailing_empty_bins": consecutive_empty,
        "stopped_early": stop_reason != "max_bins",
        "stop_reason": stop_reason,
        "max_workers": workers,
        "schema_version": TIMELINE_SCHEMA_VERSION,
    }


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


def aggregate_workload_threads(rows: list, min_instructions: int = 1_000_000) -> dict:
    """Sum counters across all threads that did meaningful work.

    For multi-threaded workloads (7-Zip -mmt8, x264 --threads 8, etc.),
    ``get_primary_thread()`` returns one arbitrary hot thread and hides the rest.
    This function aggregates all threads whose instruction count exceeds a small
    threshold (default 1M) — enough to filter out system-idle threads while
    keeping every worker thread of the target application.

    Aggregate IPC is the weighted average, computed as
    ``sum(instructions) / sum(cycles)`` — mathematically identical to
    weighting each thread's IPC by its instruction count.

    Args:
        rows: Output of parse_csv() (optionally passed through
              compute_derived_metrics()).
        min_instructions: Minimum instruction count for a thread to count as
                          "active workload work". Threads below this are
                          treated as system noise.

    Returns:
        Dict with:
            aggregate_ipc:       total_instructions / total_cycles
            total_instructions:  sum of INST_RETIRED.ANY across active threads
            total_cycles:        sum of CPU_CLK_UNHALTED.THREAD across active threads
            active_thread_count: how many threads passed the filter
            total_thread_count:  how many rows were in the input
    """
    active = [r for r in rows if r.get(EVENT_INSTRUCTIONS, 0) >= min_instructions]
    total_inst = sum(r.get(EVENT_INSTRUCTIONS, 0) for r in active)
    total_cyc = sum(r.get(EVENT_CYCLES, 0) for r in active)

    return {
        "aggregate_ipc": (total_inst / total_cyc) if total_cyc > 0 else 0.0,
        "total_instructions": total_inst,
        "total_cycles": total_cyc,
        "active_thread_count": len(active),
        "total_thread_count": len(rows),
    }


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
