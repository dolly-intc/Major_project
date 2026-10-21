"""
SQLite workload driver — multi-threaded query benchmark for the data collection
harness.

Opens the pre-built database at ``--db``, launches ``--threads`` worker threads,
and each thread runs a mix of SELECT and UPDATE queries from a fixed pool until
``--duration`` seconds have elapsed. Prints a short summary at exit.

Standard library only (uses ``sqlite3`` and ``threading``), so it runs on the
restricted test machine.

Why this exists:
    ``sqlite3.exe`` alone has no built-in benchmark and doesn't thread queries.
    To get a realistic multi-threaded SQLite workload we need our own driver.
    Every thread opens its OWN connection to the DB file (SQLite is safe when
    each thread has its own connection) so query work runs in parallel rather
    than serializing on one connection's lock.

Usage:
    python workloads_sqlite.py --db workloads_data/test.sqlite --threads 8 --duration 20
"""

import argparse
import os
import random
import sqlite3
import sys
import threading
import time
from pathlib import Path


# Queries the workers rotate through. Each returns fast on the test DB (~1M
# rows) but stresses the query engine's B-tree walk and value comparison
# paths, which is exactly the memory-latency-sensitive work we want to sample.
SELECT_QUERIES = [
    "SELECT COUNT(*) FROM t WHERE k > ? AND k < ?",
    "SELECT SUM(k) FROM t WHERE id % 1000 = ?",
    "SELECT id, k FROM t WHERE k BETWEEN ? AND ? LIMIT 100",
    "SELECT AVG(k) FROM t WHERE id > ? LIMIT 1",
]

UPDATE_QUERIES = [
    "UPDATE t SET k = k + 1 WHERE id = ?",
    "UPDATE t SET v = v WHERE id BETWEEN ? AND ?",
]


def _worker(db_path: str, stop_at: float, counters: dict, tid: int) -> None:
    """One worker thread. Opens its own connection, loops queries until stop_at.

    Args:
        db_path: Path to the SQLite database file.
        stop_at: ``time.perf_counter()`` value at which to stop.
        counters: Shared dict; this worker writes counters[tid] = query_count.
        tid: Thread index (for the counters dict).
    """
    # check_same_thread=False is not needed since each worker has its own conn
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Per-thread PRNG so threads don't contend on the global one
    rng = random.Random(tid * 1000003 + os.getpid())

    queries = 0
    try:
        while time.perf_counter() < stop_at:
            # Batch a few queries between clock checks to keep the loop tight
            for _ in range(50):
                # 80% reads, 20% writes — realistic OLTP-ish mix
                if rng.random() < 0.8:
                    q = rng.choice(SELECT_QUERIES)
                    a = rng.randint(0, 1_000_000)
                    b = a + rng.randint(1, 10_000)
                    is_fetch = True
                else:
                    q = rng.choice(UPDATE_QUERIES)
                    a = rng.randint(0, 999_999)
                    b = a + rng.randint(1, 100)
                    is_fetch = False

                # Some queries use one `?`, some use two. Count placeholders
                # and match the parameter tuple exactly, so adding a new query
                # to either pool later doesn't need any special-case wiring
                # (previous versions guessed by keyword, which was fragile).
                params = (a, b) if q.count("?") == 2 else (a,)
                cur.execute(q, params)
                if is_fetch:
                    cur.fetchall()
                queries += 1
            # Commit occasionally so writes are visible and the WAL flushes
            conn.commit()
    finally:
        try:
            conn.commit()
        except Exception:
            pass
        conn.close()
        counters[tid] = queries


def run(db_path: str, threads: int, duration_sec: float, quiet: bool = False) -> dict:
    """Run the multi-threaded SQLite workload.

    Args:
        db_path: Path to the pre-built test database.
        threads: Number of worker threads.
        duration_sec: How long to run.
        quiet: If True, suppress stdout.

    Returns:
        Dict with total queries, per-thread queries, actual duration.

    Raises:
        FileNotFoundError: If the database doesn't exist.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"SQLite database not found at {db_path}. "
            "Run collect/workload_setup.py to build it."
        )

    stop_at = time.perf_counter() + duration_sec
    counters = {}
    workers = []

    start = time.perf_counter()
    for i in range(threads):
        t = threading.Thread(target=_worker, args=(db_path, stop_at, counters, i),
                             name=f"sqlite-worker-{i}", daemon=False)
        t.start()
        workers.append(t)

    for t in workers:
        t.join()
    elapsed = time.perf_counter() - start

    total_queries = sum(counters.values())
    result = {
        "threads": threads,
        "duration_sec": duration_sec,
        "actual_duration_sec": elapsed,
        "total_queries": total_queries,
        "queries_per_sec": total_queries / elapsed if elapsed > 0 else 0.0,
        "per_thread": counters,
    }

    if not quiet:
        print(f"SQLite workload: {threads} threads, {elapsed:.2f}s, "
              f"{total_queries:,} queries ({result['queries_per_sec']:,.0f} q/s)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Multi-threaded SQLite query workload."
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file")
    parser.add_argument("--threads", "-t", type=int, default=1,
                        help="Number of worker threads (default: 1)")
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                        help="Duration in seconds (default: 10)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress output (for subprocess invocation)")
    args = parser.parse_args()

    # Resolve --db relative to this script's directory so it works when
    # runner.py invokes us from an arbitrary cwd. --db paths that are already
    # absolute stay absolute.
    db_path = args.db
    if not os.path.isabs(db_path):
        db_path = str(Path(__file__).parent / db_path)

    run(db_path=db_path, threads=args.threads,
        duration_sec=args.duration, quiet=args.quiet)


if __name__ == "__main__":
    main()
