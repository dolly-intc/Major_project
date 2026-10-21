"""Duration-driven x264 encode: loops a short clip into x264's stdin.

Why this exists
---------------
``workloads_data/sample.y4m`` is 640x360 / 720 frames — about 12 seconds of
video, generated in pure Python by ``workload_setup.py``. x264 encodes it far
faster than real time, so a single invocation lasts 1.5–3.5 s at
``--preset medium`` and produced 29 of the previous dataset's 2760 analysis
windows. Everything else in the campaign runs for ~30 s.

Feeding the same frames round and round into ``x264.exe --demuxer y4m -``
until a deadline passes makes x264 duration-driven like every other workload,
at any thread count and any preset.

**This module is gated on probe 0.2** (``collect/probe_env.py``). It only
works if VTune attributes a *grandchild* process: runner.py launches
``vtune.exe`` → ``python workloads_x264.py`` → ``x264.exe``, and the counters
we care about belong to that last process. Every timeline CSV in the previous
campaign contained exactly one PID, so there is no evidence either way. If
0.2 fails, leave ``X264_USE_FEEDER = False`` in ``workloads.py`` and take the
slower-preset path instead.

Notes that cost time to rediscover
----------------------------------
* ``--demuxer y4m`` is required. A bare ``-`` fails with "raw input requires a
  resolution", because x264 cannot guess the format of a pipe.
* Feed throughput is the real constraint. At 640x360 YUV420 a frame is 345,600
  bytes; 60 fps is ~20 MB/s and ``--preset medium`` at 8 threads wants several
  hundred. A pipe that cannot keep up turns the measurement into a study of
  this script rather than of x264, so use a slow preset (``veryslow`` needs
  roughly 14 MB/s). The run instruments its own write() calls to measure how
  much of the feed window was spent blocked on pipe back-pressure; a low
  ``blocked_fraction`` (below ``starve_threshold``, default 10%) flags the run
  as ``pipe_starved`` because x264 was idling.
* The PID file is not optional bookkeeping. With two processes under one VTune
  collection, analysis has to filter by PID rather than assume the busiest
  thread group is the encoder.

Standard library only, so it runs on the restricted test machine.

Usage:
    python workloads_x264.py --threads 8 --duration 30 \\
        --preset veryslow --input workloads_data/sample.y4m \\
        --x264 workloads_bin/x264.exe --pid-file x264_pids.json
"""

import argparse
import json
import os
import subprocess
import sys
import time

# A y4m file is a global header line, then per-frame "FRAME[params]\n" + data.
FRAME_MAGIC = b"FRAME"
Y4M_MAGIC = b"YUV4MPEG2"


# Bytes per pixel-pair for the chroma subsamplings y4m can declare. Value is
# the total payload size as a multiple of W*H.
_CHROMA_MULTIPLIER = {
    "420": 1.5, "420jpeg": 1.5, "420mpeg2": 1.5, "420paldv": 1.5,
    "422": 2.0,
    "444": 3.0,
    "mono": 1.0,
}


def _frame_payload_bytes(header: bytes) -> int:
    """Compute the fixed per-frame payload size from the y4m header.

    This has to be derived, not searched for. A y4m frame is the literal
    ``FRAME`` tag, optional parameters, ``\\n``, then exactly W*H*k bytes of
    raw planar data -- and that data contains arbitrary byte values, including
    plenty of coincidental ``\\nFRAME`` sequences. Scanning for the next marker
    therefore skips real frame boundaries and stitches many frames into one:
    against the real 720-frame sample.y4m it reported 4 "frames" of 17, 54 and
    88 MB. The result still round-tripped byte-for-byte, which is exactly what
    made that wrong answer look right.
    """
    fields = header.decode("ascii", "replace").split()
    width = height = None
    chroma = "420"  # y4m's default when C is absent
    for tok in fields[1:]:
        if tok.startswith("W"):
            width = int(tok[1:])
        elif tok.startswith("H"):
            height = int(tok[1:])
        elif tok.startswith("C"):
            chroma = tok[1:]

    if not width or not height:
        raise ValueError(f"y4m header lacks W/H: {header!r}")

    key = chroma.split("p")[0] if chroma.startswith("mono") else chroma
    if key not in _CHROMA_MULTIPLIER:
        raise ValueError(
            f"unsupported y4m colourspace C{chroma!r}; known: "
            f"{sorted(_CHROMA_MULTIPLIER)}"
        )

    size = width * height * _CHROMA_MULTIPLIER[key]
    if size != int(size):
        raise ValueError(
            f"{width}x{height} C{chroma} gives a non-integer frame size"
        )
    return int(size)


def parse_y4m(path: str) -> tuple:
    """Split a y4m file into (header_line, [frame_chunks]).

    Each frame chunk keeps its own ``FRAME...\\n`` marker, so re-emitting the
    header once followed by any sequence of chunks is a valid y4m stream. The
    whole file is held in memory deliberately: at 640x360 / 720 frames it is
    ~250 MB, and re-reading from disk every loop would make this a disk
    benchmark rather than an encode.

    Raises:
        ValueError: if the file is not a y4m stream, declares a colourspace
                    this function can't size, or does not divide cleanly into
                    whole frames.
    """
    with open(path, "rb") as f:
        data = f.read()

    if not data.startswith(Y4M_MAGIC):
        raise ValueError(
            f"{path} does not start with {Y4M_MAGIC.decode()}; "
            "not a y4m file (run workload_setup.py to regenerate it)"
        )

    nl = data.find(b"\n")
    if nl < 0:
        raise ValueError(f"{path} has no header terminator")
    header = data[:nl + 1]
    payload = _frame_payload_bytes(header)

    frames = []
    pos = nl + 1
    while pos < len(data):
        if not data.startswith(FRAME_MAGIC, pos):
            raise ValueError(
                f"{path}: expected a FRAME marker at byte {pos}, found "
                f"{data[pos:pos + 8]!r}. The file is truncated or not the "
                f"layout this parser assumes."
            )
        fnl = data.find(b"\n", pos)
        if fnl < 0:
            raise ValueError(f"{path}: unterminated FRAME header at {pos}")
        end = fnl + 1 + payload
        if end > len(data):
            raise ValueError(
                f"{path}: last frame is short by {end - len(data)} bytes "
                f"(truncated file)"
            )
        frames.append(data[pos:end])
        pos = end

    if not frames:
        raise ValueError(f"{path} contains no frames")

    return header, frames


def _resolve(path: str) -> str:
    """Make a relative path absolute against this script's directory.

    Two reasons, both learned the hard way:
    * ``CreateProcess`` refuses a relative application name like
      ``workloads_bin/x264.exe`` with WinError 2 even when the file is right
      there, so argv[0] must be absolute.
    * It removes the dependency on runner.py happening to set cwd=collect/.
    """
    if os.path.isabs(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, path)
    return candidate if os.path.exists(candidate) else os.path.abspath(path)


def run(x264_path: str, input_path: str, threads: int, duration_sec: float,
        preset: str = "veryslow", pid_file: str = None,
        starve_threshold: float = 0.10, quiet: bool = False) -> dict:
    """Encode looped frames until ``duration_sec`` elapses.

    Returns a dict of counters. ``ok`` is False if x264 died early or the
    feeder was starving x264 (low blocked-write fraction).

    Starvation detection
    --------------------
    The feeder must push data faster than x264 consumes it, otherwise x264
    idles waiting on the pipe and the measurement becomes a study of *this
    script* rather than of x264. We detect this by measuring what fraction of
    the feed window the feeder spends **blocked inside write()**.

    - If x264 is the slower side, the pipe stays full and ``write()`` calls
      block until x264 drains enough buffer space. That is healthy: the feeder
      is waiting on x264, so x264 is the constraint.
    - If the feeder is the slower side, writes complete instantly (pipe never
      fills) and x264 idles waiting on the next frame. That is starved.

    ``starve_threshold`` is the minimum blocked fraction required. Below it the
    run is flagged ``pipe_starved=True`` and ``ok=False``. Default 0.10 means
    at least 10% of the feed window must be spent blocked — a generous bar,
    since a healthy encode with a slow preset easily reaches 50–80%.
    """
    x264_path = _resolve(x264_path)
    input_path = _resolve(input_path)

    header, frames = parse_y4m(input_path)
    frame_bytes = sum(len(f) for f in frames)

    cmd = [
        x264_path,
        "--demuxer", "y4m",          # required: x264 cannot sniff a pipe
        "--threads", str(threads),
        "--preset", preset,
        "-o", "NUL",                 # no disk contention
        "-",
    ]

    # Measure total span from process creation to exit — this is the duration
    # that determines how many VTune timeline bins exist, since VTune profiles
    # the child process, not just the feeding window.
    spawn_start = time.perf_counter()

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Write the PID pair before feeding anything: if the encode dies, analysis
    # still needs to know which PID to look for (and which to ignore).
    pids = {"feeder_pid": os.getpid(), "target_pid": proc.pid,
            "x264_cmd": cmd, "preset": preset, "threads": threads}
    if pid_file:
        with open(pid_file, "w", encoding="utf-8") as f:
            json.dump(pids, f, indent=2)
    if not quiet:
        print(f"FEEDER_PID {os.getpid()}", flush=True)
        print(f"TARGET_PID {proc.pid}", flush=True)

    start = time.perf_counter()
    deadline = start + duration_sec
    frames_fed = 0
    bytes_fed = 0
    loops = 0
    broke_early = None
    blocked_sec = 0.0  # cumulative time spent inside write() calls

    def _timed_write(data: bytes) -> None:
        """Write to stdin and accumulate the time spent blocked."""
        nonlocal blocked_sec
        t0 = time.perf_counter()
        proc.stdin.write(data)
        blocked_sec += time.perf_counter() - t0

    try:
        _timed_write(header)
        while time.perf_counter() < deadline:
            for chunk in frames:
                _timed_write(chunk)
                frames_fed += 1
                bytes_fed += len(chunk)
                # Check the clock per frame, not per loop: one full pass is
                # ~12 s of video, which would overshoot a 30 s deadline by up
                # to 12 s and make the span depend on the encode speed.
                if time.perf_counter() >= deadline:
                    break
            loops += 1
            if proc.poll() is not None:
                broke_early = f"x264 exited early with code {proc.returncode}"
                break
    except (BrokenPipeError, OSError) as e:
        # x264 closing the pipe is how it reports almost any failure.
        broke_early = f"pipe closed by x264: {e}"

    fed_sec = time.perf_counter() - start

    try:
        proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        broke_early = broke_early or "x264 did not exit within 60s of EOF"

    # total_span_sec is from Popen creation to process exit — the span that
    # determines how many VTune timeline bins exist, since VTune times the
    # child process. duration_actual_sec (feed window only) is strictly less.
    total_span_sec = time.perf_counter() - spawn_start

    mbps = (bytes_fed / (1024.0 * 1024.0) / fed_sec) if fed_sec > 0 else 0.0

    # Starvation detection: what fraction of the feed window was spent blocked
    # inside write()? A high fraction (>10%) means the pipe was full and writes
    # had to wait for x264 to drain it — that is healthy. A low fraction means
    # writes completed instantly and x264 was idling waiting on the feeder.
    blocked_fraction = blocked_sec / fed_sec if fed_sec > 0 else 0.0
    pipe_starved = blocked_fraction < starve_threshold

    stats = dict(pids)
    stats.update({
        "ok": broke_early is None and proc.returncode == 0 and not pipe_starved,
        "error": broke_early,
        "returncode": proc.returncode,
        "duration_requested_sec": duration_sec,
        "duration_actual_sec": round(fed_sec, 3),
        "total_span_sec": round(total_span_sec, 3),
        "frames_fed": frames_fed,
        "loops_completed": loops,
        "source_frames": len(frames),
        "source_frame_bytes": frame_bytes,
        "fed_mb": round(bytes_fed / (1024.0 * 1024.0), 1),
        "fed_mb_per_sec": round(mbps, 1),
        "blocked_sec": round(blocked_sec, 3),
        "blocked_fraction": round(blocked_fraction, 3),
        "starve_threshold": starve_threshold,
        "pipe_starved": pipe_starved,
    })

    if not quiet:
        print(json.dumps(stats, indent=2))
        if pipe_starved:
            print(
                f"WARNING: blocked_fraction={blocked_fraction:.1%} is below "
                f"the {starve_threshold:.0%} threshold — x264 was probably "
                f"idling waiting on the feeder, which makes this a measurement "
                f"of THIS SCRIPT, not of x264. Use a slower preset or fewer "
                f"threads.",
                file=sys.stderr,
            )
        if broke_early:
            print(f"ERROR: {broke_early}", file=sys.stderr)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Loop a y4m clip into x264 until a deadline passes."
    )
    parser.add_argument("--x264", default="workloads_bin/x264.exe",
                        help="Path to x264.exe (relative to collect/)")
    parser.add_argument("--input", default="workloads_data/sample.y4m",
                        help="Source y4m clip to loop")
    parser.add_argument("--threads", "-t", type=int, default=1,
                        help="x264 --threads value")
    parser.add_argument("--duration", "-d", type=float, default=30.0,
                        help="How long to keep feeding, in seconds")
    parser.add_argument("--preset", default="veryslow",
                        help="x264 --preset. Slower presets need less feed "
                             "bandwidth; 'medium' at 8 threads cannot be fed "
                             "fast enough through a pipe.")
    parser.add_argument("--pid-file", default=None,
                        help="Write {feeder_pid, target_pid} here so analysis "
                             "can filter the timeline by PID instead of "
                             "guessing which process is the encoder")
    parser.add_argument("--starve-threshold", type=float, default=0.10,
                        help="Minimum blocked-write fraction required (0.0–1.0). "
                             "If the feeder spends less than this fraction of "
                             "its time blocked inside write(), x264 was idling "
                             "and the run is marked pipe_starved. Default 0.10.")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress output (for subprocess invocation)")

    args = parser.parse_args()

    if not os.path.exists(_resolve(args.x264)):
        print(f"ERROR: x264 not found at {args.x264}", file=sys.stderr)
        return 2
    if not os.path.exists(_resolve(args.input)):
        print(f"ERROR: input clip not found at {args.input}", file=sys.stderr)
        return 2

    stats = run(
        x264_path=args.x264,
        input_path=args.input,
        threads=args.threads,
        duration_sec=args.duration,
        preset=args.preset,
        pid_file=args.pid_file,
        starve_threshold=args.starve_threshold,
        quiet=args.quiet,
    )
    return 0 if stats["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
