"""
Dev-machine-only: download and stage the external binaries and test inputs
required by the real-application workloads (7-Zip, zstd, x264, SQLite), plus
generate the two locally-produced inputs (a synthetic .y4m video and a
populated SQLite database).

Standard library only (uses ``urllib``, ``zipfile``, ``hashlib``, ``sqlite3``,
``tarfile``), so it doesn't need pip. It runs on the DEV machine because it
needs internet — the test machine has neither internet nor pip.

The dev machine also has network constraints (corporate proxy, some hosts
blocked, variable connectivity). Every automatic download has a manual-download
fallback: if the fetch fails, the script prints a clear block naming the
official application homepage, exactly which file to grab, where to place it,
and what its expected size range is. The user drops the file into place, then
re-runs the script — completed items are skipped by SHA-256, so re-runs are
cheap.

Usage:
    python collect/workload_setup.py               # attempt all steps
    python collect/workload_setup.py --verify      # skip downloads, just check
    python collect/workload_setup.py --only zstd   # only stage one item
    python collect/workload_setup.py --list        # show what would be done

After running, the ``collect/workloads_bin/`` and ``collect/workloads_data/``
folders travel with ``collect/`` when it is copied to the test machine over
RDP. Both folders are gitignored.
"""

import argparse
import hashlib
import io
import os
import random
import shutil
import sqlite3
import struct
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


COLLECT_DIR = Path(__file__).parent
BIN_DIR = COLLECT_DIR / "workloads_bin"
DATA_DIR = COLLECT_DIR / "workloads_data"

# Short network timeout so the corporate-proxy failure mode surfaces fast
# rather than hanging. Real downloads still get plenty of wall time via the
# per-chunk read loop.
CONNECT_TIMEOUT_SEC = 30


# -----------------------------------------------------------------------------
# Download items — everything with a canonical remote URL
# -----------------------------------------------------------------------------
#
# Each entry:
#   dest:       final path (relative to collect/)
#   url:        automatic-download URL (best-known-good)
#   extract:    None (raw file) | 'zip:<member>' | '7z:<member>' | 'silesia_zip'
#   homepage:   URL to give the user in the manual-fallback block
#   manual:     multi-line human instructions for the manual path
#   min_size:   reject downloads below this many bytes (usually an error page)

DOWNLOADS = {
    "7z": {
        "dest": BIN_DIR / "7z.exe",
        # NOTE: VTune's sampling driver refuses 32-bit targets. The 7-Zip
        # "Extra" package's top-level 7zr.exe / 7za.exe are BOTH 32-bit; the
        # 64-bit standalone lives at x64/7za.exe inside that archive. To keep
        # the manual path simple, we instead ask the user for the regular
        # 7-Zip x64 installer -- installing it puts a 64-bit 7z.exe at
        # C:\Program Files\7-Zip\7z.exe which we can copy straight into
        # workloads_bin/. Auto-download of the installer requires running it,
        # which we won't do headless, so this item is always manual.
        "url": None,
        "extract": "raw_7zr",  # forces the manual-fallback path
        "homepage": "https://www.7-zip.org/download.html",
        "manual": (
            "MUST be 64-bit -- VTune refuses 32-bit binaries.\n"
            "\n"
            "Easiest path (if 7-Zip is already installed on this machine):\n"
            "  copy \"C:\\Program Files\\7-Zip\\7z.exe\" \"" + str(BIN_DIR / "7z.exe") + "\"\n"
            "\n"
            "Otherwise:\n"
            "1. Open https://www.7-zip.org/download.html in a browser.\n"
            "2. Download '7z<version>-x64.exe' (the 64-bit x64 Windows installer).\n"
            "3. Run the installer (default location C:\\Program Files\\7-Zip\\).\n"
            "4. Copy C:\\Program Files\\7-Zip\\7z.exe to:\n"
            "     " + str(BIN_DIR / "7z.exe") + "\n"
            "\n"
            "Sanity-check it's 64-bit:  7z.exe   (banner should say 'x64' or '64-bit')"
        ),
        "min_size": 100_000,
    },
    "zstd": {
        "dest": BIN_DIR / "zstd.exe",
        "url": "https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-v1.5.6-win64.zip",
        "extract": "zip:zstd.exe",
        "homepage": "https://github.com/facebook/zstd/releases",
        "manual": (
            "1. Open https://github.com/facebook/zstd/releases in a browser.\n"
            "2. Under the latest release's Assets, download 'zstd-vX.Y.Z-win64.zip'.\n"
            "3. Open the zip and extract zstd.exe (it's at the top level).\n"
            "4. Place it at: " + str(BIN_DIR / "zstd.exe")
        ),
        "min_size": 500_000,
    },
    "x264": {
        "dest": BIN_DIR / "x264.exe",
        "url": "https://artifacts.videolan.org/x264/release-win64/x264.exe",
        "extract": None,
        "homepage": "https://www.videolan.org/developers/x264.html",
        "manual": (
            "1. Open https://www.videolan.org/developers/x264.html in a browser.\n"
            "2. Follow the 'x264 nightly builds' link (or go to\n"
            "   https://artifacts.videolan.org/x264/ directly).\n"
            "3. Under release-win64/ (or a similar folder), download x264.exe\n"
            "   (a stand-alone .exe, sometimes named x264-r*-win64.exe).\n"
            "4. RENAME to x264.exe if needed.\n"
            "5. Place it at: " + str(BIN_DIR / "x264.exe")
        ),
        "min_size": 500_000,
    },
    "sqlite3": {
        "dest": BIN_DIR / "sqlite3.exe",
        # Version-pinned URLs work here; SQLite keeps old ones live indefinitely.
        "url": "https://www.sqlite.org/2024/sqlite-tools-win-x64-3460100.zip",
        "extract": "zip:sqlite3.exe",
        "homepage": "https://sqlite.org/download.html",
        "manual": (
            "1. Open https://sqlite.org/download.html in a browser.\n"
            "2. Under 'Precompiled Binaries for Windows', pick\n"
            "   'sqlite-tools-win-x64-<version>.zip'.\n"
            "3. Open the zip and extract sqlite3.exe.\n"
            "4. Place it at: " + str(BIN_DIR / "sqlite3.exe")
        ),
        "min_size": 500_000,
    },
    "silesia": {
        "dest": DATA_DIR / "silesia.tar",
        "url": "http://mattmahoney.net/dc/silesia.zip",
        "extract": "silesia_zip",  # unzip contents, re-pack as .tar
        "homepage": "http://mattmahoney.net/dc/silesia.html",
        "manual": (
            "1. Open http://mattmahoney.net/dc/silesia.html in a browser.\n"
            "2. Download silesia.zip (~68 MB).\n"
            "3. Extract all files from the zip into a folder, e.g. silesia/.\n"
            "4. Create a .tar from that folder:\n"
            "     tar -cf silesia.tar -C silesia .\n"
            "   (Windows 10+ ships tar. From PowerShell use the same command.)\n"
            "5. Place silesia.tar at: " + str(DATA_DIR / "silesia.tar")
        ),
        "min_size": 50_000_000,  # ~68MB uncompressed silesia
    },
}


# -----------------------------------------------------------------------------
# Locally-generated items — never a download failure
# -----------------------------------------------------------------------------

# .y4m dimensions. 640x360 keeps file size manageable (~500 frames -> ~250 MB)
# while still giving x264 real work to do per frame.
Y4M_WIDTH = 640
Y4M_HEIGHT = 360
Y4M_FPS = 60
Y4M_FRAMES = 720  # 12 seconds @ 60fps

# SQLite test DB row count. ~1M rows at ~200 bytes each -> ~200 MB DB.
SQLITE_ROW_COUNT = 1_000_000


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def _print_manual(key: str, item: dict, why: str) -> None:
    """Print a clear manual-download block when the auto path fails."""
    print()
    print("  " + "=" * 68)
    print(f"  MANUAL DOWNLOAD NEEDED for '{key}'")
    print(f"  Reason: {why}")
    print(f"  Homepage: {item['homepage']}")
    print("  Steps:")
    for line in item["manual"].splitlines():
        print("    " + line)
    print()
    print("  Then re-run this script -- it skips items that are already in place.")
    print("  " + "=" * 68)
    print()


def _download_to(url: str, dest: Path) -> None:
    """Download url to dest with a short connect timeout.

    Uses urllib (stdlib) so no pip is needed. Streams to a temp file first so
    a mid-transfer failure doesn't leave a partial dest behind.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "workload-setup/1.0 (stdlib-only downloader)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_SEC) as resp:
            with open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _extract_after_download(item: dict, downloaded: Path, dest: Path) -> None:
    """Post-download extraction based on the 'extract' field."""
    mode = item["extract"]

    if mode is None:
        # Already the final file
        if downloaded != dest:
            os.replace(downloaded, dest)
        return

    if mode.startswith("zip:"):
        member = mode.split(":", 1)[1]
        with zipfile.ZipFile(downloaded) as zf:
            # Support member being either a bare name or a full path in the zip
            candidates = [n for n in zf.namelist()
                          if n == member or n.endswith("/" + member)]
            if not candidates:
                raise RuntimeError(
                    f"'{member}' not found in {downloaded.name}. "
                    f"Zip contents: {zf.namelist()[:8]}..."
                )
            with zf.open(candidates[0]) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
        downloaded.unlink()
        return

    if mode == "raw_7zr":
        # The 7-Zip Extra archive itself is a .7z, which we can't extract
        # without... 7-Zip. Chicken-and-egg. We defer to the manual path here.
        raise RuntimeError(
            "The 7-Zip 'Extra' package is itself a .7z archive; extracting it "
            "requires an already-installed 7-Zip. Use the manual instructions."
        )

    if mode == "silesia_zip":
        # silesia.zip has files at the top level. Re-pack as a .tar so we
        # get a single input file for zstd's benchmark mode.
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(downloaded) as zf:
                zf.extractall(td)
            with tarfile.open(dest, "w") as tf:
                for name in sorted(os.listdir(td)):
                    full = os.path.join(td, name)
                    tf.add(full, arcname=name)
        downloaded.unlink()
        return

    raise ValueError(f"Unknown extract mode: {mode}")


def _ensure_download(key: str, item: dict, verify_only: bool = False) -> bool:
    """Fetch (or verify) one download item. Returns True on success."""
    dest = item["dest"]
    label = f"  [{key:8s}] {dest.name:<20s}"

    if dest.exists() and dest.stat().st_size >= item["min_size"]:
        sha = _sha256_of(dest)[:12]
        print(f"{label} already present ({_human_size(dest.stat().st_size)}, "
              f"sha256={sha}...)  OK")
        return True

    if verify_only:
        print(f"{label} MISSING (verify-only, not downloading)")
        _print_manual(key, item, "file is missing and --verify was set")
        return False

    if item["extract"] == "raw_7zr":
        # The auto path is blocked -- the 7-Zip "Extra" package top-level is
        # 32-bit and VTune refuses 32-bit binaries. Go straight to the manual
        # instructions, which point at the regular x64 installer.
        _print_manual(key, item, "no headless auto-download for the 64-bit 7z.exe")
        return False

    print(f"{label} downloading from {item['url'][:60]}... ", end="", flush=True)
    try:
        # Download to a temp path in the same folder, then extract into dest.
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp_dl = dest.with_suffix(dest.suffix + ".dl")
        _download_to(item["url"], temp_dl)
        _extract_after_download(item, temp_dl, dest)
        # Sanity check the extracted file
        if not dest.exists():
            raise RuntimeError("extraction produced no file at expected dest")
        size = dest.stat().st_size
        if size < item["min_size"]:
            raise RuntimeError(
                f"file is {size} bytes, expected >= {item['min_size']} "
                "(probably an error page or the archive layout changed)"
            )
        sha = _sha256_of(dest)[:12]
        print(f"OK ({_human_size(size)}, sha256={sha}...)")
        return True
    except urllib.error.URLError as e:
        print(f"FAILED: {e}")
        _print_manual(key, item, f"URLError ({e})")
        return False
    except Exception as e:
        print(f"FAILED: {e}")
        _print_manual(key, item, str(e))
        # Clean up a half-written dest so re-run doesn't misread it
        for p in (dest, dest.with_suffix(dest.suffix + ".dl"),
                  dest.with_suffix(dest.suffix + ".part")):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        return False


# -----------------------------------------------------------------------------
# Local-generation steps
# -----------------------------------------------------------------------------

def _make_y4m(dest: Path, width: int, height: int, fps: int, frames: int) -> None:
    """Write a synthetic YUV4MPEG2 (.y4m) file with a moving-gradient pattern.

    x264 encodes a moving pattern (varying pixel values across space and time),
    which is a realistic operating point — random noise defeats compression,
    static frames are trivial, gradient motion sits in between and stresses
    motion estimation.

    The Y plane holds a diagonal ramp shifted per frame; U/V planes hold slow
    color shifts. Everything is uint8. YUV420 layout: full-resolution Y
    (width*height bytes), quarter-resolution U (w/2 * h/2), quarter-resolution V.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = f"YUV4MPEG2 W{width} H{height} F{fps}:1 Ip A1:1 C420jpeg\n".encode("ascii")
    frame_header = b"FRAME\n"

    y_size = width * height
    uv_w = width // 2
    uv_h = height // 2
    uv_size = uv_w * uv_h

    # Pre-build once-per-frame arrays so the inner loop is fast. Using
    # bytearray + slice assignment keeps this stdlib-only.
    y = bytearray(y_size)
    u = bytearray(uv_size)
    v = bytearray(uv_size)

    print(f"    generating .y4m ({width}x{height}, {frames} frames, "
          f"~{_human_size(y_size + uv_size * 2)} per frame)")

    with open(dest, "wb") as out:
        out.write(header)
        for f in range(frames):
            # Diagonal ramp shifted by frame index — feeds motion estimation
            shift = f * 3
            for row in range(height):
                base = ((row + shift) & 0xFF)
                start = row * width
                # Simple per-row gradient; per-column variation from row's base
                for col in range(0, width, 32):
                    val = (base + col) & 0xFF
                    end = min(col + 32, width)
                    for c in range(col, end):
                        y[start + c] = (val + c) & 0xFF
            # U/V slow color drift
            u_base = (f * 5) & 0xFF
            v_base = (128 + f * 7) & 0xFF
            for i in range(uv_size):
                u[i] = (u_base + (i & 0x3F)) & 0xFF
                v[i] = (v_base + (i & 0x3F)) & 0xFF

            out.write(frame_header)
            out.write(bytes(y))
            out.write(bytes(u))
            out.write(bytes(v))

            if (f + 1) % 60 == 0:
                print(f"      {f+1}/{frames} frames written", flush=True)


def _ensure_y4m(verify_only: bool = False) -> bool:
    dest = DATA_DIR / "sample.y4m"
    label = f"  [y4m     ] {dest.name:<20s}"
    if dest.exists() and dest.stat().st_size > 10_000_000:
        print(f"{label} already present ({_human_size(dest.stat().st_size)})  OK")
        return True
    if verify_only:
        print(f"{label} MISSING (verify-only, not generating)")
        return False
    print(f"{label} generating...")
    _make_y4m(dest, Y4M_WIDTH, Y4M_HEIGHT, Y4M_FPS, Y4M_FRAMES)
    print(f"{label} OK ({_human_size(dest.stat().st_size)})")
    return True


def _make_sqlite_db(dest: Path, rows: int) -> None:
    """Build the test SQLite DB with ``rows`` rows of random-ish data."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()

    conn = sqlite3.connect(str(dest))
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v BLOB)")
    cur.execute("CREATE INDEX idx_k ON t(k)")

    rng = random.Random(20260921)
    # Insert in batches for speed
    batch = 5000
    print(f"    inserting {rows:,} rows (batch={batch:,})...")
    for start in range(0, rows, batch):
        vals = []
        for i in range(start, min(start + batch, rows)):
            k = rng.randint(0, 1_000_000)
            v = bytes(rng.getrandbits(8) for _ in range(100))  # 100-byte blob
            vals.append((i, k, v))
        cur.executemany("INSERT INTO t (id, k, v) VALUES (?, ?, ?)", vals)
        if (start // batch) % 20 == 0 and start > 0:
            conn.commit()
            print(f"      {start:,} / {rows:,} rows", flush=True)
    conn.commit()
    cur.execute("ANALYZE")
    conn.commit()
    conn.close()


def _ensure_sqlite_db(verify_only: bool = False) -> bool:
    dest = DATA_DIR / "test.sqlite"
    label = f"  [sqlite-db] {dest.name:<19s}"
    if dest.exists() and dest.stat().st_size > 50_000_000:
        print(f"{label} already present ({_human_size(dest.stat().st_size)})  OK")
        return True
    if verify_only:
        print(f"{label} MISSING (verify-only, not generating)")
        return False
    print(f"{label} building...")
    _make_sqlite_db(dest, SQLITE_ROW_COUNT)
    print(f"{label} OK ({_human_size(dest.stat().st_size)})")
    return True


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

STEPS = list(DOWNLOADS.keys()) + ["y4m", "sqlite-db"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--verify", action="store_true",
                        help="Don't download or generate — just report which items are missing.")
    parser.add_argument("--only", nargs="+", choices=STEPS, default=None,
                        help="Only run these steps (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="Show the planned steps and exit")
    args = parser.parse_args()

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    steps = args.only if args.only else STEPS
    if args.list:
        print("Planned steps:")
        for s in steps:
            print(f"  - {s}")
        return

    print(f"Workload staging into:")
    print(f"  binaries: {BIN_DIR}")
    print(f"  data:     {DATA_DIR}")
    print()

    started = time.perf_counter()
    ok = 0
    failed = []

    for key in steps:
        if key in DOWNLOADS:
            success = _ensure_download(key, DOWNLOADS[key], verify_only=args.verify)
        elif key == "y4m":
            success = _ensure_y4m(verify_only=args.verify)
        elif key == "sqlite-db":
            success = _ensure_sqlite_db(verify_only=args.verify)
        else:
            print(f"  [{key}] unknown step, skipping")
            success = False
        if success:
            ok += 1
        else:
            failed.append(key)

    elapsed = time.perf_counter() - started
    print()
    print(f"Done in {elapsed:.1f}s: {ok}/{len(steps)} steps OK")
    if failed:
        print(f"Failed / needs manual action: {', '.join(failed)}")
        print("Follow the manual-download blocks printed above, then re-run this script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
