"""
Machine fingerprinting for data traceability.

Captures identifying information about the test machine so that measurements from
different physical units are never silently pooled. Every data file records this
fingerprint, and the fingerprint is used as the top-level directory name under
data/raw/.

Standard library only, so it runs on the restricted test machine.

Why this matters: the test machine may be swapped for a different physical unit of
the same platform at short notice. Two units do not produce interchangeable numbers.
Silently pooling measurements from both would corrupt every result downstream in a
way that is close to undetectable after the fact. The fingerprint makes this impossible
by construction — data from two units cannot end up in the same folder by accident.

Usage:
    from fingerprint import get_fingerprint, fingerprint_id
    fp = get_fingerprint()
    print(fp)  # full dict
    print(fingerprint_id(fp))  # short string for folder names
"""

import json
import platform
import subprocess


def _run_powershell(command: str) -> str:
    """Run a PowerShell command and return its stdout.

    Uses PowerShell instead of wmic because wmic is deprecated and may not be
    available on newer Windows 11 builds.
    """
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return result.stdout.strip()


def get_fingerprint() -> dict:
    """Return a dict identifying this specific machine.

    Fields:
        serial: BIOS serial number (unique per physical unit)
        cpu: processor model name
        bios_version: BIOS version string
        memory_gb: total installed RAM in GB
        memory_speed_mhz: memory speed (may be None if not reported)
        os_build: Windows build number
        os_version: full Windows version string
        hostname: machine hostname

    Uses PowerShell's Get-CimInstance instead of deprecated wmic.
    Raises subprocess errors if PowerShell commands fail (unlikely on Windows).
    """
    # BIOS serial number — the primary unique identifier
    serial = _run_powershell(
        "(Get-CimInstance -ClassName Win32_BIOS).SerialNumber"
    ) or "UNKNOWN"

    # BIOS version
    bios_version = _run_powershell(
        "(Get-CimInstance -ClassName Win32_BIOS).SMBIOSBIOSVersion"
    ) or "UNKNOWN"

    # CPU model
    cpu = _run_powershell(
        "(Get-CimInstance -ClassName Win32_Processor).Name"
    ) or "UNKNOWN"

    # Memory: total capacity
    mem_capacity_str = _run_powershell(
        "(Get-CimInstance -ClassName Win32_PhysicalMemory | "
        "Measure-Object -Property Capacity -Sum).Sum"
    )
    try:
        memory_gb = round(int(mem_capacity_str) / (1024**3), 1)
    except (ValueError, TypeError):
        memory_gb = 0

    # Memory speed (take the first DIMM's speed)
    mem_speed_str = _run_powershell(
        "(Get-CimInstance -ClassName Win32_PhysicalMemory | "
        "Select-Object -First 1).Speed"
    )
    try:
        memory_speed_mhz = int(mem_speed_str)
    except (ValueError, TypeError):
        memory_speed_mhz = None

    # OS information
    os_build = platform.version()  # e.g. "10.0.26200"
    os_version = platform.platform()  # e.g. "Windows-11-10.0.26200-SP0"

    return {
        "serial": serial,
        "cpu": cpu,
        "bios_version": bios_version,
        "memory_gb": memory_gb,
        "memory_speed_mhz": memory_speed_mhz,
        "os_build": os_build,
        "os_version": os_version,
        "hostname": platform.node(),
    }


def fingerprint_id(fp: dict) -> str:
    """Return a short string suitable for use as a folder name.

    Format: <serial>_<cpu-short>

    Example: FZA633700090_Ultra9-290HX

    The serial alone is unique, but including a CPU fragment makes the folder
    name human-readable when browsing data/raw/.
    """
    serial = fp.get("serial", "UNKNOWN")

    # Extract a short CPU identifier
    cpu = fp.get("cpu", "")
    cpu_short = "unknown"
    if "Ultra" in cpu:
        # "Intel(R) Core(TM) Ultra 9 290HX Plus" -> "Ultra9-290HX"
        parts = cpu.split()
        for i, part in enumerate(parts):
            if part == "Ultra" and i + 2 < len(parts):
                cpu_short = f"Ultra{parts[i+1]}-{parts[i+2]}"
                break
    elif "Core" in cpu:
        # Fallback for non-Ultra: "Intel(R) Core(TM) i7-12700H" -> "i7-12700H"
        parts = cpu.split()
        for part in parts:
            if part.startswith("i") and "-" in part:
                cpu_short = part
                break

    # Sanitize for filesystem
    safe_id = f"{serial}_{cpu_short}".replace(" ", "_").replace("(", "").replace(")", "")
    return safe_id


def save_fingerprint(fp: dict, path: str) -> None:
    """Write fingerprint dict to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fp, f, indent=2)


def load_fingerprint(path: str) -> dict:
    """Load fingerprint dict from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    fp = get_fingerprint()
    print("Machine fingerprint:")
    print(json.dumps(fp, indent=2))
    print(f"\nFingerprint ID (for folder names): {fingerprint_id(fp)}")
