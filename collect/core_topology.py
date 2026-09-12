"""
CPU topology discovery for Intel hybrid processors.

Queries Windows for each physical core's efficiency class and returns the
P-core / E-core split. Standard library only, so it runs on the restricted
test machine.

IMPORTANT: always call this at startup rather than hardcoding CPU indices.
On the Core Ultra 9 285HX test machine, P-cores are NOT contiguous --
they are {0, 1, 10, 11, 12, 13, 22, 23}, interleaved with E-cores.
Assuming the conventional 0-7 layout would silently invert every label.

Usage:
    from core_topology import get_topology
    topo = get_topology()
    print(topo["p_cores"], topo["e_cores"])
"""

import ctypes
import struct
from ctypes import wintypes

RELATION_PROCESSOR_CORE = 0

# Offsets within SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX (x64):
#   0  Relationship (DWORD)
#   4  Size (DWORD)
#   8  Processor.Flags (BYTE)
#   9  Processor.EfficiencyClass (BYTE)
#  10  Processor.Reserved (BYTE[20])
#  30  Processor.GroupCount (WORD)
#  32  Processor.GroupMask[0].Mask (KAFFINITY / ULONG_PTR)
#  40  Processor.GroupMask[0].Group (WORD)
_OFF_EFFICIENCY_CLASS = 9
_OFF_GROUP_MASK = 32


def get_topology():
    """Return {'p_cores': [...], 'e_cores': [...], 'cores': [(eff, [cpus]), ...]}.

    Raises OSError if the Windows API call fails, and RuntimeError if the
    machine does not report a hybrid topology (a single efficiency class),
    which would indicate non-hybrid hardware or a virtualized environment.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    # First call with a null buffer to learn the required size. This is
    # expected to fail with ERROR_INSUFFICIENT_BUFFER (122).
    size = wintypes.DWORD(0)
    kernel32.GetLogicalProcessorInformationEx(
        RELATION_PROCESSOR_CORE, None, ctypes.byref(size)
    )

    buffer = (ctypes.c_ubyte * size.value)()
    ok = kernel32.GetLogicalProcessorInformationEx(
        RELATION_PROCESSOR_CORE, buffer, ctypes.byref(size)
    )
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())

    raw = bytes(buffer)
    offset = 0
    cores = []

    while offset < len(raw):
        relationship, entry_size = struct.unpack_from("<II", raw, offset)
        if entry_size == 0:
            break  # defensive: avoid an infinite loop on malformed data
        if relationship == RELATION_PROCESSOR_CORE:
            efficiency = raw[offset + _OFF_EFFICIENCY_CLASS]
            mask = struct.unpack_from("<Q", raw, offset + _OFF_GROUP_MASK)[0]
            cpus = [i for i in range(64) if (mask >> i) & 1]
            cores.append((efficiency, cpus))
        offset += entry_size

    classes = sorted({eff for eff, _ in cores})
    if len(classes) < 2:
        raise RuntimeError(
            f"Expected a hybrid topology but found {len(classes)} efficiency "
            "class(es). This machine may not have P/E cores, or may be a VM."
        )

    performance_class = max(classes)
    p_cores = sorted(c for eff, cpus in cores if eff == performance_class for c in cpus)
    e_cores = sorted(c for eff, cpus in cores if eff != performance_class for c in cpus)

    return {"p_cores": p_cores, "e_cores": e_cores, "cores": cores}


if __name__ == "__main__":
    topo = get_topology()
    print(f"P-cores ({len(topo['p_cores']):2d}): {topo['p_cores']}")
    print(f"E-cores ({len(topo['e_cores']):2d}): {topo['e_cores']}")
