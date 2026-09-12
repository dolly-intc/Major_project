# collect/ — data collection, runs on the test machine

> ## ⚠ STANDARD LIBRARY ONLY
>
> Every file in this folder runs on the remote test machine, which has **no internet and no
> `pip`**. Import nothing outside the Python standard library, and nothing from outside this
> folder. No `pandas`, no `numpy`, no relative imports reaching up to the repository root.

## Why the rule is this strict

The test machine is the only hybrid hardware available and it is reached over Remote Desktop.
Files get there by **clipboard or drive redirection** — you copy this folder across by hand.

So `collect/` has to survive being moved somewhere else entirely, with no repository around it
and no packages installed. If `core_topology.py` imported a helper from a sibling directory, or
if any file imported `numpy`, the failure would not appear here. It would appear after the copy,
partway into a measurement run, on scarce hardware that may be reclaimed by another team at
short notice.

Analysis is not subject to this constraint — that all happens on the dev machine in `analysis/`,
working from the CSVs this folder produces. The split is only possible because Stage 2
modelling needs saved data, not hybrid cores.

## Before copying to the test machine

```bash
python tests/test_stdlib_only.py
```

Run from the repository root. It parses every file here and fails on any non-stdlib import.

## Contents

| File | Purpose |
|---|---|
| `core_topology.py` | **Import this everywhere.** Discovers which logical CPUs are P-cores and which are E-cores, via the Windows `GetLogicalProcessorInformationEx` API. |
| `compare_workloads.py` | The Stage 0 feasibility experiment. Runs a compute-bound and a memory-bound workload on each core type and reports the E/P time ratios. Produced the numbers in `PROJECT_CONTEXT.md` §6. |

## Never hardcode core indices

On the test machine the P-cores are `{0, 1, 10, 11, 12, 13, 22, 23}` — **not** contiguous, and
interleaved with E-cores. Assuming the conventional `0–7` layout silently inverts every label,
and every result downstream would be wrong in a way that looks plausible. Always call
`get_topology()` at runtime.

## Still to be written (Stage 1)

The counter collection harness — a wrapper that drives Intel VTune from Python, runs a workload,
and parses per-thread counters back out of VTune's CSV export.

It is deliberately **not** written yet. VTune is not installed on the dev machine, so none of it
can be tested here, and its exact command-line knob names and CSV column headers vary between
VTune versions. Work through `docs/PREREQUISITES.md` first to capture the real values from the
installed build, then write the harness once against facts.
