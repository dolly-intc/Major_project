# The 4 phases found by Model A

23,897 windows from 156 runs / 908 thread-runs, clustered with KMeans (k=4, silhouette 0.319, best of k=4..12) on the 8-d autoencoder embedding.

Feature means are over every 0.5 s bin of every window in the phase. Top-down fractions (retiring / frontend / backend / bad speculation) are shares of pipeline slots.
Dwell = length of an uninterrupted stretch in the phase (stretches cut off by the start or end of a thread are included, so long-lived phases are if anything under-stated).

**Features are core-type-aware.** P-cores and E-cores expose different hardware counters, so each feature is computed from that core type's own counters (P-core top-down from `TOPDOWN.SLOTS`-family events; memory stalls from `MEMORY_STALLS.*` on P and `MEM_BOUND_STALLS_LOAD.*` on E). An earlier feature set used one counter set for both, which made the phases split almost entirely by core type; that version is archived in `results/archive_pre_core_aware_features_2026-10-21/`.

## P1 - Main Compute

- **Size:** 16,018 windows (67.0%), 134 runs, 444 thread-runs; 6 of 8 workloads present (>1%)
- **Workload mix:** x264 70%, sqlite 17%, zstd 6%
- **Signature:** IPC 2.91, retiring 44%, frontend-bound 10%, backend-bound 40%, bad speculation 6%, memory stalls 10% of cycles (DRAM 4%), branch mispredict 1.00%
- **Stability:** stays in this phase for the next window 92% of the time; median dwell 1.5 s (mean 5.1 s)
- **When it changes, goes to:** P3 (88% of exits), P2 (11% of exits)
- **Core type:** 28% of its windows ran on P-cores (4,466 P / 11,552 E)

## P3 - Bad Speculation

- **Size:** 4,903 windows (20.5%), 115 runs, 682 thread-runs; 5 of 8 workloads present (>1%)
- **Workload mix:** x264 44%, 7zip 28%, zstd 20%
- **Signature:** IPC 2.41, retiring 34%, frontend-bound 11%, backend-bound 37%, bad speculation 17%, memory stalls 18% of cycles (DRAM 9%), branch mispredict 3.40%
- **Stability:** stays in this phase for the next window 72% of the time; median dwell 1.0 s (mean 1.4 s)
- **When it changes, goes to:** P1 (88% of exits), P0 (8% of exits)
- **Core type:** 23% of its windows ran on P-cores (1,145 P / 3,758 E)

## P2 - Frontend-Bound

- **Size:** 2,091 windows (8.8%), 38 runs, 80 thread-runs; 3 of 8 workloads present (>1%)
- **Workload mix:** x264 87%, io 8%, sqlite 5%
- **Signature:** IPC 2.50, retiring 23%, frontend-bound 49%, backend-bound 23%, bad speculation 6%, memory stalls 15% of cycles (DRAM 8%), branch mispredict 0.70%
- **Stability:** stays in this phase for the next window 92% of the time; median dwell 1.5 s (mean 5.1 s)
- **When it changes, goes to:** P1 (71% of exits), P3 (27% of exits)
- **Core type:** 56% of its windows ran on P-cores (1,171 P / 920 E)

## P0 - Memory-Bound

- **Size:** 885 windows (3.7%), 55 runs, 105 thread-runs; 5 of 8 workloads present (>1%)
- **Workload mix:** memory 41%, 7zip 32%, zstd 16%
- **Signature:** IPC 1.30, retiring 14%, frontend-bound 5%, backend-bound 72%, bad speculation 9%, memory stalls 64% of cycles (DRAM 57%), branch mispredict 2.10%
- **Stability:** stays in this phase for the next window 84% of the time; median dwell 1.0 s (mean 2.3 s)
- **When it changes, goes to:** P3 (83% of exits), P1 (15% of exits)
- **Core type:** 41% of its windows ran on P-cores (361 P / 524 E)
