# The 8 phases found by Model A

28,601 windows from 166 runs / 1,188 thread-runs, clustered with KMeans (k=8, silhouette 0.442) on the 8-d autoencoder embedding.

Feature means are over every 0.5 s bin of every window in the phase. Top-down fractions (retiring / frontend / backend / bad speculation) are shares of pipeline slots.
Dwell = length of an uninterrupted stretch in the phase (stretches cut off by the start or end of a thread are included, so long-lived phases are if anything under-stated).

**Known limitation - phases are almost split by core type.** P-cores and E-cores expose different hardware counters. The four top-down features are built from E-core-only events (`TOPDOWN_*.ALL_P`), so they read 0 on every P-core thread; the L1/L3 miss rates are built from P-core-only events (`MEM_LOAD_RETIRED.*`), so they read 0 on every E-core thread. KMeans therefore separates core types first: P3, P0 and P6 are E-core phases, P7, P4, P5 and P2 are P-core phases, and only P1 (low activity) mixes both. Within one core type the phases are still meaningful behaviour groups.

## P3 - Main Compute

- **Size:** 11,019 windows (38.5%), 86 runs, 482 thread-runs; 4 of 8 workloads present (>1%)
- **Workload mix:** x264 80%, zstd 9%, sqlite 8%
- **Signature:** IPC 2.47, retiring 41%, frontend-bound 12%, backend-bound 35%, bad speculation 10%, L1 miss 0%, L3 miss 0%, branch mispredict 1.20%
- **Stability:** stays in this phase for the next window 89% of the time; median dwell 1.0 s (mean 3.9 s)
- **When it changes, goes to:** P1 (62% of exits), P6 (23% of exits)
- **Core type:** 0% of its windows ran on P-cores (11 P / 11,008 E)

## P1 - Low-Activity Catch-all

- **Size:** 9,705 windows (33.9%), 126 runs, 464 thread-runs; 8 of 8 workloads present (>1%)
- **Workload mix:** x264 58%, sqlite 14%, zstd 7%
- **Signature:** IPC 2.60, retiring 9%, frontend-bound 2%, backend-bound 23%, bad speculation 1%, L1 miss 1%, L3 miss 1%, branch mispredict 0.40%
- **Stability:** stays in this phase for the next window 91% of the time; median dwell 1.5 s (mean 4.1 s)
- **When it changes, goes to:** P3 (79% of exits), P7 (15% of exits)
- **Core type:** 48% of its windows ran on P-cores (4,700 P / 5,005 E)

## P7 - Light Memory, High IPC

- **Size:** 1,899 windows (6.6%), 41 runs, 98 thread-runs; 4 of 8 workloads present (>1%)
- **Workload mix:** x264 68%, zstd 21%, sqlite 6%
- **Signature:** IPC 3.06, retiring 0%, frontend-bound 0%, backend-bound 0%, bad speculation 0%, L1 miss 14%, L3 miss 25%, branch mispredict 1.00%
- **Stability:** stays in this phase for the next window 90% of the time; median dwell 2.0 s (mean 4.1 s)
- **When it changes, goes to:** P1 (62% of exits), P5 (18% of exits)
- **Core type:** 100% of its windows ran on P-cores (1,898 P / 1 E)

## P0 - Speculative / Backend-Bound

- **Size:** 1,807 windows (6.3%), 41 runs, 600 thread-runs; 2 of 8 workloads present (>1%)
- **Workload mix:** 7zip 89%, x264 11%, sqlite 0%
- **Signature:** IPC 1.68, retiring 22%, frontend-bound 12%, backend-bound 42%, bad speculation 24%, L1 miss 0%, L3 miss 0%, branch mispredict 8.30%
- **Stability:** stays in this phase for the next window 82% of the time; median dwell 1.5 s (mean 1.3 s)
- **When it changes, goes to:** P3 (73% of exits), P6 (23% of exits)
- **Core type:** 0% of its windows ran on P-cores (6 P / 1,801 E)

## P6 - Frontend-Bound, Low IPC

- **Size:** 1,347 windows (4.7%), 38 runs, 119 thread-runs; 2 of 8 workloads present (>1%)
- **Workload mix:** x264 88%, sqlite 11%, io 1%
- **Signature:** IPC 0.67, retiring 15%, frontend-bound 42%, backend-bound 28%, bad speculation 13%, L1 miss 0%, L3 miss 0%, branch mispredict 0.60%
- **Stability:** stays in this phase for the next window 73% of the time; median dwell 0.5 s (mean 1.7 s)
- **When it changes, goes to:** P3 (79% of exits), P0 (12% of exits)
- **Core type:** 0% of its windows ran on P-cores (1 P / 1,346 E)

## P4 - L3-Bound

- **Size:** 1,222 windows (4.3%), 39 runs, 58 thread-runs; 6 of 8 workloads present (>1%)
- **Workload mix:** sqlite 50%, memory 20%, mixed 20%
- **Signature:** IPC 2.46, retiring 0%, frontend-bound 0%, backend-bound 0%, bad speculation 0%, L1 miss 3%, L3 miss 82%, branch mispredict 0.20%
- **Stability:** stays in this phase for the next window 96% of the time; median dwell 2.5 s (mean 7.2 s)
- **When it changes, goes to:** P7 (82% of exits), P2 (10% of exits)
- **Core type:** 100% of its windows ran on P-cores (1,222 P / 0 E)

## P5 - L1-Bound, High Throughput

- **Size:** 1,071 windows (3.7%), 9 runs, 31 thread-runs; 3 of 8 workloads present (>1%)
- **Workload mix:** x264 93%, memory 6%, zstd 1%
- **Signature:** IPC 3.04, retiring 0%, frontend-bound 0%, backend-bound 0%, bad speculation 0%, L1 miss 96%, L3 miss 7%, branch mispredict 0.80%
- **Stability:** stays in this phase for the next window 96% of the time; median dwell 2.5 s (mean 9.1 s)
- **When it changes, goes to:** P2 (54% of exits), P7 (38% of exits)
- **Core type:** 100% of its windows ran on P-cores (1,071 P / 0 E)

## P2 - L1 + L3 Pressure

- **Size:** 531 windows (1.9%), 7 runs, 15 thread-runs; 3 of 8 workloads present (>1%)
- **Workload mix:** x264 93%, zstd 6%, sqlite 1%
- **Signature:** IPC 1.23, retiring 0%, frontend-bound 0%, backend-bound 0%, bad speculation 0%, L1 miss 89%, L3 miss 69%, branch mispredict 0.30%
- **Stability:** stays in this phase for the next window 96% of the time; median dwell 2.0 s (mean 8.1 s)
- **When it changes, goes to:** P5 (74% of exits), P4 (26% of exits)
- **Core type:** 100% of its windows ran on P-cores (531 P / 0 E)
