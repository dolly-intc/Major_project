# Stage 1 final dataset

Machine: Intel Core Ultra 9 290HX (8 P-cores, 16 E-cores), 32 GB DDR5-6400, Windows 11 (build 26200). Counters from Intel VTune (uarch-exploration); energy from Intel SoC Watch.

| dataset | description | dates | runs | ok | failed | workloads | configs | thread_counts | reps_per_config | workload_span_sec | run_wall_min_total | ipc_range | vtune_counters | timeline_bins_total | avg_power_w_mean | energy_j_mean | energy_truncated_runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage1_v4 | All 8 workloads (its 7-Zip runs failed: 32-bit exe) | 2026-10-01 to 2026-10-02 | 180 | 141 | 39 | 7zip, zstd, x264, sqlite, compute, memory, mixed, io | 29 | 1, 4, 8, 16 | 4-5 | 18.5-33.5 | 603.3 | 0.74-7.16 | 218 | 8874 | 25.12 | 2388.9 | 0 |
| stage1_v4_7zip | 7-Zip re-collected with the 64-bit exe | 2026-10-02 to 2026-10-03 | 35 | 34 | 1 | 7zip | 7 | 1, 4, 8, 16 | 4-5 | 11.5-23.5 | 100.4 | 1.40-1.99 | 218 | 1133 | 27.8 | 2643.1 | 0 |
| **final dataset** | OK runs of both, 7-Zip from the retry |  | 175 | 175 | 0 |  |  |  |  |  |  |  |  | 10007 |  |  |  |

Notes:
- `ok`/`failed` come from each run's `status` field. Counters are recorded every 0.5 s (the per-run timeline used by Stage 2).
- Energy is one package-level number per run over a window that includes idle time (at least 40 s), so compare runs with each other, not with workload-only energy.
- stage1_v4 failures: 35 of 35 first-pass 7-Zip runs (32-bit `7z.exe`, which VTune refuses) plus 4 others; 7-Zip was re-collected in stage1_v4_7zip. 7-Zip's high active-thread count is real: its benchmark spawns many short-lived worker threads.
