# Final Stage 1 dataset (stage1_v4 + stage1_v4_7zip), per workload

Mean over all OK runs of that workload on that core type (all thread counts).
P/E IPC ratio > 1 means the workload retires more instructions per cycle on P-cores.
Power/energy: package-level, measured over a window that includes idle time (comparable between runs, not workload-only).

| Workload | Kind | Runs (P/E) | Threads tested (P / E) | Span (s) | IPC P | IPC E | P/E IPC | Active threads (mean) | Timeline bins / run | Power P (W) | Power E (W) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7zip | real app | 14/20 | 1,4,8 / 1,4,8,16 | 16.7 | 1.85 | 1.60 | 1.16 | 69.1 | 33 | 32.1 | 24.8 |
| zstd | real app | 15/19 | 1,4,8 / 1,4,8,16 | 32.0 | 2.77 | 2.14 | 1.29 | 6.7 | 64 | 33.9 | 22.9 |
| x264 | real app | 15/19 | 1,4,8 / 1,4,8,16 | 31.2 | 3.64 | 2.83 | 1.29 | 10.8 | 62 | 36.6 | 25.5 |
| sqlite | real app | 14/19 | 1,4,8 / 1,4,8,16 | 31.3 | 3.16 | 2.36 | 1.34 | 8.3 | 63 | 27.7 | 19.0 |
| compute | Python micro | 5/5 | 1 / 1 | 30.5 | 6.97 | 5.81 | 1.20 | 1.3 | 61 | 26.9 | 18.3 |
| memory | Python micro | 5/5 | 1 / 1 | 32.2 | 0.89 | 1.01 | 0.88 | 1.3 | 64 | 23.1 | 16.7 |
| mixed | Python micro | 5/5 | 1 / 1 | 32.2 | 3.82 | 4.06 | 0.94 | 1.2 | 64 | 24.7 | 17.9 |
| io | Python micro | 5/5 | 1 / 1 | 30.5 | 2.76 | 2.56 | 1.08 | 1.2 | 61 | 20.0 | 15.4 |

Total: 175 runs, 10,007 timeline bins (0.5 s each, per thread rows not counted), 125.0 trillion instructions sampled.
