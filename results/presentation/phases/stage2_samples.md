# Stage 2 samples (after windowing)

A window = 4 consecutive 0.5 s bins (2 s) of one thread; windows slide by one bin. Each thread contributes its longest gap-free stretch.

| workload | windows | runs | thread_runs | share_of_windows | seconds_of_thread_time |
|---|---|---|---|---|---|
| 7zip | 1670 | 24 | 439 | 0.07 | 835 |
| zstd | 2046 | 32 | 44 | 0.086 | 1023 |
| x264 | 15247 | 34 | 263 | 0.638 | 7624 |
| sqlite | 2985 | 33 | 129 | 0.125 | 1492 |
| compute | 570 | 10 | 10 | 0.024 | 285 |
| memory | 545 | 9 | 9 | 0.023 | 272 |
| mixed | 606 | 10 | 10 | 0.025 | 303 |
| io | 228 | 4 | 4 | 0.01 | 114 |
| total | 23897 | 156 | 908 | 1.0 | 11948 |
