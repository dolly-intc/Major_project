# Stage 2 samples (after windowing)

A window = 4 consecutive 0.5 s bins (2 s) of one thread; windows slide by one bin. Each thread contributes its longest gap-free stretch.

| workload | windows | runs | thread_runs | share_of_windows | seconds_of_thread_time |
|---|---|---|---|---|---|
| 7zip | 2264 | 25 | 625 | 0.079 | 1132 |
| zstd | 2174 | 34 | 46 | 0.076 | 1087 |
| x264 | 18670 | 34 | 322 | 0.653 | 9335 |
| sqlite | 3142 | 33 | 155 | 0.11 | 1571 |
| compute | 570 | 10 | 10 | 0.02 | 285 |
| memory | 605 | 10 | 10 | 0.021 | 302 |
| mixed | 606 | 10 | 10 | 0.021 | 303 |
| io | 570 | 10 | 10 | 0.02 | 285 |
| total | 28601 | 166 | 1188 | 1.0 | 14300 |
