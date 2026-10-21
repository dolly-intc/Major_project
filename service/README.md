# service/ — the thread placement service (Stage 3, Month 4)

> ## ⚠ STANDARD LIBRARY ONLY
>
> This runs on the test machine alongside `collect/`. Same rule, same reason — see
> `collect/README.md`. `tests/test_stdlib_only.py` covers this folder too.

The background service that samples counters for running threads, runs the trained models, and
applies placement decisions.

**Placement mechanism is settled** (`PROJECT_CONTEXT.md` §5): steer threads using
`SetThreadInformation` with `ThreadPowerThrottling`, or `SetThreadSelectedCpuSets` as a softer
option. These cooperate with Intel Thread Director.

**Never use `SetThreadAffinityMask` as the placement mechanism.** A hard affinity mask overrides
Thread Director and disables kernel load balancing for that thread, which makes results both
worse and unrepresentative. It is fine inside isolated micro-benchmarks, where it is used
deliberately — `collect/compare_workloads.py` does exactly that — but not in the service.

Empty until Stage 3. Requires trained models from `analysis/`, which requires the Stage 1
dataset, which requires the harness in `collect/`.

**Open dependency question:** running trained models here means either transferring PyTorch as
offline wheels (untested — `PROJECT_CONTEXT.md` open item 2) or exporting the trained weights to
JSON and implementing the inference path in the standard library. The models are small, so the
second option is very achievable and carries no dependency risk. Decide before Stage 3 starts.
