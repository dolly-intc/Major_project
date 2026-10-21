# analysis/ — modelling and analysis (Stage 2, Months 2–3)

Runs on the **dev machine** with the full scientific Python stack. No restrictions here — this
is the one part of the project with unrestricted dependencies, and it carries the academic
weight of the thesis.

Works entirely from saved CSVs in `data/raw/`. It never needs the hybrid hardware, which is why
the project can survive the test machine being reclaimed.

## The three models

**Model A — phase discovery (unsupervised).** An autoencoder compresses sliding windows of
counter readings into a low-dimensional embedding; clustering over that embedding identifies
recurring execution phases.

*Unsupervised is a settled decision.* The "correct" core type for a thread depends partly on
system-wide context — other running work, power budget, thermal state — that is not visible in
thread-local counters. Manual labels would impose a target the inputs do not determine.

**Model B — phase transition prediction.** A temporal convolutional network or LSTM over recent
embedding history, predicting the phase 50–200 ms ahead. Prediction rather than reaction,
because thread migration has real latency and cache-warming cost; reacting to an observed phase
change is already too late.

**Model C — core-type preference.** A regression head mapping a phase embedding to a continuous
preference score. Continuous rather than binary so the score doubles as a confidence measure —
the service acts only when it is far from neutral, which suppresses pointless migrations.

Plus a gradient-boosted-tree baseline on raw counters, to show what the learned representation
actually adds, and a feature ablation to identify which counters matter.

Targets: ≥75% phase transition accuracy, ≥80% core-preference accuracy.
