# dashboard/ — live demonstration dashboard (Stage 5, Months 5–6)

The demo deliverable: a live view of core activity split by type, thread placement, the current
phase embedding, model predictions, and instantaneous power draw.

This matters more than its stage number suggests. Success for this project is defined as
*credits plus a demo that impresses the guide* — so this folder is a large part of the grade.

## Decide the dependency approach before building

**The dashboard runs on the test machine**, where `pip` does not work. Two options:

1. **Streamlit or Dash**, transferred as offline wheels. Faster to write, but the wheel transfer
   is listed as untested in `PROJECT_CONTEXT.md` open item 2 — putting an unverified dependency
   under the single deliverable that decides the grade.
2. **Standard library only** — `http.server` serving a static page, with the placement service
   writing JSON that the page polls, and a charting library vendored into the repo as a plain
   `.js` file. A vendored `.js` asset is not a `pip install`, so this sidesteps the restriction
   entirely and inherits zero dependency risk.

Option 2 is the safer choice and is not much more work for a fixed, known set of charts.
Worth settling early rather than during Month 5.

## Demonstration plan

**Part 1 — live behavioural analysis.** The dashboard runs while applications are launched,
showing each thread's current phase, the predicted upcoming phase, and the core-type
recommendation.

**Part 2 — controlled A/B comparison.** A heavy background job (video encode) runs alongside a
**latency probe**: a periodic task waking every 10 ms, doing a small fixed unit of work, and
recording wake-up lateness. With the service disabled the probe shows pronounced latency spikes;
with it enabled the background job is steered to E-cores and the probe's distribution tightens.

A numerical probe is used instead of an animation or frame-time counter because the test machine
is reached over Remote Desktop, where graphics timing measures the network rather than the
scheduler.
