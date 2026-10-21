# Model B v2 — results

Seeds: [0, 1, 2, 3, 4] · split by run_id (test 15%, val 15%) · horizon 2 windows · anchors need 8 windows of past · hysteresis MIN_DURATION=3
Shared test anchors: every model is scored on the same anchors (20860 anchors from 485/1188 threads, 150 runs).

Transition rate: raw 11.4%, hyst 8.1%. Raw transitions that land in a phase lasting < 3 windows (flicker): 48.5%.

All numbers are % (mean ± std over seeds). `trans_recall` is the 'transition accuracy' that model_b.py reports.

## Label definition: `raw`

| model | what it adds | overall | same-phase | trans recall | trans precision | trans F1 | change detected | n trans (test) |
|---|---|---|---|---|---|---|---|---|
| persistence | always predict the current phase | 89.0 ± 2.2 | 100.0 ± 0.0 | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 368 |
| markov1 | most frequent next phase given current phase (train counts) | 89.0 ± 2.2 | 99.7 ± 0.4 | **2.3 ± 3.3** | 17.1 ± 23.4 | 4.0 ± 5.7 | 2.7 ± 3.9 | 368 |
| markov2 | same, given current phase + last two raw phases | 89.0 ± 2.3 | 98.5 ± 0.8 | **13.3 ± 6.5** | 45.9 ± 4.2 | 19.9 ± 8.1 | 16.0 ± 7.4 | 368 |
| mlp_emb_ce | v1 MLP: anchor embedding only, plain cross-entropy | 90.6 ± 2.3 | 97.7 ± 0.9 | **32.7 ± 6.4** | 61.7 ± 5.9 | 42.5 ± 6.0 | 35.4 ± 6.5 | 368 |
| lstm4_emb_ce | v1 LSTM-4: 4 embeddings (2 s), plain cross-entropy | 90.6 ± 1.8 | 97.3 ± 0.7 | **37.1 ± 7.7** | 58.9 ± 4.3 | 45.1 ± 6.2 | 40.3 ± 7.6 | 368 |
| lstm6_emb_ce | v1 LSTM-6: 6 embeddings (3 s), plain cross-entropy | 90.6 ± 2.1 | 96.6 ± 0.8 | **42.4 ± 5.6** | 57.6 ± 3.9 | 48.8 ± 4.8 | 46.2 ± 5.0 | 368 |
| lstm8_emb_ce | v1 LSTM-8: 8 embeddings (4 s), plain cross-entropy | 90.3 ± 1.9 | 96.1 ± 0.8 | **43.4 ± 3.9** | 55.4 ± 2.0 | 48.6 ± 2.9 | 46.7 ± 3.5 | 368 |
| mlp_ctx_ce | + centroid distances, margin, current/committed phase, deltas, raw window | 91.5 ± 2.0 | 97.6 ± 0.9 | **42.1 ± 5.4** | 66.2 ± 3.9 | 51.3 ± 3.8 | 44.7 ± 6.4 | 368 |
| mlp_ctx_weighted | + transition samples weighted x4 in the loss | 88.4 ± 3.0 | 91.2 ± 2.5 | **65.4 ± 7.1** | 47.0 ± 3.0 | 54.6 ± 3.8 | 69.3 ± 7.4 | 368 |
| lstm4_ctx_weighted | LSTM over 4 steps of context features, x4 weights | 85.8 ± 4.6 | 88.5 ± 4.6 | **63.8 ± 3.3** | 40.4 ± 4.8 | 49.4 ± 3.9 | 69.1 ± 4.3 | 368 |
| lstm6_ctx_weighted | LSTM over 6 steps of context features, x4 weights | 85.9 ± 3.7 | 88.4 ± 3.3 | **65.1 ± 4.6** | 40.2 ± 4.0 | 49.7 ± 4.1 | 70.2 ± 4.6 | 368 |
| lstm8_ctx_weighted | LSTM over 8 steps of context features, x4 weights | 86.2 ± 4.0 | 88.6 ± 3.9 | **65.9 ± 5.0** | 41.0 ± 4.1 | 50.5 ± 4.0 | 71.6 ± 5.3 | 368 |
| mlp_ctx_twohead | change? head + destination head, threshold tuned on val | 84.9 ± 2.8 | 87.8 ± 2.0 | **62.8 ± 9.3** | 37.5 ± 5.0 | 46.7 ± 4.9 | 67.2 ± 8.3 | 368 |
| lstm6_ctx_twohead | two-head on 6 steps of context features | 82.3 ± 3.8 | 85.0 ± 4.1 | **60.4 ± 8.0** | 32.1 ± 1.7 | 41.7 ± 2.2 | 67.4 ± 8.5 | 368 |
| gbdt_ctx_weighted | gradient-boosted trees on context features, x4 weights | 90.7 ± 2.4 | 95.7 ± 1.1 | **50.8 ± 5.2** | 57.3 ± 3.3 | 53.8 ± 3.9 | 54.0 ± 5.1 | 368 |

## Label definition: `hyst`

| model | what it adds | overall | same-phase | trans recall | trans precision | trans F1 | change detected | n trans (test) |
|---|---|---|---|---|---|---|---|---|
| persistence | always predict the current phase | 91.7 ± 2.2 | 100.0 ± 0.0 | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 275 |
| markov1 | most frequent next phase given current phase (train counts) | 91.7 ± 2.2 | 100.0 ± 0.0 | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 275 |
| markov2 | same, given current phase + last two raw phases | 93.5 ± 1.7 | 98.9 ± 0.3 | **34.8 ± 5.1** | 72.2 ± 4.8 | 46.7 ± 4.9 | 35.2 ± 5.1 | 275 |
| mlp_emb_ce | v1 MLP: anchor embedding only, plain cross-entropy | 92.3 ± 1.5 | 95.6 ± 1.1 | **56.3 ± 2.7** | 52.7 ± 7.2 | 54.1 ± 2.9 | 57.7 ± 2.9 | 275 |
| lstm4_emb_ce | v1 LSTM-4: 4 embeddings (2 s), plain cross-entropy | 92.3 ± 1.6 | 96.5 ± 0.8 | **46.4 ± 2.2** | 53.0 ± 6.6 | 49.2 ± 2.1 | 48.3 ± 2.5 | 275 |
| lstm6_emb_ce | v1 LSTM-6: 6 embeddings (3 s), plain cross-entropy | 92.0 ± 1.4 | 96.3 ± 0.7 | **44.8 ± 1.5** | 50.9 ± 8.5 | 47.4 ± 4.1 | 47.0 ± 0.7 | 275 |
| lstm8_emb_ce | v1 LSTM-8: 8 embeddings (4 s), plain cross-entropy | 91.9 ± 1.5 | 96.1 ± 0.8 | **45.3 ± 2.4** | 50.5 ± 9.4 | 47.5 ± 5.0 | 46.6 ± 1.7 | 275 |
| mlp_ctx_ce | + centroid distances, margin, current/committed phase, deltas, raw window | 94.1 ± 1.6 | 98.7 ± 0.4 | **43.5 ± 7.6** | 72.7 ± 4.6 | 54.3 ± 7.3 | 45.1 ± 6.9 | 275 |
| mlp_ctx_weighted | + transition samples weighted x4 in the loss | 92.0 ± 2.3 | 94.2 ± 2.1 | **66.8 ± 2.2** | 51.4 ± 6.8 | 57.9 ± 4.3 | 68.7 ± 2.7 | 275 |
| lstm4_ctx_weighted | LSTM over 4 steps of context features, x4 weights | 90.8 ± 2.1 | 93.4 ± 1.7 | **60.8 ± 3.6** | 44.9 ± 8.0 | 51.5 ± 6.4 | 63.3 ± 3.4 | 275 |
| lstm6_ctx_weighted | LSTM over 6 steps of context features, x4 weights | 90.4 ± 2.5 | 92.8 ± 2.5 | **63.2 ± 4.7** | 44.0 ± 7.6 | 51.7 ± 6.3 | 65.7 ± 4.7 | 275 |
| lstm8_ctx_weighted | LSTM over 8 steps of context features, x4 weights | 90.9 ± 1.9 | 93.4 ± 1.6 | **62.5 ± 4.4** | 45.3 ± 6.9 | 52.4 ± 6.0 | 65.1 ± 3.9 | 275 |
| mlp_ctx_twohead | change? head + destination head, threshold tuned on val | 87.3 ± 1.3 | 89.6 ± 2.1 | **63.6 ± 11.5** | 35.1 ± 6.2 | 44.2 ± 3.1 | 65.6 ± 11.2 | 275 |
| lstm6_ctx_twohead | two-head on 6 steps of context features | 81.9 ± 6.6 | 83.7 ± 8.3 | **63.0 ± 23.4** | 26.5 ± 8.9 | 35.8 ± 11.3 | 66.5 ± 24.2 | 275 |
| gbdt_ctx_weighted | gradient-boosted trees on context features, x4 weights | 90.6 ± 5.3 | 94.4 ± 6.2 | **49.0 ± 4.5** | 52.5 ± 22.3 | 48.3 ± 16.2 | 55.4 ± 10.2 | 275 |
