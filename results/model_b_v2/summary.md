# Model B v2 — results

Seeds: [0, 1, 2, 3, 4] · split by run_id (test 15%, val 15%) · horizon 2 windows · anchors need 8 windows of past · hysteresis MIN_DURATION=3
Shared test anchors: every model is scored on the same anchors (17637 anchors from 417/908 threads, 141 runs).

Transition rate: raw 17.2%, hyst 7.1%. Raw transitions that land in a phase lasting < 3 windows (flicker): 55.6%.

All numbers are % (mean ± std over seeds). `trans_recall` is the 'transition accuracy' that model_b.py reports.

## Label definition: `raw`

| model | what it adds | overall | same-phase | trans recall | trans precision | trans F1 | change detected | n trans (test) |
|---|---|---|---|---|---|---|---|---|
| persistence | always predict the current phase | 87.8 ± 3.9 | 100.0 ± 0.0 | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 314 |
| markov1 | most frequent next phase given current phase (train counts) | 87.8 ± 3.9 | 100.0 ± 0.0 | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 314 |
| markov2 | same, given current phase + last two raw phases | 88.8 ± 3.6 | 98.1 ± 0.8 | **21.2 ± 5.6** | 54.9 ± 12.8 | 30.6 ± 7.8 | 25.0 ± 3.6 | 314 |
| mlp_emb_ce | v1 MLP: anchor embedding only, plain cross-entropy | 92.6 ± 2.8 | 97.4 ± 1.4 | **58.5 ± 4.7** | 74.3 ± 3.3 | 65.4 ± 3.5 | 61.1 ± 5.3 | 314 |
| lstm4_emb_ce | v1 LSTM-4: 4 embeddings (2 s), plain cross-entropy | 93.0 ± 2.5 | 97.5 ± 0.9 | **60.6 ± 5.8** | 74.9 ± 3.7 | 66.9 ± 4.9 | 63.3 ± 7.0 | 314 |
| lstm6_emb_ce | v1 LSTM-6: 6 embeddings (3 s), plain cross-entropy | 92.5 ± 1.9 | 97.1 ± 0.8 | **60.0 ± 6.2** | 70.6 ± 8.2 | 64.3 ± 1.7 | 63.2 ± 7.8 | 314 |
| lstm8_emb_ce | v1 LSTM-8: 8 embeddings (4 s), plain cross-entropy | 92.3 ± 1.9 | 97.1 ± 0.6 | **58.7 ± 6.2** | 70.0 ± 7.5 | 63.3 ± 1.6 | 61.6 ± 6.8 | 314 |
| mlp_ctx_ce | + centroid distances, margin, current/committed phase, deltas, raw window | 93.9 ± 2.2 | 97.9 ± 1.0 | **64.9 ± 4.2** | 79.4 ± 3.7 | 71.4 ± 4.0 | 67.1 ± 4.8 | 314 |
| mlp_ctx_weighted | + transition samples weighted x4 in the loss | 92.4 ± 2.8 | 94.6 ± 2.3 | **76.2 ± 5.5** | 64.7 ± 5.1 | 69.9 ± 5.0 | 79.8 ± 6.4 | 314 |
| lstm4_ctx_weighted | LSTM over 4 steps of context features, x4 weights | 90.2 ± 2.8 | 92.9 ± 2.0 | **71.1 ± 4.5** | 55.7 ± 4.5 | 62.3 ± 3.1 | 75.3 ± 5.7 | 314 |
| lstm6_ctx_weighted | LSTM over 6 steps of context features, x4 weights | 89.8 ± 1.5 | 92.5 ± 0.5 | **70.8 ± 3.7** | 53.7 ± 8.4 | 60.6 ± 5.2 | 75.4 ± 5.4 | 314 |
| lstm8_ctx_weighted | LSTM over 8 steps of context features, x4 weights | 89.9 ± 2.8 | 92.6 ± 2.1 | **70.2 ± 2.1** | 54.6 ± 4.2 | 61.3 ± 3.0 | 74.8 ± 3.2 | 314 |
| mlp_ctx_twohead | change? head + destination head, threshold tuned on val | 92.4 ± 2.5 | 96.0 ± 1.1 | **68.3 ± 9.6** | 67.1 ± 5.0 | 67.2 ± 3.3 | 71.7 ± 11.2 | 314 |
| lstm6_ctx_twohead | two-head on 6 steps of context features | 90.5 ± 3.6 | 94.7 ± 2.9 | **60.3 ± 12.0** | 60.3 ± 6.6 | 59.4 ± 4.7 | 65.3 ± 14.0 | 314 |
| gbdt_ctx_weighted | gradient-boosted trees on context features, x4 weights | 92.7 ± 1.8 | 96.1 ± 1.0 | **69.1 ± 6.2** | 68.4 ± 6.9 | 68.3 ± 2.3 | 71.9 ± 6.8 | 314 |

## Label definition: `hyst`

| model | what it adds | overall | same-phase | trans recall | trans precision | trans F1 | change detected | n trans (test) |
|---|---|---|---|---|---|---|---|---|
| persistence | always predict the current phase | 94.4 ± 1.9 | 100.0 ± 0.0 | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 145 |
| markov1 | most frequent next phase given current phase (train counts) | 94.4 ± 1.9 | 100.0 ± 0.0 | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 145 |
| markov2 | same, given current phase + last two raw phases | 94.7 ± 2.2 | 99.1 ± 0.6 | **22.7 ± 8.9** | 60.3 ± 14.2 | 32.4 ± 9.8 | 24.1 ± 9.6 | 145 |
| mlp_emb_ce | v1 MLP: anchor embedding only, plain cross-entropy | 94.4 ± 2.9 | 96.6 ± 2.4 | **57.9 ± 7.6** | 52.1 ± 10.5 | 54.4 ± 7.8 | 60.3 ± 7.6 | 145 |
| lstm4_emb_ce | v1 LSTM-4: 4 embeddings (2 s), plain cross-entropy | 94.2 ± 2.3 | 96.9 ± 1.5 | **48.4 ± 7.4** | 48.2 ± 7.2 | 48.1 ± 6.5 | 49.5 ± 7.5 | 145 |
| lstm6_emb_ce | v1 LSTM-6: 6 embeddings (3 s), plain cross-entropy | 94.3 ± 2.4 | 97.0 ± 1.6 | **48.3 ± 11.3** | 49.6 ± 9.6 | 48.5 ± 8.8 | 49.8 ± 11.4 | 145 |
| lstm8_emb_ce | v1 LSTM-8: 8 embeddings (4 s), plain cross-entropy | 94.3 ± 2.4 | 97.0 ± 1.7 | **48.6 ± 8.6** | 50.2 ± 7.2 | 48.7 ± 4.5 | 51.0 ± 9.1 | 145 |
| mlp_ctx_ce | + centroid distances, margin, current/committed phase, deltas, raw window | 95.7 ± 1.8 | 98.4 ± 1.0 | **49.0 ± 4.2** | 66.3 ± 7.8 | 56.2 ± 4.2 | 50.3 ± 3.8 | 145 |
| mlp_ctx_weighted | + transition samples weighted x4 in the loss | 94.6 ± 2.1 | 96.2 ± 1.6 | **68.5 ± 5.4** | 51.5 ± 3.2 | 58.7 ± 3.4 | 70.6 ± 4.7 | 145 |
| lstm4_ctx_weighted | LSTM over 4 steps of context features, x4 weights | 94.0 ± 2.8 | 95.9 ± 2.3 | **61.5 ± 6.7** | 48.3 ± 8.8 | 53.8 ± 6.5 | 65.6 ± 7.4 | 145 |
| lstm6_ctx_weighted | LSTM over 6 steps of context features, x4 weights | 93.8 ± 2.4 | 95.8 ± 2.0 | **59.3 ± 8.0** | 45.8 ± 6.5 | 51.4 ± 5.8 | 62.9 ± 8.6 | 145 |
| lstm8_ctx_weighted | LSTM over 8 steps of context features, x4 weights | 93.4 ± 2.0 | 95.4 ± 1.8 | **57.8 ± 8.3** | 41.8 ± 5.5 | 48.3 ± 5.5 | 62.5 ± 7.9 | 145 |
| mlp_ctx_twohead | change? head + destination head, threshold tuned on val | 90.9 ± 4.2 | 93.5 ± 4.0 | **50.0 ± 13.9** | 33.7 ± 10.0 | 39.0 ± 8.5 | 52.1 ± 13.9 | 145 |
| lstm6_ctx_twohead | two-head on 6 steps of context features | 91.1 ± 2.6 | 94.3 ± 2.7 | **40.0 ± 19.5** | 27.9 ± 5.3 | 31.1 ± 6.8 | 43.0 ± 22.3 | 145 |
| gbdt_ctx_weighted | gradient-boosted trees on context features, x4 weights | 95.6 ± 1.3 | 97.9 ± 0.8 | **57.4 ± 5.1** | 61.7 ± 10.3 | 58.8 ± 4.3 | 57.7 ± 5.0 | 145 |
