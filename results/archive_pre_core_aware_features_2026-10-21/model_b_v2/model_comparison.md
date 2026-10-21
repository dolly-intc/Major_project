# Model B — model comparison (presentation table)

Same shared test anchors for every row; run-level split; mean ± std over 5 seeds. Labels: hysteresis (phase changes shorter than 1.5 s treated as noise). Transition = the phase 1 s ahead differs from the current committed phase.

| model | what it is | trans recall | trans precision | trans F1 | overall | params | train time | service port |
|---|---|---|---|---|---|---|---|---|
| persistence | always predict the current phase | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.0 ± 0.0 | 91.7 ± 2.2 | — | — | trivial (lookup) |
| markov1 | most frequent next phase given current phase (train counts) | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.0 ± 0.0 | 91.7 ± 2.2 | — | — | trivial (lookup) |
| markov2 | same, given current phase + last two raw phases | **34.8 ± 5.1** | 72.2 ± 4.8 | 46.7 ± 4.9 | 93.5 ± 1.7 | — | — | trivial (lookup) |
| mlp_emb_ce | v1 MLP: anchor embedding only, plain cross-entropy | **56.3 ± 2.7** | 52.7 ± 7.2 | 54.1 ± 2.9 | 92.3 ± 1.5 | 1,608 | 9 s | easy (3 matrix multiplies) |
| lstm4_emb_ce | v1 LSTM-4: 4 embeddings (2 s), plain cross-entropy | **46.4 ± 2.2** | 53.0 ± 6.6 | 49.2 ± 2.1 | 92.3 ± 1.6 | 14,088 | 21 s | moderate (hand-written gates, state per thread) |
| lstm6_emb_ce | v1 LSTM-6: 6 embeddings (3 s), plain cross-entropy | **44.8 ± 1.5** | 50.9 ± 8.5 | 47.4 ± 4.1 | 92.0 ± 1.4 | 14,088 | 21 s | moderate (hand-written gates, state per thread) |
| lstm8_emb_ce | v1 LSTM-8: 8 embeddings (4 s), plain cross-entropy | **45.3 ± 2.4** | 50.5 ± 9.4 | 47.5 ± 5.0 | 91.9 ± 1.5 | 14,088 | 22 s | moderate (hand-written gates, state per thread) |
| mlp_ctx_ce | + centroid distances, margin, current/committed phase, deltas, raw window | **43.5 ± 7.6** | 72.7 ± 4.6 | 54.3 ± 7.3 | 94.1 ± 1.6 | 4,488 | 6 s | easy (3 matrix multiplies) |
| **mlp_ctx_weighted** (chosen) | + transition samples weighted x4 in the loss | **66.8 ± 2.2** | 51.4 ± 6.8 | 57.9 ± 4.3 | 92.0 ± 2.3 | 4,488 | 5 s | easy (3 matrix multiplies) |
| lstm4_ctx_weighted | LSTM over 4 steps of context features, x4 weights | **60.8 ± 3.6** | 44.9 ± 8.0 | 51.5 ± 6.4 | 90.8 ± 2.1 | 17,416 | 13 s | moderate (hand-written gates, state per thread) |
| lstm6_ctx_weighted | LSTM over 6 steps of context features, x4 weights | **63.2 ± 4.7** | 44.0 ± 7.6 | 51.7 ± 6.3 | 90.4 ± 2.5 | 17,416 | 15 s | moderate (hand-written gates, state per thread) |
| lstm8_ctx_weighted | LSTM over 8 steps of context features, x4 weights | **62.5 ± 4.4** | 45.3 ± 6.9 | 52.4 ± 6.0 | 90.9 ± 1.9 | 17,416 | 16 s | moderate (hand-written gates, state per thread) |
| mlp_ctx_twohead | change? head + destination head, threshold tuned on val | **63.6 ± 11.5** | 35.1 ± 6.2 | 44.2 ± 3.1 | 87.3 ± 1.3 | 4,521 | 5 s | easy (3 matrix multiplies) |
| lstm6_ctx_twohead | two-head on 6 steps of context features | **63.0 ± 23.4** | 26.5 ± 8.9 | 35.8 ± 11.3 | 81.9 ± 6.6 | 17,449 | 12 s | moderate (hand-written gates, state per thread) |
| gbdt_ctx_weighted | gradient-boosted trees on context features, x4 weights | **49.0 ± 4.5** | 52.5 ± 22.3 | 48.3 ± 16.2 | 90.6 ± 5.3 | 36,535 leaves | 20 s | hard (export + walk 200 x 8 trees) |
