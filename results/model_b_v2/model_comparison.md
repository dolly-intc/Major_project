# Model B — model comparison (presentation table)

Same shared test anchors for every row; run-level split; mean ± std over 5 seeds. Labels: hysteresis (phase changes shorter than 1.5 s treated as noise). Transition = the phase 1 s ahead differs from the current committed phase.

| model | what it is | trans recall | trans precision | trans F1 | overall | params | train time | service port |
|---|---|---|---|---|---|---|---|---|
| persistence | always predict the current phase | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.0 ± 0.0 | 94.4 ± 1.9 | — | — | trivial (lookup) |
| markov1 | most frequent next phase given current phase (train counts) | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.0 ± 0.0 | 94.4 ± 1.9 | — | — | trivial (lookup) |
| markov2 | same, given current phase + last two raw phases | **22.7 ± 8.9** | 60.3 ± 14.2 | 32.4 ± 9.8 | 94.7 ± 2.2 | — | — | trivial (lookup) |
| mlp_emb_ce | v1 MLP: anchor embedding only, plain cross-entropy | **57.9 ± 7.6** | 52.1 ± 10.5 | 54.4 ± 7.8 | 94.4 ± 2.9 | 1,476 | 4 s | easy (3 matrix multiplies) |
| lstm4_emb_ce | v1 LSTM-4: 4 embeddings (2 s), plain cross-entropy | **48.4 ± 7.4** | 48.2 ± 7.2 | 48.1 ± 6.5 | 94.2 ± 2.3 | 13,956 | 9 s | moderate (hand-written gates, state per thread) |
| lstm6_emb_ce | v1 LSTM-6: 6 embeddings (3 s), plain cross-entropy | **48.3 ± 11.3** | 49.6 ± 9.6 | 48.5 ± 8.8 | 94.3 ± 2.4 | 13,956 | 9 s | moderate (hand-written gates, state per thread) |
| lstm8_emb_ce | v1 LSTM-8: 8 embeddings (4 s), plain cross-entropy | **48.6 ± 8.6** | 50.2 ± 7.2 | 48.7 ± 4.5 | 94.3 ± 2.4 | 13,956 | 11 s | moderate (hand-written gates, state per thread) |
| mlp_ctx_ce | + centroid distances, margin, current/committed phase, deltas, raw window | **49.0 ± 4.2** | 66.3 ± 7.8 | 56.2 ± 4.2 | 95.7 ± 1.8 | 3,716 | 3 s | easy (3 matrix multiplies) |
| **mlp_ctx_weighted** (chosen) | + transition samples weighted x4 in the loss | **68.5 ± 5.4** | 51.5 ± 3.2 | 58.7 ± 3.4 | 94.6 ± 2.1 | 3,716 | 2 s | easy (3 matrix multiplies) |
| lstm4_ctx_weighted | LSTM over 4 steps of context features, x4 weights | **61.5 ± 6.7** | 48.3 ± 8.8 | 53.8 ± 6.5 | 94.0 ± 2.8 | 15,748 | 6 s | moderate (hand-written gates, state per thread) |
| lstm6_ctx_weighted | LSTM over 6 steps of context features, x4 weights | **59.3 ± 8.0** | 45.8 ± 6.5 | 51.4 ± 5.8 | 93.8 ± 2.4 | 15,748 | 7 s | moderate (hand-written gates, state per thread) |
| lstm8_ctx_weighted | LSTM over 8 steps of context features, x4 weights | **57.8 ± 8.3** | 41.8 ± 5.5 | 48.3 ± 5.5 | 93.4 ± 2.0 | 15,748 | 9 s | moderate (hand-written gates, state per thread) |
| mlp_ctx_twohead | change? head + destination head, threshold tuned on val | **50.0 ± 13.9** | 33.7 ± 10.0 | 39.0 ± 8.5 | 90.9 ± 4.2 | 3,749 | 3 s | easy (3 matrix multiplies) |
| lstm6_ctx_twohead | two-head on 6 steps of context features | **40.0 ± 19.5** | 27.9 ± 5.3 | 31.1 ± 6.8 | 91.1 ± 2.6 | 15,781 | 7 s | moderate (hand-written gates, state per thread) |
| gbdt_ctx_weighted | gradient-boosted trees on context features, x4 weights | **57.4 ± 5.1** | 61.7 ± 10.3 | 58.8 ± 4.3 | 95.6 ± 1.3 | 22,170 leaves | 10 s | hard (export + walk 200 x 8 trees) |
