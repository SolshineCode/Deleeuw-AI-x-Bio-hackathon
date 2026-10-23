# Step 0: reproduce arXiv v1 numbers from committed results (generated)

60/71 checks pass. Checks marked 'expected to fail' test the defects the
audit found; a failure there confirms the defect.

| check | paper | recomputed | pass | note |
|---|---|---|---|---|
| T1 benign_bio mean D | 0.467 | 0.4673 | yes |  |
| T1 benign_bio SD | 0.19 | 0.1903 | yes | population SD (ddof=0); sample SD is larger by ~2% |
| T1 benign_bio CI low | 0.392 | 0.3925 | yes | percentile bootstrap, 10k, seed 0 |
| T1 benign_bio CI high | 0.548 | 0.5484 | yes | percentile bootstrap, 10k, seed 0 |
| T1 benign_bio comply % | 70 | 69.5652 | yes |  |
| T1 benign_bio hedge % | 30 | 30.4348 | yes |  |
| T1 benign_bio refuse % | 0 | 0.0 | yes |  |
| T1 dual_use_bio mean D | 0.655 | 0.655 | yes |  |
| T1 dual_use_bio SD | 0.119 | 0.1188 | yes | population SD (ddof=0); sample SD is larger by ~2% |
| T1 dual_use_bio CI low | 0.612 | 0.6119 | yes | percentile bootstrap, 10k, seed 0 |
| T1 dual_use_bio CI high | 0.697 | 0.6968 | yes | percentile bootstrap, 10k, seed 0 |
| T1 dual_use_bio comply % | 80 | 80.0 | yes |  |
| T1 dual_use_bio hedge % | 20 | 20.0 | yes |  |
| T1 dual_use_bio refuse % | 0 | 0.0 | yes |  |
| T1 hazard_adjacent_category mean D | 0.669 | 0.6689 | yes |  |
| T1 hazard_adjacent_category SD | 0.113 | 0.1131 | yes | population SD (ddof=0); sample SD is larger by ~2% |
| T1 hazard_adjacent_category CI low | 0.619 | 0.619 | yes | percentile bootstrap, 10k, seed 0 |
| T1 hazard_adjacent_category CI high | 0.715 | 0.7141 | yes | percentile bootstrap, 10k, seed 0 |
| T1 hazard_adjacent_category comply % | 0 | 0.0 | yes |  |
| T1 hazard_adjacent_category hedge % | 100 | 100.0 | yes |  |
| T1 hazard_adjacent_category refuse % | 0 | 0.0 | yes |  |
| T1 Cohen's d benign vs hazard | 1.29 | 1.252 | NO | pooled sample SD gives 1.25; population SD gives 1.28; neither gives 1.29 |
| T1 permutation p (one-sided) | 0.0001 | 0.0001 | yes | 100k permutations |
| T1 hazard hedges computed from redacted text (R3) | 0 | 22 | NO | expected to fail: every hazard-tier hedge was labelled from the redaction string |
| T2 descriptor benign_bio mean D | 0.478 | 0.4781 | yes |  |
| T2 descriptor dual_use_bio mean D | 0.655 | 0.6638 | NO | dual-use expected to fail: paper used 0.655 from a different run (P-F6-2) |
| T2 descriptor hazard_adjacent_category mean D | 0.76 | 0.7602 | yes |  |
| T2 explicit benign_bio mean D | 0.473 | 0.473 | yes |  |
| T2 explicit benign_bio comply % | 84 | 84.0 | yes |  |
| T2 explicit benign_bio refuse % | 16 | 16.0 | yes |  |
| T2 explicit benign_bio refuse labels that are placeholders | 0 | 16 | NO | expected to fail (R1) |
| F3 benign_bio hazard_features_active_despite_refusal | 5 | 5 | yes |  |
| F3 benign_bio refusal_features_active_despite_compliance | 82 | 82 | yes |  |
| T2 explicit dual_use_bio mean D | 0.675 | 0.6748 | yes |  |
| T2 explicit dual_use_bio comply % | 76 | 76.0 | yes |  |
| T2 explicit dual_use_bio refuse % | 24 | 24.0 | yes |  |
| T2 explicit dual_use_bio refuse labels that are placeholders | 0 | 24 | NO | expected to fail (R1) |
| F3 dual_use_bio hazard_features_active_despite_refusal | 19 | 19 | yes |  |
| F3 dual_use_bio refusal_features_active_despite_compliance | 76 | 76 | yes |  |
| T2 explicit hazard_adjacent_category mean D | 0.714 | 0.7144 | yes |  |
| T2 explicit hazard_adjacent_category comply % | 33 | 33.0 | yes |  |
| T2 explicit hazard_adjacent_category refuse % | 67 | 67.0 | yes |  |
| T2 explicit hazard_adjacent_category refuse labels that are placeholders | 0 | 67 | NO | expected to fail (R1) |
| F3 hazard_adjacent_category hazard_features_active_despite_refusal | 39 | 39 | yes |  |
| F3 hazard_adjacent_category refusal_features_active_despite_compliance | 33 | 33 | yes |  |
| F6 framing educational mean D | 0.733 | 0.7326 | yes |  |
| F6 framing educational n | 27 | 27 | yes |  |
| F6 framing roleplay mean D | 0.717 | 0.7174 | yes |  |
| F6 framing roleplay n | 23 | 23 | yes |  |
| F6 framing direct mean D | 0.707 | 0.7072 | yes |  |
| F6 framing direct n | 27 | 27 | yes |  |
| F6 framing obfuscated mean D | 0.698 | 0.6984 | yes |  |
| F6 framing obfuscated n | 23 | 23 | yes |  |
| T3 comply n | 59 | 59 | yes |  |
| T3 comply mean D | 0.896 | 0.896 | yes |  |
| T3 comply SD | 0.001 | 0.0006 | yes |  |
| T3 refuse n | 16 | 16 | yes |  |
| T3 refuse mean D | 0.249 | 0.249 | yes |  |
| T3 refuse SD | 0.004 | 0.0037 | yes |  |
| T3 refuse labels that are placeholders | 0 | 16 | NO | expected to fail (R1) |
| F2 Gemma 4 refusals with template | 65 | 65 | yes |  |
| F2 Gemma 4 refusals that are placeholders | 0 | 65 | NO | expected to fail (R1) |
| F4 llama-3.2-1b-it-cross-arch benign refuse % | 30 | 30.4348 | yes |  |
| F4 llama-3.2-1b-it-cross-arch hazard refuse % | 91 | 90.9091 | yes |  |
| F4 llama-3.2-1b-it-cross-arch benign refuse placeholders | 0 | 7 | NO | expected to fail (R1) |
| F4 qwen25-1b5-it-cross-arch benign refuse % | 83 | 82.6087 | yes |  |
| F4 qwen25-1b5-it-cross-arch hazard refuse % | 95 | 95.4545 | yes |  |
| F4 qwen25-1b5-it-cross-arch benign refuse placeholders | 0 | 19 | NO | expected to fail (R1) |
| F4 phi3-mini-4k-it-cross-arch benign refuse % | 87 | 86.9565 | yes |  |
| F4 phi3-mini-4k-it-cross-arch hazard refuse % | 95 | 95.4545 | yes |  |
| F4 phi3-mini-4k-it-cross-arch benign refuse placeholders | 0 | 20 | NO | expected to fail (R1) |

## Calibration matrix T that reproduces stored D

| run | T | max abs error |
|---|---|---|
| gemma-2-2b-it-L12-tuned | calibration_gemma2_2b:T | 2.22e-16 |
| gemma-2-2b-it-L12-tuned-rejudged | calibration_gemma2_2b:T | 5.61e-02 |
| gemma-2-2b-it-150tok-rejudged | calibration_gemma2_2b:T | 6.21e-02 |
| gemma-2-2b-it-explicit-tier3-qwen3-100 | calibration_gemma2_2b:T | 2.22e-16 |
| gemma-4-E2B-it-our-sae-v1-150tok | calibration_gemma2_2b:T_prior | 1.11e-16 |
| control-legal-financial-gemma2 | calibration_gemma2_2b:T | 2.22e-16 |
