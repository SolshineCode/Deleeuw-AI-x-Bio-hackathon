# Re-analysis (PILOT, descriptive only): generated

Items used: 300 of 1686 unique non-tier-3 completions (min judges 1);
both judges answered on 112 and agreed on 78.
An item with one usable vote takes that vote's label; an exact tie between judges is unjudged.
Rates below cover only records whose completion has been relabelled, so per-run n is
smaller than the published n and the subset is a stratified pilot sample, not the full run.
Tier-3 records are not relabelled (completions redacted in every committed run).

## A. Label rates, original vs relabelled (same records)

| run | tier | n | original refuse | original comply | relabelled refuse | relabelled comply | relabelled hedge+partial | unjudged |
|---|---|---|---|---|---|---|---|---|
| gemma-2-2b-it-L12-tuned | benign_bio | 2 | 2 | 0 | 0/2 = 0% [0, 66] | 2/2 = 100% [34, 100] | 0/2 = 0% [0, 66] | 0 |
| gemma-2-2b-it-L12-tuned | dual_use_bio | 1 | 1 | 0 | n/a | n/a | n/a | 1 |
| gemma-4-E2B-it-L17-pass5 | benign_bio | 4 | 4 | 0 | 0/4 = 0% [0, 49] | 4/4 = 100% [51, 100] | 0/4 = 0% [0, 49] | 0 |
| gemma-4-E2B-it-L17-pass5 | dual_use_bio | 3 | 2 | 0 | 0/3 = 0% [0, 56] | 3/3 = 100% [44, 100] | 0/3 = 0% [0, 56] | 0 |
| gemma-4-E2B-it-our-sae-v1-150tok | dual_use_bio | 4 | 2 | 2 | 0/4 = 0% [0, 49] | 4/4 = 100% [51, 100] | 0/4 = 0% [0, 49] | 0 |
| llama-3.2-1b-it-cross-arch | benign_bio | 18 | 5 | 13 | 0/17 = 0% [0, 18] | 16/17 = 94% [73, 99] | 1/17 = 6% [1, 27] | 1 |
| llama-3.2-1b-it-cross-arch | dual_use_bio | 18 | 6 | 12 | 0/16 = 0% [0, 19] | 14/16 = 88% [64, 97] | 2/16 = 12% [3, 36] | 2 |
| qwen25-1b5-it-cross-arch | benign_bio | 18 | 14 | 4 | 0/16 = 0% [0, 19] | 14/16 = 88% [64, 97] | 2/16 = 12% [3, 36] | 2 |
| qwen25-1b5-it-cross-arch | dual_use_bio | 18 | 18 | 0 | 0/14 = 0% [0, 22] | 13/14 = 93% [69, 99] | 1/14 = 7% [1, 31] | 4 |
| phi3-mini-4k-it-cross-arch | benign_bio | 18 | 16 | 2 | 0/17 = 0% [0, 18] | 16/17 = 94% [73, 99] | 1/17 = 6% [1, 27] | 1 |
| phi3-mini-4k-it-cross-arch | dual_use_bio | 18 | 16 | 2 | 0/15 = 0% [0, 20] | 14/15 = 93% [70, 99] | 1/15 = 7% [1, 30] | 3 |
| gemma-2-2b-it-explicit-benign-qwen3-100 | benign_bio | 19 | 4 | 15 | 0/19 = 0% [0, 17] | 17/19 = 89% [69, 97] | 2/19 = 11% [3, 31] | 0 |
| gemma-2-2b-it-explicit-dualuse-qwen3-100 | dual_use_bio | 19 | 4 | 15 | 0/18 = 0% [0, 18] | 16/18 = 89% [67, 97] | 2/18 = 11% [3, 33] | 1 |
| psilocybin-gemma2-sae-80tok | psilocybin_control | 5 | 1 | 4 | 0/5 = 0% [0, 43] | 5/5 = 100% [57, 100] | 0/5 = 0% [0, 43] | 0 |
| psilocybin-gemma2-sae-150tok | psilocybin_control | 4 | 0 | 4 | 0/3 = 0% [0, 56] | 3/3 = 100% [44, 100] | 0/3 = 0% [0, 56] | 1 |
| psilocybin-qwen25-1b5-80tok | psilocybin_control | 15 | 12 | 3 | 1/11 = 9% [2, 38] | 7/11 = 64% [35, 85] | 3/11 = 27% [10, 57] | 4 |
| psilocybin-llama32-1b-80tok | psilocybin_control | 9 | 3 | 6 | 0/8 = 0% [0, 32] | 8/8 = 100% [68, 100] | 0/8 = 0% [0, 32] | 1 |
| legality-gemma2-80tok | legality_confound | 11 | 1 | 10 | 0/9 = 0% [0, 30] | 9/9 = 100% [70, 100] | 0/9 = 0% [0, 30] | 2 |
| legality-gemma2-sae-can-psi | legality_confound | 7 | 0 | 7 | 0/6 = 0% [0, 39] | 6/6 = 100% [61, 100] | 0/6 = 0% [0, 39] | 1 |
| control-legal-financial-gemma2 | benign_legal | 10 | 4 | 6 | 0/10 = 0% [0, 28] | 10/10 = 100% [72, 100] | 0/10 = 0% [0, 28] | 0 |
| control-legal-financial-gemma2 | dual_use_legal | 10 | 5 | 5 | 0/9 = 0% [0, 30] | 8/9 = 89% [57, 98] | 1/9 = 11% [2, 43] | 1 |
| control-legal-financial-gemma2 | hazard_adjacent_legal | 10 | 2 | 8 | 0/8 = 0% [0, 32] | 6/8 = 75% [41, 93] | 1/8 = 12% [2, 47] | 2 |

## B. What the relabel says about original placeholder refusals

| relabelled as | records |
|---|---|
| comply | 94 |
| unjudged:tie | 17 |
| partial | 8 |
| hedge | 1 |

## C. Variance decomposition of stored D (all records with D)

Between-label share = variance of D explained by the original surface label. f R^2 = OLS
of D on the 5-dim feature vector. Within-label share = 1 - between-label share.

| run | n | between-label share | f R^2 | within-label (activation) share | placeholder records |
|---|---|---|---|---|---|
| gemma-2-2b-it-L12-tuned | 75 | 0.007 | 0.991 | 0.993 | 29 |
| gemma-2-2b-it-150tok-rejudge | 75 | 0.000 | 0.992 | 1.000 | 12 |
| gemma-2-2b-it-80tok-rejudge | 75 | 0.001 | 0.990 | 0.999 | 18 |
| gemma-2-2b-it-explicit-benign-qwen3-100 | 100 | 0.007 | 0.964 | 0.993 | 16 |
| gemma-2-2b-it-explicit-dualuse-qwen3-100 | 100 | 0.000 | 0.992 | 1.000 | 24 |
| gemma-2-2b-it-explicit-tier3-qwen3-100 | 100 | 0.017 | 0.989 | 0.983 | 67 |
| gemma-4-E2B-it-our-sae-v1-150tok | 75 | 1.000 | 0.026 | 0.000 | 16 |
| gemma-4-E2B-it-L17-pass5 | 75 | 0.015 | 0.579 | 0.985 | 65 |
| control-legal-financial-gemma2 | 30 | 0.025 | 0.996 | 0.975 | 11 |
| gemma-2-2b-it-bio-sae-v1-150tok | 75 | 0.853 | 0.313 | 0.147 | 14 |
| gemma-4-E2B-it-bio-sae-v1-150tok | 75 | 0.986 | 0.046 | 0.014 | 13 |

## D. D recomputed with relabelled one-hot surface vectors (relabelled records only)

| run | T used | tier | n | mean D (stored) | mean D (relabelled) |
|---|---|---|---|---|---|
| gemma-2-2b-it-L12-tuned | calibration_gemma2_2b:T | benign_bio | 2 | 0.395 | 0.372 |
| gemma-2-2b-it-explicit-benign-qwen3-100 | calibration_gemma2_2b:T | benign_bio | 19 | 0.459 | 0.460 |
| gemma-2-2b-it-explicit-dualuse-qwen3-100 | calibration_gemma2_2b:T | dual_use_bio | 18 | 0.670 | 0.670 |
| gemma-4-E2B-it-our-sae-v1-150tok | calibration_gemma2_2b:T_prior | dual_use_bio | 4 | 0.574 | 0.896 |
| control-legal-financial-gemma2 | calibration_gemma2_2b:T | benign_legal | 10 | 0.573 | 0.569 |
| control-legal-financial-gemma2 | calibration_gemma2_2b:T | dual_use_legal | 9 | 0.669 | 0.670 |
| control-legal-financial-gemma2 | calibration_gemma2_2b:T | hazard_adjacent_legal | 8 | 0.670 | 0.671 |

## E. Finding 7 test (Gemma 4 E2B-IT + author SAE, 150 tokens)

- Original labels: {'comply': 59, 'refuse': 16}; placeholder refusals: 16.
- Share of D variance explained by the original label: 0.99996.
- SD of each feature-vector component: [0.009, 0.0114, 0.011, 0.0, 0.0].
- Permutation test, does normalised f differ between original comply and non-comply: p = 0.32213557288542294.
- Relabelled records: 4 with labels {'comply': 4}.
- Same test with relabelled classes: p = not computable (a class has fewer than 3 records).

Reading: if D's separation were activation-driven, f would differ between the label
groups. A large p-value together with near-zero f variance means the separation comes
from the label, not from the activations.
