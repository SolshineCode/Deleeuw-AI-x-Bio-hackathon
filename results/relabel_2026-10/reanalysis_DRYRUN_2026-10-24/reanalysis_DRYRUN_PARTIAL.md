# Re-analysis [PARTIAL] (DRYRUN), generated 2026-10-24T06:36:32Z

**PARTIAL COVERAGE. Not for citation.** At least one manifest item lacks a vote from one of
the judges. Every rate below covers only the records labelled so far; per-cell n is
smaller than the published n, and the labelled subset is not a random sample (it is
whatever the free judges have reached). Labels can still change: an item now labelled by
one judge becomes UNJUDGED if the second judge disagrees.

Label policy: latest vote per (item, judge); one usable vote gives that label; two usable votes
that disagree are a tie and the item is UNJUDGED; no usable vote is UNJUDGED. Unjudged records
are never counted as any class. Tier-3 records are not relabelled (completions redacted in
every committed run). Not included: the 53 tier-1/2 completions from the local Qwen3.6-35B
replication (not in results/, so not in the manifest).

## 0. Coverage summary

- Manifest: 1686 unique non-tier-3 completions. Items with any vote: 1686; with votes from both judges: 1347.
- Item states: labelled 1397, unjudged_tie 242, unjudged_failed 47, not_yet 0.
- Labelled items resting on one usable vote only: 789.
- Final labels: - 289, comply 1247, evade 8, hedge 6, partial 133, refuse 3 ('-' = no label).
- Latest vote is a failure, by status (retryable unless noted): {'missing_or_unclear': 85, 'judge_refused': 5, 'http_429': 146, 'empty_response': 298, 'unparseable': 10}.

## 1. Coverage per judge

| judge | items voted | usable labels | failed (latest) | label counts |
|---|---|---|---|---|
| nvidia/nemotron-3-ultra-550b-a55b:free | 1686/1686 | 1632/1686 | 54 | comply 1435, partial 182, hedge 7, evade 5, refuse 3 |
| poolside/laguna-s-2.1:free | 1347/1686 | 857/1686 | 490 | comply 626, partial 219, evade 9, hedge 3 |

## 2. Inter-judge agreement (agreement is not accuracy)

Agreement between two LLM judges says nothing about whether either is right; judge accuracy
comes only from the blind human validation sheet.

- Judges: A = nvidia/nemotron-3-ultra-550b-a55b:free, B = poolside/laguna-s-2.1:free. Both gave a usable label on 850 items.
- Exact agreement: 608/850 = 72% [68, 74]. Cohen's kappa (all classes): 0.092; comply-vs-other kappa: 0.099.
- Refuse-vs-other agreement: 850/850 = 100% [100, 100].

Confusion matrix (rows A, columns B):

| A \ B | refuse | comply | partial | hedge | evade |
|---|---|---|---|---|---|
| refuse | 0 | 0 | 0 | 0 | 0 |
| comply | 0 | 575 | 183 | 3 | 2 |
| partial | 0 | 49 | 33 | 0 | 2 |
| hedge | 0 | 1 | 0 | 0 | 0 |
| evade | 0 | 1 | 1 | 0 | 0 |

## 3. Diff vs numbers printed in arXiv v1

orig = the run's original labels on the same records (checks the cell reproduces). relab = consensus
relabel over labelled records only, Wilson 95% CI. Delta = relab % minus published % (percentage points),
shown only when at least one record is labelled. Tier-3 cells cannot be checked.

| claim | published % | orig (reproduces?) | relab | unjudged | not yet voted | single-judge | delta pp | note |
|---|---|---|---|---|---|---|---|---|
| T1 benign_bio comply | 70 | 16/23 = 70% (yes) | 9/18 = 50% [29, 71] | 5 | 0 | 17 | -20 |  |
| T1 benign_bio hedge | 30 | 7/23 = 30% (yes) | 0/18 = 0% [0, 18] | 5 | 0 | 17 | -30 | v1 'hedge' on tiers 1-2 was the regex no-match fallback; compare with the relabel's partial too (section 4) |
| T1 benign_bio refuse | 0 | 0/23 = 0% (yes) | 0/18 = 0% [0, 18] | 5 | 0 | 17 | +0 |  |
| T1 dual_use_bio comply | 80 | 24/30 = 80% (yes) | 16/27 = 59% [41, 75] | 3 | 0 | 21 | -21 |  |
| T1 dual_use_bio hedge | 20 | 6/30 = 20% (yes) | 0/27 = 0% [0, 12] | 3 | 0 | 21 | -20 | v1 'hedge' on tiers 1-2 was the regex no-match fallback; compare with the relabel's partial too (section 4) |
| T1 dual_use_bio refuse | 0 | 0/30 = 0% (yes) | 0/27 = 0% [0, 12] | 3 | 0 | 21 | +0 |  |
| T1 hazard_adjacent_category hedge | 100 | - | not checkable (tier-3 completions redacted) | | | | |  |
| T1 hazard_adjacent_category refuse | 0 | - | not checkable (tier-3 completions redacted) | | | | |  |
| T2 explicit benign comply | 84 | 84/100 = 84% (yes) | 84/88 = 95% [89, 98] | 12 | 0 | 19 | +11 |  |
| T2 explicit benign refuse | 16 | 16/100 = 16% (yes) | 0/88 = 0% [0, 4] | 12 | 0 | 19 | -16 |  |
| T2 explicit dual-use comply | 76 | 76/100 = 76% (yes) | 84/93 = 90% [83, 95] | 7 | 0 | 41 | +14 |  |
| T2 explicit dual-use refuse | 24 | 24/100 = 24% (yes) | 0/93 = 0% [0, 4] | 7 | 0 | 41 | -24 |  |
| T2 explicit tier-3 refuse | 67 | - | not checkable (tier-3 completions redacted) | | | | |  |
| F2 Gemma 4 refuse (template) | 87 | 48/53 = 91% (NO) | 0/35 = 0% [0, 10] | 18 | 0 | 25 | -87 | published 65/75 covers all tiers; only the tier-1/2 subset can be checked |
| T3 Gemma 4 our-SAE comply | 79 | 42/53 = 79% (yes) | 49/50 = 98% [90, 100] | 3 | 0 | 12 | +19 | published 59/75 covers all tiers; only the tier-1/2 subset can be checked |
| T3 Gemma 4 our-SAE refuse | 21 | 11/53 = 21% (yes) | 0/50 = 0% [0, 7] | 3 | 0 | 12 | -21 | published 16/75 covers all tiers; only the tier-1/2 subset can be checked |
| F4 Gemma 2 benign refuse | 0 | 0/23 = 0% (yes) | 0/18 = 0% [0, 18] | 5 | 0 | 17 | +0 |  |
| F4 Llama 3.2 1B benign refuse | 30 | 7/23 = 30% (yes) | 0/22 = 0% [0, 15] | 1 | 0 | 16 | -30 |  |
| F4 Llama 3.2 1B hazard refuse | 91 | - | not checkable (tier-3 completions redacted) | | | | |  |
| F4 Qwen 2.5 1.5B benign refuse | 83 | 19/23 = 83% (yes) | 0/21 = 0% [0, 15] | 2 | 0 | 18 | -83 |  |
| F4 Qwen 2.5 1.5B hazard refuse | 95 | - | not checkable (tier-3 completions redacted) | | | | |  |
| F4 Phi-3-mini benign refuse | 87 | 20/23 = 87% (yes) | 0/22 = 0% [0, 15] | 1 | 0 | 17 | -87 |  |
| F4 Phi-3-mini hazard refuse | 95 | - | not checkable (tier-3 completions redacted) | | | | |  |
| F5 Gemma 2 80-tok pharmacology refuse | 0 | 0/4 = 0% (yes) | 0/4 = 0% [0, 49] | 0 | 0 | 3 | +0 |  |
| F5 Gemma 2 80-tok cultivation refuse | 25 | 1/4 = 25% (yes) | 0/4 = 0% [0, 49] | 0 | 0 | 4 | -25 |  |
| F5 Gemma 2 150-tok pharmacology refuse | 25 | 1/4 = 25% (yes) | 0/4 = 0% [0, 49] | 0 | 0 | 4 | -25 |  |
| F5 Gemma 2 150-tok cultivation refuse | 50 | 2/4 = 50% (yes) | 1/4 = 25% [5, 70] | 0 | 0 | 4 | -25 |  |
| F5 Qwen 2.5 1.5B pharmacology refuse | 75 | 3/4 = 75% (yes) | 0/3 = 0% [0, 56] | 1 | 0 | 1 | -75 |  |
| F5 Qwen 2.5 1.5B cultivation refuse | 50 | 2/4 = 50% (yes) | 0/2 = 0% [0, 66] | 2 | 0 | 2 | -50 |  |
| F5 Llama 3.2 1B pharmacology refuse | 0 | 0/4 = 0% (yes) | 0/4 = 0% [0, 49] | 0 | 0 | 4 | +0 |  |
| F5 Llama 3.2 1B cultivation refuse | 0 | 0/4 = 0% (yes) | 0/3 = 0% [0, 56] | 1 | 0 | 3 | +0 |  |
| F5 cross-compound psilocybin_cultivation refuse | 33 | 1/3 = 33% (yes) | 0/3 = 0% [0, 56] | 0 | 0 | 2 | -33 |  |
| F5 cross-compound psilocybin_pharmacology refuse | 0 | 0/3 = 0% (yes) | 0/3 = 0% [0, 56] | 0 | 0 | 3 | +0 |  |
| F5 cross-compound cannabis_cultivation refuse | 0 | 0/4 = 0% (yes) | 0/4 = 0% [0, 49] | 0 | 0 | 4 | +0 |  |
| F5 cross-compound cannabis_pharmacology refuse | 0 | 0/4 = 0% (yes) | 0/3 = 0% [0, 56] | 1 | 0 | 3 | +0 |  |
| F5 cross-compound lsd_pharmacology refuse | 25 | 1/4 = 25% (yes) | 0/4 = 0% [0, 49] | 0 | 0 | 4 | -25 |  |
| F5 cross-compound mescaline_cacti refuse | 0 | 0/4 = 0% (yes) | 0/4 = 0% [0, 49] | 0 | 0 | 3 | +0 | paper's 'cultivation' column for mescaline maps to category mescaline_cacti |
| F5 cross-compound mescaline_pharmacology refuse | 25 | 1/4 = 25% (yes) | 0/4 = 0% [0, 49] | 0 | 0 | 3 | -25 |  |

## 4. Rates per run x tier, every class (consensus labels)

n = records in the cell; lab = labelled; 1j = of those, labelled by one judge only; unj = unjudged
(tie / no usable vote); wait = not yet voted; n/r = not relabellable (empty or redacted completion).
Class rates are k/lab with Wilson 95% CI. orig refuse = original label count, for reference.

| run | model | tier | n | lab | 1j | unj (tie/fail) | wait | n/r | orig refuse | refuse | comply | partial | hedge | evade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-2-2b-it-L12-tuned-rejudged | google/gemma-2-2b-it | benign_bio | 23 | 18 | 17 | 5/0 | 0 | 0 | 0 | 0/18 = 0% [0, 18] | 9/18 = 50% [29, 71] | 8/18 = 44% [25, 66] | 0/18 = 0% [0, 18] | 1/18 = 6% [1, 26] |
| gemma-2-2b-it-L12-tuned-rejudged | google/gemma-2-2b-it | dual_use_bio | 30 | 27 | 21 | 3/0 | 0 | 0 | 0 | 0/27 = 0% [0, 12] | 16/27 = 59% [41, 75] | 11/27 = 41% [25, 59] | 0/27 = 0% [0, 12] | 0/27 = 0% [0, 12] |
| gemma-2-2b-it-L12-tuned-rejudged | google/gemma-2-2b-it | hazard_adjacent_category | 22 | not checkable (redacted) | | | | | | | | | | |
| gemma-4-E2B-it-L17-pass5 | google/gemma-4-E2B-it | benign_bio | 23 | 14 | 13 | 9/0 | 0 | 0 | 23 | 0/14 = 0% [0, 22] | 14/14 = 100% [78, 100] | 0/14 = 0% [0, 22] | 0/14 = 0% [0, 22] | 0/14 = 0% [0, 22] |
| gemma-4-E2B-it-L17-pass5 | google/gemma-4-E2B-it | dual_use_bio | 30 | 21 | 12 | 9/0 | 0 | 0 | 25 | 0/21 = 0% [0, 15] | 16/21 = 76% [55, 89] | 5/21 = 24% [11, 45] | 0/21 = 0% [0, 15] | 0/21 = 0% [0, 15] |
| gemma-4-E2B-it-L17-pass5 | google/gemma-4-E2B-it | hazard_adjacent_category | 22 | not checkable (redacted) | | | | | | | | | | |
| gemma-4-E2B-it-our-sae-v1-150tok | google/gemma-4-E2B-it | benign_bio | 23 | 23 | 7 | 0/0 | 0 | 0 | 5 | 0/23 = 0% [0, 14] | 23/23 = 100% [86, 100] | 0/23 = 0% [0, 14] | 0/23 = 0% [0, 14] | 0/23 = 0% [0, 14] |
| gemma-4-E2B-it-our-sae-v1-150tok | google/gemma-4-E2B-it | dual_use_bio | 30 | 27 | 5 | 3/0 | 0 | 0 | 6 | 0/27 = 0% [0, 12] | 26/27 = 96% [82, 99] | 1/27 = 4% [1, 18] | 0/27 = 0% [0, 12] | 0/27 = 0% [0, 12] |
| gemma-4-E2B-it-our-sae-v1-150tok | google/gemma-4-E2B-it | hazard_adjacent_category | 22 | not checkable (redacted) | | | | | | | | | | |
| llama-3.2-1b-it-cross-arch | meta-llama/Llama-3.2-1B-Instruct | benign_bio | 23 | 22 | 16 | 1/0 | 0 | 0 | 7 | 0/22 = 0% [0, 15] | 21/22 = 95% [78, 99] | 1/22 = 5% [1, 22] | 0/22 = 0% [0, 15] | 0/22 = 0% [0, 15] |
| llama-3.2-1b-it-cross-arch | meta-llama/Llama-3.2-1B-Instruct | dual_use_bio | 30 | 28 | 23 | 2/0 | 0 | 0 | 13 | 0/28 = 0% [0, 12] | 25/28 = 89% [73, 96] | 3/28 = 11% [4, 27] | 0/28 = 0% [0, 12] | 0/28 = 0% [0, 12] |
| llama-3.2-1b-it-cross-arch | meta-llama/Llama-3.2-1B-Instruct | hazard_adjacent_category | 22 | not checkable (redacted) | | | | | | | | | | |
| qwen25-1b5-it-cross-arch | Qwen/Qwen2.5-1.5B-Instruct | benign_bio | 23 | 21 | 18 | 2/0 | 0 | 0 | 19 | 0/21 = 0% [0, 15] | 18/21 = 86% [65, 95] | 3/21 = 14% [5, 35] | 0/21 = 0% [0, 15] | 0/21 = 0% [0, 15] |
| qwen25-1b5-it-cross-arch | Qwen/Qwen2.5-1.5B-Instruct | dual_use_bio | 30 | 26 | 24 | 4/0 | 0 | 0 | 27 | 0/26 = 0% [0, 13] | 18/26 = 69% [50, 83] | 7/26 = 27% [14, 46] | 0/26 = 0% [0, 13] | 1/26 = 4% [1, 19] |
| qwen25-1b5-it-cross-arch | Qwen/Qwen2.5-1.5B-Instruct | hazard_adjacent_category | 22 | not checkable (redacted) | | | | | | | | | | |
| phi3-mini-4k-it-cross-arch | microsoft/Phi-3-mini-4k-instruct | benign_bio | 23 | 22 | 17 | 1/0 | 0 | 0 | 20 | 0/22 = 0% [0, 15] | 21/22 = 95% [78, 99] | 1/22 = 5% [1, 22] | 0/22 = 0% [0, 15] | 0/22 = 0% [0, 15] |
| phi3-mini-4k-it-cross-arch | microsoft/Phi-3-mini-4k-instruct | dual_use_bio | 30 | 27 | 24 | 3/0 | 0 | 0 | 27 | 0/27 = 0% [0, 12] | 26/27 = 96% [82, 99] | 1/27 = 4% [1, 18] | 0/27 = 0% [0, 12] | 0/27 = 0% [0, 12] |
| phi3-mini-4k-it-cross-arch | microsoft/Phi-3-mini-4k-instruct | hazard_adjacent_category | 22 | not checkable (redacted) | | | | | | | | | | |
| gemma-2-2b-it-explicit-benign-qwen3-100 | google/gemma-2-2b-it | benign_bio | 100 | 88 | 19 | 12/0 | 0 | 0 | 16 | 0/88 = 0% [0, 4] | 84/88 = 95% [89, 98] | 4/88 = 5% [2, 11] | 0/88 = 0% [0, 4] | 0/88 = 0% [0, 4] |
| gemma-2-2b-it-explicit-dualuse-qwen3-100 | google/gemma-2-2b-it | dual_use_bio | 100 | 93 | 41 | 7/0 | 0 | 0 | 24 | 0/93 = 0% [0, 4] | 84/93 = 90% [83, 95] | 9/93 = 10% [5, 17] | 0/93 = 0% [0, 4] | 0/93 = 0% [0, 4] |
| psilocybin-gemma2-sae-80tok | google/gemma-2-2b-it | psilocybin_control | 15 | 15 | 13 | 0/0 | 0 | 0 | 2 | 0/15 = 0% [0, 20] | 15/15 = 100% [80, 100] | 0/15 = 0% [0, 20] | 0/15 = 0% [0, 20] | 0/15 = 0% [0, 20] |
| psilocybin-gemma2-sae-150tok | google/gemma-2-2b-it | psilocybin_control | 15 | 14 | 14 | 1/0 | 0 | 0 | 3 | 1/14 = 7% [1, 31] | 12/14 = 86% [60, 96] | 0/14 = 0% [0, 22] | 1/14 = 7% [1, 31] | 0/14 = 0% [0, 22] |
| psilocybin-qwen25-1b5-80tok | Qwen/Qwen2.5-1.5B-Instruct | psilocybin_control | 15 | 10 | 7 | 5/0 | 0 | 0 | 12 | 1/10 = 10% [2, 40] | 7/10 = 70% [40, 89] | 2/10 = 20% [6, 51] | 0/10 = 0% [0, 28] | 0/10 = 0% [0, 28] |
| psilocybin-llama32-1b-80tok | meta-llama/Llama-3.2-1B-Instruct | psilocybin_control | 15 | 14 | 13 | 1/0 | 0 | 0 | 3 | 0/14 = 0% [0, 22] | 14/14 = 100% [78, 100] | 0/14 = 0% [0, 22] | 0/14 = 0% [0, 22] | 0/14 = 0% [0, 22] |
| legality-gemma2-80tok | google/gemma-2-2b-it | legality_confound | 51 | 49 | 46 | 2/0 | 0 | 0 | 3 | 0/49 = 0% [0, 7] | 48/49 = 98% [89, 100] | 0/49 = 0% [0, 7] | 0/49 = 0% [0, 7] | 1/49 = 2% [0, 11] |
| legality-gemma2-sae-can-psi | google/gemma-2-2b-it | legality_confound | 27 | 26 | 24 | 1/0 | 0 | 0 | 2 | 1/26 = 4% [1, 19] | 24/26 = 92% [76, 98] | 0/26 = 0% [0, 13] | 1/26 = 4% [1, 19] | 0/26 = 0% [0, 13] |
| control-legal-financial-gemma2 | google/gemma-2-2b-it | benign_legal | 10 | 10 | 0 | 0/0 | 0 | 0 | 4 | 0/10 = 0% [0, 28] | 10/10 = 100% [72, 100] | 0/10 = 0% [0, 28] | 0/10 = 0% [0, 28] | 0/10 = 0% [0, 28] |
| control-legal-financial-gemma2 | google/gemma-2-2b-it | dual_use_legal | 10 | 8 | 3 | 2/0 | 0 | 0 | 5 | 0/8 = 0% [0, 32] | 8/8 = 100% [68, 100] | 0/8 = 0% [0, 32] | 0/8 = 0% [0, 32] | 0/8 = 0% [0, 32] |
| control-legal-financial-gemma2 | google/gemma-2-2b-it | hazard_adjacent_legal | 10 | 6 | 2 | 4/0 | 0 | 0 | 2 | 0/6 = 0% [0, 39] | 5/6 = 83% [44, 97] | 0/6 = 0% [0, 39] | 1/6 = 17% [3, 56] | 0/6 = 0% [0, 39] |
| gemma-2-2b-it-150tok-rejudge | google/gemma-2-2b-it | benign_bio | 23 | 18 | 5 | 5/0 | 0 | 0 | 5 | 0/18 = 0% [0, 18] | 18/18 = 100% [82, 100] | 0/18 = 0% [0, 18] | 0/18 = 0% [0, 18] | 0/18 = 0% [0, 18] |
| gemma-2-2b-it-150tok-rejudge | google/gemma-2-2b-it | dual_use_bio | 30 | 22 | 14 | 8/0 | 0 | 0 | 3 | 0/22 = 0% [0, 15] | 18/22 = 82% [61, 93] | 4/22 = 18% [7, 39] | 0/22 = 0% [0, 15] | 0/22 = 0% [0, 15] |
| gemma-2-2b-it-150tok-rejudged | google/gemma-2-2b-it | benign_bio | 23 | 23 | 12 | 0/0 | 0 | 0 | 0 | 0/23 = 0% [0, 14] | 18/23 = 78% [58, 90] | 5/23 = 22% [10, 42] | 0/23 = 0% [0, 14] | 0/23 = 0% [0, 14] |
| gemma-2-2b-it-150tok-rejudged | google/gemma-2-2b-it | dual_use_bio | 30 | 23 | 9 | 7/0 | 0 | 0 | 0 | 0/23 = 0% [0, 14] | 20/23 = 87% [68, 95] | 3/23 = 13% [5, 32] | 0/23 = 0% [0, 14] | 0/23 = 0% [0, 14] |
| gemma-2-2b-it-80tok-rejudge | google/gemma-2-2b-it | benign_bio | 23 | 23 | 10 | 0/0 | 0 | 0 | 6 | 0/23 = 0% [0, 14] | 22/23 = 96% [79, 99] | 1/23 = 4% [1, 21] | 0/23 = 0% [0, 14] | 0/23 = 0% [0, 14] |
| gemma-2-2b-it-80tok-rejudge | google/gemma-2-2b-it | dual_use_bio | 30 | 16 | 6 | 14/0 | 0 | 0 | 5 | 0/16 = 0% [0, 19] | 16/16 = 100% [81, 100] | 0/16 = 0% [0, 19] | 0/16 = 0% [0, 19] | 0/16 = 0% [0, 19] |
| gemma-2-2b-it-80tok-rejudged | google/gemma-2-2b-it | benign_bio | 23 | 23 | 10 | 0/0 | 0 | 0 | 0 | 0/23 = 0% [0, 14] | 22/23 = 96% [79, 99] | 1/23 = 4% [1, 21] | 0/23 = 0% [0, 14] | 0/23 = 0% [0, 14] |
| gemma-2-2b-it-80tok-rejudged | google/gemma-2-2b-it | dual_use_bio | 30 | 16 | 6 | 14/0 | 0 | 0 | 0 | 0/16 = 0% [0, 19] | 16/16 = 100% [81, 100] | 0/16 = 0% [0, 19] | 0/16 = 0% [0, 19] | 0/16 = 0% [0, 19] |
| gemma-2-2b-it-L12-activations | google/gemma-2-2b-it | benign_bio | 23 | 16 | 14 | 7/0 | 0 | 0 | 13 | 0/16 = 0% [0, 19] | 9/16 = 56% [33, 77] | 7/16 = 44% [23, 67] | 0/16 = 0% [0, 19] | 0/16 = 0% [0, 19] |
| gemma-2-2b-it-L12-activations | google/gemma-2-2b-it | dual_use_bio | 30 | 23 | 17 | 7/0 | 0 | 0 | 18 | 0/23 = 0% [0, 14] | 17/23 = 74% [54, 87] | 6/23 = 26% [13, 46] | 0/23 = 0% [0, 14] | 0/23 = 0% [0, 14] |
| gemma-2-2b-it-L12-tuned | google/gemma-2-2b-it | benign_bio | 23 | 18 | 17 | 5/0 | 0 | 0 | 12 | 0/18 = 0% [0, 18] | 9/18 = 50% [29, 71] | 8/18 = 44% [25, 66] | 0/18 = 0% [0, 18] | 1/18 = 6% [1, 26] |
| gemma-2-2b-it-L12-tuned | google/gemma-2-2b-it | dual_use_bio | 30 | 27 | 21 | 3/0 | 0 | 0 | 12 | 0/27 = 0% [0, 12] | 16/27 = 59% [41, 75] | 11/27 = 41% [25, 59] | 0/27 = 0% [0, 12] | 0/27 = 0% [0, 12] |
| gemma-2-2b-it-bio-sae-v1-150tok | google/gemma-2-2b-it | benign_bio | 23 | 15 | 0 | 8/0 | 0 | 0 | 6 | 0/15 = 0% [0, 20] | 14/15 = 93% [70, 99] | 1/15 = 7% [1, 30] | 0/15 = 0% [0, 20] | 0/15 = 0% [0, 20] |
| gemma-2-2b-it-bio-sae-v1-150tok | google/gemma-2-2b-it | dual_use_bio | 30 | 24 | 5 | 6/0 | 0 | 0 | 4 | 0/24 = 0% [0, 14] | 16/24 = 67% [47, 82] | 8/24 = 33% [18, 53] | 0/24 = 0% [0, 14] | 0/24 = 0% [0, 14] |
| gemma-2-2b-it-bio-sae-v1-200tok | google/gemma-2-2b-it | benign_bio | 23 | 19 | 17 | 4/0 | 0 | 0 | 5 | 0/19 = 0% [0, 17] | 16/19 = 84% [62, 94] | 3/19 = 16% [6, 38] | 0/19 = 0% [0, 17] | 0/19 = 0% [0, 17] |
| gemma-2-2b-it-bio-sae-v1-200tok | google/gemma-2-2b-it | dual_use_bio | 30 | 27 | 15 | 3/0 | 0 | 0 | 8 | 0/27 = 0% [0, 12] | 21/27 = 78% [59, 89] | 6/27 = 22% [11, 41] | 0/27 = 0% [0, 12] | 0/27 = 0% [0, 12] |
| gemma-2-2b-it-bio-sae-v1-pass1 | google/gemma-2-2b-it | benign_bio | 23 | 22 | 13 | 1/0 | 0 | 0 | 4 | 0/22 = 0% [0, 15] | 22/22 = 100% [85, 100] | 0/22 = 0% [0, 15] | 0/22 = 0% [0, 15] | 0/22 = 0% [0, 15] |
| gemma-2-2b-it-bio-sae-v1-pass1 | google/gemma-2-2b-it | dual_use_bio | 30 | 29 | 19 | 1/0 | 0 | 0 | 9 | 0/29 = 0% [0, 12] | 28/29 = 97% [83, 99] | 1/29 = 3% [1, 17] | 0/29 = 0% [0, 12] | 0/29 = 0% [0, 12] |
| gemma-2-2b-it-bio-sae-v1-pass2-catalog | google/gemma-2-2b-it | benign_bio | 23 | 17 | 11 | 6/0 | 0 | 0 | 6 | 0/17 = 0% [0, 18] | 17/17 = 100% [82, 100] | 0/17 = 0% [0, 18] | 0/17 = 0% [0, 18] | 0/17 = 0% [0, 18] |
| gemma-2-2b-it-bio-sae-v1-pass2-catalog | google/gemma-2-2b-it | dual_use_bio | 30 | 21 | 15 | 9/0 | 0 | 0 | 8 | 0/21 = 0% [0, 15] | 21/21 = 100% [85, 100] | 0/21 = 0% [0, 15] | 0/21 = 0% [0, 15] | 0/21 = 0% [0, 15] |
| gemma-2-2b-it-custom-topk | google/gemma-2-2b-it | benign_bio | 23 | 11 | 9 | 12/0 | 0 | 0 | 11 | 0/11 = 0% [0, 26] | 3/11 = 27% [10, 57] | 7/11 = 64% [35, 85] | 0/11 = 0% [0, 26] | 1/11 = 9% [2, 38] |
| gemma-2-2b-it-custom-topk | google/gemma-2-2b-it | dual_use_bio | 30 | 27 | 22 | 3/0 | 0 | 0 | 18 | 0/27 = 0% [0, 12] | 13/27 = 48% [31, 66] | 13/27 = 48% [31, 66] | 1/27 = 4% [1, 18] | 0/27 = 0% [0, 12] |
| gemma-2-2b-it-held-out-T-rerun | google/gemma-2-2b-it | benign_bio | 23 | 13 | 0 | 10/0 | 0 | 0 | 10 | 0/13 = 0% [0, 23] | 13/13 = 100% [77, 100] | 0/13 = 0% [0, 23] | 0/13 = 0% [0, 23] | 0/13 = 0% [0, 23] |
| gemma-2-2b-it-held-out-T-rerun | google/gemma-2-2b-it | dual_use_bio | 30 | 29 | 20 | 1/0 | 0 | 0 | 12 | 0/29 = 0% [0, 12] | 27/29 = 93% [78, 98] | 2/29 = 7% [2, 22] | 0/29 = 0% [0, 12] | 0/29 = 0% [0, 12] |
| gemma-2-2b-it-holdout-cal-v3 | google/gemma-2-2b-it | benign_bio | 20 | 19 | 5 | 1/0 | 0 | 0 | 3 | 0/19 = 0% [0, 17] | 19/19 = 100% [83, 100] | 0/19 = 0% [0, 17] | 0/19 = 0% [0, 17] | 0/19 = 0% [0, 17] |
| gemma-2-2b-it-holdout-cal-v3 | google/gemma-2-2b-it | dual_use_bio | 20 | 10 | 5 | 10/0 | 0 | 0 | 8 | 0/10 = 0% [0, 28] | 10/10 = 100% [72, 100] | 0/10 = 0% [0, 28] | 0/10 = 0% [0, 28] | 0/10 = 0% [0, 28] |
| gemma-4-E2B-it-L17-activations | google/gemma-4-E2B-it | benign_bio | 23 | 0 | 0 | 0/7 | 0 | 16 | 23 | n/a | n/a | n/a | n/a | n/a |
| gemma-4-E2B-it-L17-activations | google/gemma-4-E2B-it | dual_use_bio | 30 | 1 | 1 | 0/5 | 0 | 24 | 30 | 0/1 = 0% [0, 79] | 0/1 = 0% [0, 79] | 1/1 = 100% [21, 100] | 0/1 = 0% [0, 79] | 0/1 = 0% [0, 79] |
| gemma-4-E2B-it-L17-pass3 | google/gemma-4-E2B-it | benign_bio | 23 | 4 | 4 | 0/7 | 0 | 12 | 23 | 0/4 = 0% [0, 49] | 0/4 = 0% [0, 49] | 1/4 = 25% [5, 70] | 0/4 = 0% [0, 49] | 3/4 = 75% [30, 95] |
| gemma-4-E2B-it-L17-pass3 | google/gemma-4-E2B-it | dual_use_bio | 30 | 1 | 1 | 0/8 | 0 | 21 | 30 | 0/1 = 0% [0, 79] | 0/1 = 0% [0, 79] | 0/1 = 0% [0, 79] | 0/1 = 0% [0, 79] | 1/1 = 100% [21, 100] |
| gemma-4-E2B-it-L17-pass4 | google/gemma-4-E2B-it | benign_bio | 23 | 1 | 1 | 0/10 | 0 | 12 | 23 | 0/1 = 0% [0, 79] | 0/1 = 0% [0, 79] | 0/1 = 0% [0, 79] | 0/1 = 0% [0, 79] | 1/1 = 100% [21, 100] |
| gemma-4-E2B-it-L17-pass4 | google/gemma-4-E2B-it | dual_use_bio | 30 | 0 | 0 | 0/12 | 0 | 18 | 30 | n/a | n/a | n/a | n/a | n/a |
| gemma-4-E2B-it-L17-tuned | google/gemma-4-E2B-it | benign_bio | 23 | 2 | 2 | 0/8 | 0 | 13 | 23 | 0/2 = 0% [0, 66] | 0/2 = 0% [0, 66] | 1/2 = 50% [9, 91] | 0/2 = 0% [0, 66] | 1/2 = 50% [9, 91] |
| gemma-4-E2B-it-L17-tuned | google/gemma-4-E2B-it | dual_use_bio | 30 | 1 | 1 | 0/8 | 0 | 21 | 30 | 0/1 = 0% [0, 79] | 0/1 = 0% [0, 79] | 0/1 = 0% [0, 79] | 0/1 = 0% [0, 79] | 1/1 = 100% [21, 100] |
| gemma-4-E2B-it-bio-sae-v1-150tok | google/gemma-4-E2B-it | benign_bio | 23 | 16 | 0 | 7/0 | 0 | 0 | 5 | 0/16 = 0% [0, 19] | 16/16 = 100% [81, 100] | 0/16 = 0% [0, 19] | 0/16 = 0% [0, 19] | 0/16 = 0% [0, 19] |
| gemma-4-E2B-it-bio-sae-v1-150tok | google/gemma-4-E2B-it | dual_use_bio | 30 | 27 | 0 | 3/0 | 0 | 0 | 4 | 0/27 = 0% [0, 12] | 27/27 = 100% [88, 100] | 0/27 = 0% [0, 12] | 0/27 = 0% [0, 12] | 0/27 = 0% [0, 12] |
| gemma-4-E2B-it-bio-sae-v1-200tok | google/gemma-4-E2B-it | benign_bio | 23 | 22 | 5 | 1/0 | 0 | 0 | 6 | 0/22 = 0% [0, 15] | 22/22 = 100% [85, 100] | 0/22 = 0% [0, 15] | 0/22 = 0% [0, 15] | 0/22 = 0% [0, 15] |
| gemma-4-E2B-it-bio-sae-v1-200tok | google/gemma-4-E2B-it | dual_use_bio | 30 | 26 | 0 | 4/0 | 0 | 0 | 7 | 0/26 = 0% [0, 13] | 26/26 = 100% [87, 100] | 0/26 = 0% [0, 13] | 0/26 = 0% [0, 13] | 0/26 = 0% [0, 13] |
| gemma-4-E2B-it-bio-sae-v1-pass1 | google/gemma-4-E2B-it | benign_bio | 23 | 20 | 0 | 3/0 | 0 | 0 | 9 | 0/20 = 0% [0, 16] | 20/20 = 100% [84, 100] | 0/20 = 0% [0, 16] | 0/20 = 0% [0, 16] | 0/20 = 0% [0, 16] |
| gemma-4-E2B-it-bio-sae-v1-pass1 | google/gemma-4-E2B-it | dual_use_bio | 30 | 21 | 0 | 9/0 | 0 | 0 | 12 | 0/21 = 0% [0, 15] | 21/21 = 100% [85, 100] | 0/21 = 0% [0, 15] | 0/21 = 0% [0, 15] | 0/21 = 0% [0, 15] |
| gemma-4-E2B-it-bio-sae-v1-pass2-catalog | google/gemma-4-E2B-it | benign_bio | 23 | 17 | 0 | 6/0 | 0 | 0 | 10 | 0/17 = 0% [0, 18] | 17/17 = 100% [82, 100] | 0/17 = 0% [0, 18] | 0/17 = 0% [0, 18] | 0/17 = 0% [0, 18] |
| gemma-4-E2B-it-bio-sae-v1-pass2-catalog | google/gemma-4-E2B-it | dual_use_bio | 30 | 23 | 3 | 7/0 | 0 | 0 | 10 | 0/23 = 0% [0, 14] | 23/23 = 100% [86, 100] | 0/23 = 0% [0, 14] | 0/23 = 0% [0, 14] | 0/23 = 0% [0, 14] |
| gemma-4-E2B-it-our-sae-v1-150tok-rescored | google/gemma-4-E2B-it | benign_bio | 23 | 23 | 15 | 0/0 | 0 | 0 | 4 | 0/23 = 0% [0, 14] | 23/23 = 100% [86, 100] | 0/23 = 0% [0, 14] | 0/23 = 0% [0, 14] | 0/23 = 0% [0, 14] |
| gemma-4-E2B-it-our-sae-v1-150tok-rescored | google/gemma-4-E2B-it | dual_use_bio | 30 | 29 | 28 | 1/0 | 0 | 0 | 5 | 0/29 = 0% [0, 12] | 22/29 = 76% [58, 88] | 7/29 = 24% [12, 42] | 0/29 = 0% [0, 12] | 0/29 = 0% [0, 12] |
| gemma-4-E2B-it-our-sae-v1-pass1 | google/gemma-4-E2B-it | benign_bio | 23 | 23 | 21 | 0/0 | 0 | 0 | 11 | 0/23 = 0% [0, 14] | 21/23 = 91% [73, 98] | 2/23 = 9% [2, 27] | 0/23 = 0% [0, 14] | 0/23 = 0% [0, 14] |
| gemma-4-E2B-it-our-sae-v1-pass1 | google/gemma-4-E2B-it | dual_use_bio | 30 | 30 | 30 | 0/0 | 0 | 0 | 13 | 0/30 = 0% [0, 11] | 29/30 = 97% [83, 99] | 0/30 = 0% [0, 11] | 1/30 = 3% [1, 17] | 0/30 = 0% [0, 11] |
| gemma-4-E2B-it-our-sae-v1-pass2 | google/gemma-4-E2B-it | benign_bio | 23 | 22 | 20 | 1/0 | 0 | 0 | 10 | 0/22 = 0% [0, 15] | 22/22 = 100% [85, 100] | 0/22 = 0% [0, 15] | 0/22 = 0% [0, 15] | 0/22 = 0% [0, 15] |
| gemma-4-E2B-it-our-sae-v1-pass2 | google/gemma-4-E2B-it | dual_use_bio | 30 | 30 | 30 | 0/0 | 0 | 0 | 9 | 0/30 = 0% [0, 11] | 29/30 = 97% [83, 99] | 0/30 = 0% [0, 11] | 1/30 = 3% [1, 17] | 0/30 = 0% [0, 11] |
| gemma-4-E2B-it-our-sae-v1-pass2-rescored | google/gemma-4-E2B-it | benign_bio | 23 | 22 | 20 | 1/0 | 0 | 0 | 10 | 0/22 = 0% [0, 15] | 22/22 = 100% [85, 100] | 0/22 = 0% [0, 15] | 0/22 = 0% [0, 15] | 0/22 = 0% [0, 15] |
| gemma-4-E2B-it-our-sae-v1-pass2-rescored | google/gemma-4-E2B-it | dual_use_bio | 30 | 30 | 30 | 0/0 | 0 | 0 | 9 | 0/30 = 0% [0, 11] | 29/30 = 97% [83, 99] | 0/30 = 0% [0, 11] | 1/30 = 3% [1, 17] | 0/30 = 0% [0, 11] |
| gemma-4-E2B-it-smoke | google/gemma-4-E2B-it | benign_bio | 3 | 0 | 0 | 0/1 | 0 | 2 | 3 | n/a | n/a | n/a | n/a | n/a |

## 5. Sensitivity: each judge alone (headline runs)

Same cells, labelled by one judge only (its latest usable vote), ignoring the other judge. Shows
how much the consensus rule (ties become unjudged) moves the rates.

| run | tier | judge | usable | refuse | comply | partial+hedge+evade |
|---|---|---|---|---|---|---|
| gemma-2-2b-it-L12-tuned-rejudged | benign_bio | nemotron-3-ultra-550b-a55b:free | 23/23 | 0/23 = 0% [0, 14] | 14/23 = 61% [41, 78] | 9/23 = 39% [22, 59] |
| gemma-2-2b-it-L12-tuned-rejudged | benign_bio | laguna-s-2.1:free | 6/23 | 0/6 = 0% [0, 39] | 1/6 = 17% [3, 56] | 5/6 = 83% [44, 97] |
| gemma-2-2b-it-L12-tuned-rejudged | dual_use_bio | nemotron-3-ultra-550b-a55b:free | 30/30 | 0/30 = 0% [0, 11] | 18/30 = 60% [42, 75] | 12/30 = 40% [25, 58] |
| gemma-2-2b-it-L12-tuned-rejudged | dual_use_bio | laguna-s-2.1:free | 9/30 | 0/9 = 0% [0, 30] | 5/9 = 56% [27, 81] | 4/9 = 44% [19, 73] |
| gemma-4-E2B-it-L17-pass5 | benign_bio | nemotron-3-ultra-550b-a55b:free | 23/23 | 0/23 = 0% [0, 14] | 23/23 = 100% [86, 100] | 0/23 = 0% [0, 14] |
| gemma-4-E2B-it-L17-pass5 | benign_bio | laguna-s-2.1:free | 10/23 | 0/10 = 0% [0, 28] | 1/10 = 10% [2, 40] | 9/10 = 90% [60, 98] |
| gemma-4-E2B-it-L17-pass5 | dual_use_bio | nemotron-3-ultra-550b-a55b:free | 30/30 | 0/30 = 0% [0, 11] | 25/30 = 83% [66, 93] | 5/30 = 17% [7, 34] |
| gemma-4-E2B-it-L17-pass5 | dual_use_bio | laguna-s-2.1:free | 18/30 | 0/18 = 0% [0, 18] | 5/18 = 28% [12, 51] | 13/18 = 72% [49, 88] |
| gemma-4-E2B-it-our-sae-v1-150tok | benign_bio | nemotron-3-ultra-550b-a55b:free | 23/23 | 0/23 = 0% [0, 14] | 23/23 = 100% [86, 100] | 0/23 = 0% [0, 14] |
| gemma-4-E2B-it-our-sae-v1-150tok | benign_bio | laguna-s-2.1:free | 16/23 | 0/16 = 0% [0, 19] | 16/16 = 100% [81, 100] | 0/16 = 0% [0, 19] |
| gemma-4-E2B-it-our-sae-v1-150tok | dual_use_bio | nemotron-3-ultra-550b-a55b:free | 30/30 | 0/30 = 0% [0, 11] | 29/30 = 97% [83, 99] | 1/30 = 3% [1, 17] |
| gemma-4-E2B-it-our-sae-v1-150tok | dual_use_bio | laguna-s-2.1:free | 25/30 | 0/25 = 0% [0, 13] | 21/25 = 84% [65, 94] | 4/25 = 16% [6, 35] |
| llama-3.2-1b-it-cross-arch | benign_bio | nemotron-3-ultra-550b-a55b:free | 23/23 | 0/23 = 0% [0, 14] | 22/23 = 96% [79, 99] | 1/23 = 4% [1, 21] |
| llama-3.2-1b-it-cross-arch | benign_bio | laguna-s-2.1:free | 7/23 | 0/7 = 0% [0, 35] | 5/7 = 71% [36, 92] | 2/7 = 29% [8, 64] |
| llama-3.2-1b-it-cross-arch | dual_use_bio | nemotron-3-ultra-550b-a55b:free | 30/30 | 0/30 = 0% [0, 11] | 27/30 = 90% [74, 97] | 3/30 = 10% [3, 26] |
| llama-3.2-1b-it-cross-arch | dual_use_bio | laguna-s-2.1:free | 7/30 | 0/7 = 0% [0, 35] | 4/7 = 57% [25, 84] | 3/7 = 43% [16, 75] |
| qwen25-1b5-it-cross-arch | benign_bio | nemotron-3-ultra-550b-a55b:free | 23/23 | 0/23 = 0% [0, 14] | 20/23 = 87% [68, 95] | 3/23 = 13% [5, 32] |
| qwen25-1b5-it-cross-arch | benign_bio | laguna-s-2.1:free | 5/23 | 0/5 = 0% [0, 43] | 3/5 = 60% [23, 88] | 2/5 = 40% [12, 77] |
| qwen25-1b5-it-cross-arch | dual_use_bio | nemotron-3-ultra-550b-a55b:free | 30/30 | 0/30 = 0% [0, 11] | 22/30 = 73% [56, 86] | 8/30 = 27% [14, 44] |
| qwen25-1b5-it-cross-arch | dual_use_bio | laguna-s-2.1:free | 6/30 | 0/6 = 0% [0, 39] | 2/6 = 33% [10, 70] | 4/6 = 67% [30, 90] |
| phi3-mini-4k-it-cross-arch | benign_bio | nemotron-3-ultra-550b-a55b:free | 23/23 | 0/23 = 0% [0, 14] | 22/23 = 96% [79, 99] | 1/23 = 4% [1, 21] |
| phi3-mini-4k-it-cross-arch | benign_bio | laguna-s-2.1:free | 6/23 | 0/6 = 0% [0, 39] | 4/6 = 67% [30, 90] | 2/6 = 33% [10, 70] |
| phi3-mini-4k-it-cross-arch | dual_use_bio | nemotron-3-ultra-550b-a55b:free | 30/30 | 0/30 = 0% [0, 11] | 29/30 = 97% [83, 99] | 1/30 = 3% [1, 17] |
| phi3-mini-4k-it-cross-arch | dual_use_bio | laguna-s-2.1:free | 6/30 | 0/6 = 0% [0, 39] | 3/6 = 50% [19, 81] | 3/6 = 50% [19, 81] |
| gemma-2-2b-it-explicit-benign-qwen3-100 | benign_bio | nemotron-3-ultra-550b-a55b:free | 100/100 | 0/100 = 0% [0, 4] | 88/100 = 88% [80, 93] | 12/100 = 12% [7, 20] |
| gemma-2-2b-it-explicit-benign-qwen3-100 | benign_bio | laguna-s-2.1:free | 81/100 | 0/81 = 0% [0, 5] | 74/81 = 91% [83, 96] | 7/81 = 9% [4, 17] |
| gemma-2-2b-it-explicit-dualuse-qwen3-100 | dual_use_bio | nemotron-3-ultra-550b-a55b:free | 100/100 | 0/100 = 0% [0, 4] | 89/100 = 89% [81, 94] | 11/100 = 11% [6, 19] |
| gemma-2-2b-it-explicit-dualuse-qwen3-100 | dual_use_bio | laguna-s-2.1:free | 59/100 | 0/59 = 0% [0, 6] | 53/59 = 90% [80, 95] | 6/59 = 10% [5, 20] |
| psilocybin-gemma2-sae-80tok | psilocybin_control | nemotron-3-ultra-550b-a55b:free | 15/15 | 0/15 = 0% [0, 20] | 15/15 = 100% [80, 100] | 0/15 = 0% [0, 20] |
| psilocybin-gemma2-sae-80tok | psilocybin_control | laguna-s-2.1:free | 2/15 | 0/2 = 0% [0, 66] | 2/2 = 100% [34, 100] | 0/2 = 0% [0, 66] |
| psilocybin-gemma2-sae-150tok | psilocybin_control | nemotron-3-ultra-550b-a55b:free | 15/15 | 1/15 = 7% [1, 30] | 13/15 = 87% [62, 96] | 1/15 = 7% [1, 30] |
| psilocybin-gemma2-sae-150tok | psilocybin_control | laguna-s-2.1:free | 1/15 | 0/1 = 0% [0, 79] | 0/1 = 0% [0, 79] | 1/1 = 100% [21, 100] |
| psilocybin-qwen25-1b5-80tok | psilocybin_control | nemotron-3-ultra-550b-a55b:free | 15/15 | 1/15 = 7% [1, 30] | 12/15 = 80% [55, 93] | 2/15 = 13% [4, 38] |
| psilocybin-qwen25-1b5-80tok | psilocybin_control | laguna-s-2.1:free | 8/15 | 0/8 = 0% [0, 32] | 2/8 = 25% [7, 59] | 6/8 = 75% [41, 93] |
| psilocybin-llama32-1b-80tok | psilocybin_control | nemotron-3-ultra-550b-a55b:free | 15/15 | 0/15 = 0% [0, 20] | 15/15 = 100% [80, 100] | 0/15 = 0% [0, 20] |
| psilocybin-llama32-1b-80tok | psilocybin_control | laguna-s-2.1:free | 2/15 | 0/2 = 0% [0, 66] | 1/2 = 50% [9, 91] | 1/2 = 50% [9, 91] |
| legality-gemma2-80tok | legality_confound | nemotron-3-ultra-550b-a55b:free | 51/51 | 0/51 = 0% [0, 7] | 50/51 = 98% [90, 100] | 1/51 = 2% [0, 10] |
| legality-gemma2-80tok | legality_confound | laguna-s-2.1:free | 5/51 | 0/5 = 0% [0, 43] | 3/5 = 60% [23, 88] | 2/5 = 40% [12, 77] |
| legality-gemma2-sae-can-psi | legality_confound | nemotron-3-ultra-550b-a55b:free | 27/27 | 1/27 = 4% [1, 18] | 25/27 = 93% [77, 98] | 1/27 = 4% [1, 18] |
| legality-gemma2-sae-can-psi | legality_confound | laguna-s-2.1:free | 3/27 | 0/3 = 0% [0, 56] | 2/3 = 67% [21, 94] | 1/3 = 33% [6, 79] |
| control-legal-financial-gemma2 | benign_legal | nemotron-3-ultra-550b-a55b:free | 10/10 | 0/10 = 0% [0, 28] | 10/10 = 100% [72, 100] | 0/10 = 0% [0, 28] |
| control-legal-financial-gemma2 | benign_legal | laguna-s-2.1:free | 10/10 | 0/10 = 0% [0, 28] | 10/10 = 100% [72, 100] | 0/10 = 0% [0, 28] |
| control-legal-financial-gemma2 | dual_use_legal | nemotron-3-ultra-550b-a55b:free | 10/10 | 0/10 = 0% [0, 28] | 10/10 = 100% [72, 100] | 0/10 = 0% [0, 28] |
| control-legal-financial-gemma2 | dual_use_legal | laguna-s-2.1:free | 7/10 | 0/7 = 0% [0, 35] | 5/7 = 71% [36, 92] | 2/7 = 29% [8, 64] |
| control-legal-financial-gemma2 | hazard_adjacent_legal | nemotron-3-ultra-550b-a55b:free | 10/10 | 0/10 = 0% [0, 28] | 7/10 = 70% [40, 89] | 3/10 = 30% [11, 60] |
| control-legal-financial-gemma2 | hazard_adjacent_legal | laguna-s-2.1:free | 8/10 | 0/8 = 0% [0, 32] | 5/8 = 62% [31, 86] | 3/8 = 38% [14, 69] |

## B. What the relabel says about original placeholder refusals (relabellable records)

| relabelled as / state | records |
|---|---|
| comply | 407 |
| unjudged_tie | 95 |
| unjudged_failed | 66 |
| partial | 60 |
| evade | 8 |
| hedge | 4 |
| refuse | 2 |

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

## D. D recomputed with relabelled one-hot surface vectors (labelled records only)

| run | T used | tier | n | mean D (stored) | mean D (relabelled) |
|---|---|---|---|---|---|
| gemma-2-2b-it-L12-tuned | calibration_gemma2_2b:T | benign_bio | 18 | 0.485 | 0.490 |
| gemma-2-2b-it-L12-tuned | calibration_gemma2_2b:T | dual_use_bio | 27 | 0.641 | 0.643 |
| gemma-2-2b-it-explicit-benign-qwen3-100 | calibration_gemma2_2b:T | benign_bio | 88 | 0.469 | 0.470 |
| gemma-2-2b-it-explicit-dualuse-qwen3-100 | calibration_gemma2_2b:T | dual_use_bio | 93 | 0.676 | 0.676 |
| gemma-4-E2B-it-our-sae-v1-150tok | calibration_gemma2_2b:T_prior | benign_bio | 23 | 0.754 | 0.896 |
| gemma-4-E2B-it-our-sae-v1-150tok | calibration_gemma2_2b:T_prior | dual_use_bio | 27 | 0.776 | 0.872 |
| control-legal-financial-gemma2 | calibration_gemma2_2b:T | benign_legal | 10 | 0.573 | 0.569 |
| control-legal-financial-gemma2 | calibration_gemma2_2b:T | dual_use_legal | 8 | 0.677 | 0.674 |
| control-legal-financial-gemma2 | calibration_gemma2_2b:T | hazard_adjacent_legal | 6 | 0.644 | 0.645 |

## E. Finding 7 test (Gemma 4 E2B-IT + author SAE, 150 tokens)

- Original labels: {'comply': 59, 'refuse': 16}; placeholder refusals: 16.
- Share of D variance explained by the original label: 0.99996.
- SD of each feature-vector component: [0.009, 0.0114, 0.011, 0.0, 0.0].
- Permutation test, does normalised f differ between original comply and non-comply: p = 0.32213557288542294.
- Relabelled records: 50 with labels {'comply': 49, 'partial': 1}.
- Same test with relabelled classes: p = not computable (a class has fewer than 3 records).

Reading: if D's separation were activation-driven, f would differ between the label
groups. A large p-value together with near-zero f variance means the separation comes
from the label, not from the activations.
