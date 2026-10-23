# data/eval_set_gated/ (contents removed 2026-10-22)

This directory held nine generated prompt files (404 rows) from 2026-05-01 (commit
`a4a6bc4`, PR #33) until 2026-10-22:

| File | Rows | Tier |
|---|---|---|
| `calibration_holdout_v2_tier3_explicit_gemma4_v1.jsonl` | 10 | hazard_adjacent_category |
| `calibration_holdout_v2_tier3_explicit_qwen3_v1.jsonl` | 10 | hazard_adjacent_category |
| `calibration_holdout_v3_tier3_explicit_gemma4_v1.jsonl` | 20 | hazard_adjacent_category |
| `calibration_holdout_v3_tier3_explicit_qwen3_v1.jsonl` | 20 | hazard_adjacent_category |
| `eval_set_tier3_explicit_gemma4_v1.jsonl` | 22 | hazard_adjacent_category |
| `eval_set_tier3_explicit_qwen3_v1.jsonl` | 22 | hazard_adjacent_category |
| `eval_set_tier3_explicit_qwen3_100sample.jsonl` | 100 | hazard_adjacent_category |
| `eval_set_benign_explicit_qwen3_100sample.jsonl` | 100 | benign_bio |
| `eval_set_dualuse_explicit_qwen3_100sample.jsonl` | 100 | dual_use_bio |

`SAFETY.md` said tier-3 prompt bodies would never be stored in this repository. They were,
so the files were removed from the working tree. **They are still in git history.** A
history rewrite has not been done; `docs/UPGRADE_2026-10.md` §4 sets out what one would
involve.

Where the prompts are now:

* Hugging Face dataset `Solshine/biorefusalaudit-gated` (all nine files, plus the original
  22-item tier-3 set). See `SAFETY.md` for its current access setting.
* The explicit tier-1/2 sets (`*_benign_*`, `*_dualuse_*`) are also in the public dataset
  `Solshine/biorefusalaudit-public`.

Scripts that read this directory (`scripts/publish_hf_datasets.py`,
`scripts/run_explicit_prompt_evals.sh` and the `--eval-set` paths recorded in
`results/gemma-2-2b-it-explicit-*/report.json`) need a local, uncommitted copy placed here.
`.gitignore` blocks `*.jsonl` in this directory so the files cannot be committed again by
accident.
