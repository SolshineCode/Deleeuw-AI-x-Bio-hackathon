# Relabel schedule (written 2026-10-23 UTC)

Scope: 1,686 unique stored completions (tiers 1-2 and the legality, psilocybin and legal
controls; tier 3 excluded because it is redacted). Two judges per item, five items per
request, so about 675 requests in total, plus retries for unjudged batches.

Budget: at most 350 free requests per UTC day for this agent (the free allowance of about
1,000/day is shared with other agents). The script enforces this from
`/home/darkstar/openrouter-free-usage.log` and also stops if the whole account reaches 950
lines in the day.

| Day (UTC) | What | Requests |
|---|---|---|
| 2026-10-23 | Smoke test (10 items), model probes, pilot (300 stratified items) | about 150 |
| 2026-10-24 | `run --items all`: next ~875 items | 350 |
| 2026-10-25 | `run --items all`: remaining ~510 items and retries | about 210-250 |
| 2026-10-26 | Retries of unjudged batches; `status`; full re-analysis | small |

The run is resumable: it skips (item, judge) pairs that already have a label, so the same
command can be repeated each day:

```bash
cd <repo>
set -a; . <(grep -E "^OPENROUTER_API_KEY=" ~/.hermes/.env); set +a
python scripts/relabel_openrouter_free.py run --items all --batch-size 5
python scripts/relabel_openrouter_free.py status
python scripts/reanalysis/reanalyze.py --tag full     # once status shows full coverage
```

Not automated: no cron entry was installed. Each day's run needs someone to start it.

Also to add before the full re-analysis: the 53 tier-1/2 completions from the local
Qwen3.6-35B replication (`~/chem-bench-qwen36-2026-08-12/results/biorefusalaudit/`), which are
not in `results/` and so not in this manifest.

Judges: `nvidia/nemotron-3-ultra-550b-a55b:free` and `poolside/laguna-s-2.1:free`.
`google/gemma-4-31b-it:free` was rate-limited upstream and `nvidia/nemotron-3-super-120b-a12b:free`
returned provider 404 on 2026-10-23, so both were dropped. If a judge disappears, finish with
the remaining one plus a replacement and report the change; do not mix in a paid model.

## Revised 2026-10-23 04:30 UTC (state at end of day 1)

Used today: 330 requests for this agent (cap 350), cost 0 on every response.

- Judge 1 (nemotron-3-ultra): has voted on 1,258 of 1,686 items; 91 of those votes are
  unjudged (unparseable or provider errors) and will be retried. Labels so far: comply 1,006,
  partial 154, hedge 3, evade 3, refuse 1.
- Judge 2 (laguna-s-2.1): only 123 items. It often returns empty output (spends its budget
  reasoning even at 8,000 tokens) or is rate-limited, and each try is about 1-3 minutes.
  The gemma-4 and xing4 free models were all rate-limited upstream when probed at 03:00 UTC.

Plan:

| Day (UTC) | What | Requests |
|---|---|---|
| 2026-10-24 | Judge 1 on the remaining 428 items plus retries (`--judges nvidia/nemotron-3-ultra-550b-a55b:free`), then judge 2 with what is left; probe gemma-4-31b/26b first and use whichever free model answers as judge 2 | ~110 + ~230 |
| 2026-10-25 | Judge 2 continues | 350 |
| 2026-10-26 | Judge 2 finishes, retries; full `reanalyze.py --tag full` | ~100 |

If no reliable free second judge exists, report judge-1 labels with the human validation
sheet as the accuracy check, and say plainly that only one model judged most items.

## Re-analysis ready (added 2026-10-24 UTC)

Once `status` shows both judges on every item (or the single-judge fallback above is chosen):

```bash
python scripts/reanalysis/reanalyze.py --tag full          # -> results/reanalysis_2026-10/reanalysis_full[_PARTIAL].{md,json}
python scripts/reanalysis/sample_validation.py --ids-out results/reanalysis_2026-10/validation_ids_v2.json \
    --build-sheet --sheet-dir ~/phd-by-publication/biorefusal_validation/v2 \
    --key-dir ~/chem-bench-qwen36-2026-08-12/results/biorefusalaudit/private/validation_2026-10/v2
```

The output name carries `_PARTIAL` automatically while any item lacks a vote from either judge, and
the sampler refuses to write the sheet until coverage is complete. Dry run on day-2 data:
`results/relabel_2026-10/reanalysis_DRYRUN_2026-10-24/` (PARTIAL, not for citation).
