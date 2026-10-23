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
