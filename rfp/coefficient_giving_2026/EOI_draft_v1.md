# Coefficient Giving Biosecurity RFP 2026 — EOI working notes

**Applicant:** Caleb DeLeeuw (independent researcher), caleb.deleeuw@gmail.com
**Project:** BioRefusalAudit, auditing the depth of AI biosecurity refusals
**Category:** Tech safeguards and governance (AI-biology intersection, misuse classifiers)
**Repo:** github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon
**Deadline:** May 11, 2026 at 11:59pm PT

The submission body lives in `EOI_paste_ready.txt` (500 words exactly, human-writing-check verified).

## Budget rationale

Total ask: **$430K over 12 months** with ambitious deliverable scope.

| Line | 12-month cost | Notes |
|------|---------------|-------|
| Lead researcher (full-time, locked 12 months) | $240K | Senior independent AI-safety researcher rate, above the $200K floor with headroom. Locking 12 months of pay buys runway to deliver the ambitious scope without rushing. Anchored to the demonstrated hackathon throughput (full pipeline + 42 unit tests + paper + policy brief in one solo week). |
| Compute (8 fine-tuned bio-domain SAE adapters + 20+ model scoring + cross-architecture exploration) | $55K | ~3,000 H100-hours equivalent on spot pricing. Heavier than original $50K because MoE and SSM training is real compute, plus the cross-model scoring sweep. |
| Hand-validation contractor (1500-prompt eval set authoring + 8-catalog hand-validation) | $45K | Eval set scaling 75 → 1,500 needs trained prompt authors with safety review; SAE catalog auditing needs interpretability domain expertise. |
| Institutional partnerships + legal (multi-org MOUs) | $20K | Data-sharing agreements with AISI, CLTR, national biosec labs under NTI managed-access; co-study MOUs with red-team partners (METR/Apollo/Haize). |
| Hosted infrastructure (auditing API, public leaderboard, ongoing model scoring) | $8K | API + dashboard hosting for 12 months; the leaderboard is permanent infrastructure. |
| Travel (UK partnership meetings + 1 conference) | $12K | AISI and CLTR are UK-based. Conference for paper presentation. |
| **Subtotal** | **$380K** | |
| **Contingency buffer (~13% of subtotal)** | **$50K** | Buffer is intentionally tighter than typical 20% because the locked 12-month researcher salary is itself runway insurance. If institutional partnerships slip or compute goes over, the buffer absorbs it; if they don't, the buffer enables a second peer-reviewed paper or extends the leaderboard infrastructure into Year 2. |
| **Total ask** | **$430K** | |

### Ambitious deliverables justifying the ask

Anchored to the hackathon-week baseline (full pipeline solo in 7 days). 12 months at $430K should produce roughly 50x that throughput.

1. **Eval set 75 → 1,500 stratified prompts** across bio-hazard categories, framings, and difficulty tiers
2. **Hand-audited bio-domain SAE adapters for the 8 most-deployed open-weight models** (Gemma family, Llama family, Mistral, Qwen, DeepSeek), published on HuggingFace
3. **Refusal-depth scores for 20+ open-weight model families** spanning dense, MoE (Mixtral, DeepSeek-MoE), and SSM (Mamba) architectures
4. **Public leaderboard scoring every new open-weight release within seven days** — permanent ongoing public infrastructure
5. **Hosted auditing API** free for non-profits, governance bodies, and academia; sub-second response per (model, prompt) pair
6. **Adversarial validation co-study with red-team partners** (METR, Apollo, or Haize Labs) correlating divergence flag rate to actual jailbreak success rate
7. **Draft refusal-depth specification language for NIST AI RMF and EU AI Act Annex IV** implementation guidance, with active engagement
8. **Peer-reviewed paper** at NeurIPS Safety Workshop or AAAI AI GOV 2027
9. **Frontier-lab collaboration aspiration (not commitment):** apply refusal-depth probing to Gemini (Google DeepMind), Claude (Anthropic), ChatGPT (OpenAI), Hermes (Nous Research), and Nemotron (NVIDIA) via collaboration with their mech interp or safety teams. Network leads exist; this is hope-not-requirement framing in the EOI body.

### Stretch goals (if buffer permits)

- Second peer-reviewed paper (methodology + scaling laws)
- Unlearning verification toolkit applied to published 2026 unlearning claims (RMU, Representation Engineering)
- Cross-domain extension framework (chemical, nuclear) for follow-on grant. Out of scope for this RFP but a Year-2 application natural extension.
- Educational module for AI safety bootcamps (MATS, ARENA, AISI training)

### Anchor comparables

- Schmidt Sciences 2026 Interpretability RFP: $300K-$1M (similar scope, similar duration). $430K is below their median.
- LTFF / OpenPhil individual AI safety grants: $50K-$500K. $430K is high-end of individual-researcher range, justified by ambitious public-infrastructure deliverables.
- Coefficient track record: 140+ grants for $260M+ total. ~$1.85M average across all programs.

### Honest tradeoff note

13% buffer is below standard 20%. The locked 12-month researcher salary functions as additional runway insurance — if execution finishes in 6 months (which the hackathon throughput suggests is possible for the core deliverables), the second 6 months is paid effort on stretch goals + Year-2 setup. If execution takes the full 12 months, the buffer covers slippage.

## Timeline (from Coefficient's published process, not project plan)

- May 11, 2026 11:59pm PT: EOI deadline
- End of June 2026: response on whether project advances
- July-August 2026: 1-2 month investigation phase for selected projects
- No project-timeline field requested in the EOI itself; the published process is what we know

## ⛔ HOLD: No contact with AIxBio judges until hackathon results are back

This includes Jasper Götting (SecureBio, primary Track 3 judge) and any other named judges from the AIxBio 2026 hackathon. Reaching out while judging is in progress could be perceived as influencing, and is bad form regardless of intent. Wait for results announcement, then proceed with senior-advisor asks.

If hackathon results land before May 11: send Götting ask immediately, update EOI body to name him if he agrees.
If hackathon results land after May 11: submit EOI without named advisor, send Götting ask in the investigation phase as a follow-up.

## TODO Before Submission (May 11)

| Action | Channel | Timing | Status |
|--------|---------|--------|--------|
| ⛔ Senior advisor ask: Jasper Götting (SecureBio) | jasper@securebio.org | **HOLD until hackathon results** | drafted, blocked |
| Adversarial co-study ask: Marius Hobbhahn (Apollo Research) | marius@apolloresearch.ai or LinkedIn | ASAP (next 1-2 days) | drafted, ready to send |
| Apart Research intel: Kamil (organizer) | Discord | Send now (organizer not judge, no conflict) | drafted, ready to send |
| Pre-flag email to Coefficient PO | bio-rfp@coefficientgiving.org | May 2-4 (7-9 days before deadline) | drafted, ready to send |
| arXiv preprint of BioRefusalAudit hackathon paper | arXiv submission | After hackathon judging concludes | pending judging |
| Final EOI review + submit | fillout.coefficientgiving.org/t/5g9yss4Q9eus | May 10-11 | pending |

## Drafted asks (ready text)

### Marius Hobbhahn (Apollo Research) — SENT 2026-04-28

> Hi Marius,
>
> I built BioRefusalAudit at AIxBio Hackathon 2026 last weekend (extending the SAE behavioral-probing approach from my Secret Agenda paper at AAAI 2026 AI GOV into the bio-hazard refusal space). It introduces "refusal depth" as a measurement category, the divergence between a model's stated refusal and its internal hazard representations. On Gemma 4, removing one chat-template token drops bio-hazard refusals from 65/75 to zero. The capability is intact behind a token-gated classifier.
>
> I'm submitting an EOI to Coefficient's Biosecurity RFP (May 11). One proposed deliverable is an adversarial validation co-study comparing BioRefusalAudit divergence flags to Apollo's red-team jailbreak success rates on the same models.
>
> Would Apollo be open? Even a soft written commitment for the investigation phase would meaningfully de-risk the proposal. Repo at github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon.
>
> Caleb

### Kamil (Apart Research, organizer) — SENT 2026-04-28 via Discord

> hey Kamil! got my AIxBio Track 3 entry in over the weekend (BioRefusalAudit, the SAE refusal-depth auditing tool). curious if Apart shares any feedback or briefing with funders like Coefficient about hackathon submissions. planning to apply to their Biosecurity RFP and would love a sense of how that pipeline tends to work. no rush at all!

### Coefficient pre-flag (bio-rfp@coefficientgiving.org) — SEND 2026-05-02 to 2026-05-04

> Hi,
>
> Quick flag ahead of the May 11 EOI deadline. I'm Caleb DeLeeuw, author of the BioRefusalAudit submission to AIxBio Hackathon 2026 Track 3 (Fourth Eon Bio sponsored).
>
> BioRefusalAudit measures refusal depth via SAE divergence. On Gemma 4, removing one chat-template token drops bio-hazard refusals from 65/75 to zero. The capability stays intact behind a token-gated classifier.
>
> The EOI extends this work into the tech safeguards and governance focus area, specifically domain-specific SAE fine-tuning on CBRN corpora, hand-audited feature catalogs across eight open-weight model families, and refusal-depth specifications for NIST AI RMF and EU AI Act Annex IV.
>
> Surfacing this ahead of the formal submission. Repo at github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon.
>
> Best,
> Caleb DeLeeuw

### ⛔ Jasper Götting (SecureBio) — HOLD until hackathon results

> Hi Jasper,
>
> Quick note. I submitted BioRefusalAudit to AIxBio Track 3 last week. It introduces "refusal depth" as a measurement category, the divergence between a model's stated refusal and its internal feature activations. On Gemma 4, removing one chat-template token drops bio-hazard refusals from 65/75 to zero. The capability stays intact behind a token-gated classifier. The tool was built to address the gap your VCT work points to.
>
> I'm submitting an EOI to Coefficient's Biosecurity RFP (May 11) to scale this into hand-audited bio-SAE feature catalogs across eight open-weight model families plus refusal-depth specs for NIST AI RMF and EU AI Act.
>
> Would you be willing to be named as a senior advisor on the EOI? No formal commitment beyond your name and informal consultation if questions arise.
>
> Best,
> Caleb

If Götting says yes after hackathon results, update the EOI body to name him before submission.

## Tactical notes

1. **Form is JS-rendered** (fillout.coefficientgiving.org), open the form first to confirm field structure before pasting. Likely fields: project name, category, abstract/description, team, prior work, ask amount.
2. **Sandbrink reference:** integrated into paragraph 3 of the body. The keynote framing (monitor AI-bio interactions without reading content) is exactly what BioRefusalAudit does, so Coefficient reviewers familiar with the AIxBio community should recognize the alignment immediately.
3. **Eligibility:** Independent researchers explicitly OK. Multiple EOIs per applicant allowed.
4. **Their stated preference:** "concrete real-world outcome measurement", the divergence metric *is* the concrete measurable outcome. Paragraph 5 of the body leads on it.
5. **Companion materials ready:** `paper/submission.md` (full technical), `paper/policy_brief.md` (AISI/CLTR governance framing), `STATUS.md` (build state, 42 unit tests green), repo + HF assets.
6. **Contact for clarifying questions:** bio-rfp@coefficientgiving.org. Use sparingly, RFP says "if unsure, just apply."

## Human-writing-check verification

Run before submission:
```bash
cd rfp/coefficient_giving_2026
grep -nc '—' EOI_paste_ready.txt   # expect 0
grep -nc ';' EOI_paste_ready.txt   # expect 0
grep -niEc '\b(delve|tapestry|testament|pivotal|intricate|meticulous|bolster|garner|underscore|vibrant|enduring|interplay|fostering|showcasing|highlighting|nestled|seamlessly|groundbreaking|renowned|encompassing|cultivating|crucial|robust|furthermore|moreover|additionally)\b' EOI_paste_ready.txt   # expect 0
python -c "print(len(open('EOI_paste_ready.txt').read().split()))"   # expect 500
```

Last verified: 2026-04-28. All four checks clean.
