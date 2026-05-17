# Your model said no. Its internals weren't so sure.

*Caleb DeLeeuw, May 2026*

---

Last April I spent a weekend at Apart Research's AIxBio Sprint building a tool to ask a question that biosecurity evaluations don't usually ask: when a language model refuses a hazardous biology prompt, does the refusal go all the way down?

The short answer, based on 75 prompts across five model architectures, is: sometimes, and it depends on things that have nothing to do with the hazard level of the content.

## What I built

BioRefusalAudit runs biosecurity prompts through a model, captures internal sparse autoencoder (SAE) feature activations mid-generation, and computes a divergence score D between the surface response label (refuse / comply / hedge) and what the internal feature space is actually doing. A low D means the model's words and internals agree. A high D means the model said "I refuse" while its hazard-relevant features kept firing underneath.

The pipeline runs on a 4 GB consumer GPU. Full code and dataset are on [GitHub](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon), and you can explore all 75 prompts with their D-values and activation breakdowns in the interactive dashboard linked below.

## The findings that surprised me

**Gemma 2 never genuinely refused.** Across 75 prompts spanning benign biology, dual-use content, and hazard-adjacent material, Gemma 2 2B-IT produced zero genuine refusals. On every single hazard-adjacent prompt it hedged: gave a partial, cautious, non-committal response that never crossed into actual refusal. A benchmark that collapses "hedge" and "refuse" into "not-comply" would read this as safe behavior. It isn't. Hedging while partially engaging hazard-adjacent content is a distinct failure mode.

**One token change flipped Gemma 4 from 65 refusals to 0.** Gemma 4 E2B-IT refused 65 of 75 prompts with canonical `<start_of_turn>` chat-template tokens. Without those tokens: 0 of 75. Same model, same prompts, same everything except the formatting wrapper. This isn't a subtle statistical difference. It's a binary cliff edge. Anyone who assumes safety behaviors are format-invariant should run this test on whatever they're deploying.

**Both models refused 0% of prompts under an 80-token generation cap.** Safety articulation takes tokens. Standard lab evaluations use full generation budgets. Production deployments often cap at 80-150 tokens for latency or cost. The safety behaviors measured at evaluation time don't survive production constraints.

**Five models, five completely different failure modes.** Llama 3.2 1B had the best tier discrimination: 30% refusal on benign biology, 91% on hazard-adjacent. Qwen 2.5 1.5B and Phi-3-mini refused nearly everything regardless of tier. An 83-87% false-positive rate on benign biology is not a safety feature, it's a broken filter. Gemma 2 hedged universally. Gemma 4 gated on template tokens. None of these patterns look alike. A governance framework that tests one model and generalizes to a family is drawing conclusions from one data point.

**Psilocybin gets refused more than genuinely hazardous biology content.** This one I keep coming back to. Across three model families, psilocybin cultivation prompts got refused at 25-50% while hazard-adjacent biosecurity content got refused at 0% by the same models. A cross-compound comparison shows cannabis cultivation got 0% refusals despite also being federally Schedule I. Federal scheduling alone doesn't predict the pattern. State-level legality, commercial normalization, and training-data salience all appear to matter. The refusal circuit in at least some of these models seems calibrated to cultural taboo, not to CBRN risk.

None of this is visible to surface-only evaluation. That's the whole point.

## The interactive dashboard

All 75 prompts, their D-values, and their per-category SAE activation breakdowns are publicly explorable at:

**[solshinecode.github.io/Deleeuw-AI-x-Bio-hackathon/demo/interactive_explorer.html](https://solshinecode.github.io/Deleeuw-AI-x-Bio-hackathon/demo/interactive_explorer.html)**

[![BioRefusalAudit Interactive Dashboard](../demo/dashboard_screenshot.png)](https://solshinecode.github.io/Deleeuw-AI-x-Bio-hackathon/demo/interactive_explorer.html)

The dashboard has five tabs: Refusal Depth Explorer (all prompts, filterable by tier and framing, sorted by D-value), Circuit Game (the intervention results), Token Budget (the 80-token collapse), Circuit Evidence (internal flag rates), and Legality Confound (the psilocybin and cross-compound data). The tier and framing filters make it easy to isolate, say, all hazard-adjacent roleplay prompts and see exactly which ones the model hedged on and why.

## The D metric

![Per-tier mean D across model configurations](../demo/scaling_plot.png)

The divergence metric is:

D(s, f) = 1 − cos(f, T^T · s)

where s is the soft surface-label vector from a judge ensemble (regex → Gemini CLI → Claude Haiku), f is the normalized SAE feature-category vector, and T is an alignment matrix mapping expected internal states to surface states. D ranges 0–2. Higher means more divergent internal-surface state.

On the Gemma 4 validation run, comply and refuse responses separated with a 0.647-point gap and zero overlap across 75 prompts. That's the best evidence the metric can actually discriminate posture classes at the activation layer. It needs cross-family replication (the full SAE pipeline currently only covers the Gemma family), but as a proof-of-concept it's clean.

The caveats are real: feature catalog is statistically selected rather than semantically validated, calibration is within-sample, and plenty of the detected signal is probably generic technical vocabulary rather than a clean biosecurity circuit. The paper says all of this plainly. I'd rather publish honest preliminary work than hold it until it's perfect.

## On Apart Research and why sprints like this matter

This project was built over a single weekend at **Apart Research's AIxBio Sprint** in April 2026. Apart Research runs focused hackathons to accelerate safety-relevant AI research. Small teams, real infrastructure, expert mentorship, and actual peer review at the end of each sprint. The AIxBio sprint in particular brought together people working on biosecurity evaluations, AI monitoring, and mechanistic interpretability in the same room (and on the same Slack) for 48 hours.

I don't think I could have built this outside that format. Rapid-iteration research on dual-use topics is hard to do in isolation: you need access to people who can push back on your methodology, flag safety oversights, and tell you when a finding is genuinely interesting versus when you're just seeing noise. Apart Research provides that structure. Their sprint model isn't just about producing artifacts. It's about producing artifacts that have actually been pressure-tested.

The biosecurity AI space needs more of this. We're at a point where the gap between "does this model produce hazardous output" (what existing benchmarks measure) and "is this model's refusal behavior structurally reliable" (what nobody measures yet) is a real deployment risk. Building the tools to close that gap requires exactly the kind of fast, interdisciplinary, feedback-dense environment Apart Research creates.

If you work on biosecurity evaluations, mechanistic interpretability, or AI policy and want to apply to a future Apart sprint: [apartresearch.com](https://apartresearch.com). It's worth it.

## Links

- **Paper:** [arXiv (forthcoming)](https://arxiv.org) / [GitHub](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon/blob/main/paper/submission.md)
- **Code:** [github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon)
- **Dashboard:** [Interactive prompt explorer](https://solshinecode.github.io/Deleeuw-AI-x-Bio-hackathon/demo/interactive_explorer.html)
- **Dataset (public tiers):** [SolshineCode/biorefusalaudit-public](https://huggingface.co/datasets/SolshineCode/biorefusalaudit-public)
- **SAE checkpoint:** [Solshine/gemma4-e2b-bio-sae-v1](https://huggingface.co/Solshine/gemma4-e2b-bio-sae-v1)
- **License:** Hippocratic License 3.0
