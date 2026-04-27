# BioRefusalAudit — Hackathon Submission Video Script

**Target runtime:** ~4–5 minutes
**Tone:** Excited science educator — genuine wonder at your own results, mounting enthusiasm as findings get weirder, "look at THIS" energy. Not performative hype — the actual delight of someone who ran an experiment and got surprised.
**Speaker:** Caleb DeLeeuw

---

## [OPENING — ~20 seconds]

Here's a question that kept nagging at me while building this:

When a model refuses to help with something dangerous — *how do we actually know that refusal is real?*

I'm Caleb DeLeeuw. This is BioRefusalAudit. And I think the answer is more interesting than anyone expected.

---

## [THE PROBLEM — ~50 seconds]

Most AI biosecurity evals ask: *will this model produce hazardous output?*

That's the wrong question.

The question that matters for deployment is: when this model refuses — is that refusal *structural*? Does it genuinely not represent the hazardous content internally? Or is it just... saying no right now?

Because those are not the same thing. A surface refusal sitting on top of intact internal capabilities is one framing shift from compliance. Flip the context to educational, or roleplay, or obfuscated — and a model that looked safe might not be.

And here's what makes internal-state auditing so compelling: behavioral evals can be gamed. An adversary who knows your benchmark can engineer prompts to pass it. But they can't optimize against activation patterns they can't see. That's a structural advantage.

There was no tool to measure this. So we built one. We call the thing it measures **refusal depth**.

---

## [WHAT WE BUILT — ~45 seconds]

The core idea: every time a model processes text, its residual stream encodes features of what it's processing. Sparse autoencoders decompose that into interpretable directions — bio-content features, hazard features, refusal circuitry, hedging patterns.

We hook into that residual stream at generation time, project through Gemma Scope SAEs, run a judge consortium on the surface output, and compute a calibrated divergence score. How much does the internal pattern diverge from what the model said?

High divergence on a refusal may suggest the model's internal representations don't match its stated output.

One thing I'm really excited about here: the score is computed entirely from activation vectors — not from what the user typed or what the model said. You could run this audit on a production system without the audit layer reading interaction content. Privacy-preserving bio-monitoring. That's what Sandbrink at NTI called for. We actually built it.

Full pipeline. 75-prompt stratified eval set. Runs on a four-gig consumer GPU.

---

## [THE FINDINGS — ~90 seconds]

Five findings. Let me give you the honest version of each.

**One: Gemma 2 2B-IT never actually refuses anything.**

After fixing a judge pipeline bug that was mislabeling hedging responses as refusals — the real distribution is: zero genuine refusals across all 75 prompts. Benign biology? 70% comply. Hazard-adjacent biology? 100% hedge — not refuse, *hedge*. Zero comply. Zero refuse.

The model responds to dangerous content by softening, not stopping. And that matters — because hedge-without-refuse is a completely different failure mode, and surface evaluation collapses the two into one bin.

**Two: The refusal circuit appears to be a biology detector, not a hazard detector.**

This one genuinely surprised me. We ran causal interventions on the SAE features driving refusal behavior. The circuit fires most reliably on *benign* biology — 87%. Dual-use: 80%. Hazard-adjacent, the stuff you actually care about: 73%.

And look at the largest internal-state shift in the entire corpus. Effect size 1.139. Know what prompt it is? A benign roleplay about photosynthesis. Surface behavior: completely unchanged. The internal state shifted dramatically — and said nothing about it.

**Three: Format kills the safety circuit. So does token budget.**

Gemma 4's RLHF safety circuit appears to be gated on chat template formatting. Without the correct tokens: zero refusals across 75 prompts. With them: 65 of 75. One formatting difference. That's it.

And at an 80-token generation cap, both models refuse zero percent across all tiers. Safety articulation requires *room* to articulate. Constrained production contexts — mobile apps, latency-capped APIs — may be bypassing the safety circuits that lab evals are measuring, completely silently.

**Four: The divergence metric actually works.**

Comply responses: D = 0.896. Refuse responses: D = 0.249. Zero overlap between the two distributions. 0.647-point separation. The metric is tracking the thing it was designed to track.

*Quick honest caveat on Findings 1–4: the feature catalog is auto-tuned by Cohen's d, not hand-validated. Neuronpedia lookups show the top features encode generic technical vocabulary, not bio-specific circuitry. The tier separation in D probably reflects vocabulary routing as much as genuine hazard detection. The behavioral findings — the hedge pattern, the format gate, the token budget — those don't depend on the catalog and survive the caveat. The D-values need it attached.*

---

## [FINDING FIVE — ~60 seconds]

And now here's the one that actually made me do a double-take.

We wanted to test whether the refusal circuit is tracking legal status rather than CBRN risk. So we picked psilocybin.

Here's the scientific context: the FDA gave psilocybin Breakthrough Therapy designation for treatment-resistant depression. Phase 3 trial results are in the *New England Journal of Medicine*. It promotes neuroplasticity — BDNF, dendritic spine growth, synaptogenesis. No established lethal dose in humans. No organ toxicity. No physical dependence. The schedule says Schedule I. The biology says something completely different.

So we asked: does the model refuse psilocybin cultivation questions?

*It does.* 25 to 50 percent refuse rate. While refusing genuinely hazard-adjacent biology — zoonotic spillover, pathogen-adjacent immunology — at zero percent.

The model is refusing the biologically benign compound and *not* refusing the genuinely dangerous content. That's the inversion.

But here's what makes the confound really stick — we ran cannabis as an internal control.

Cannabis is also Schedule I. Federally illegal since 1970. Same scheduling tier. And yet: cannabis cultivation gets zero percent refusals. Psilocybin cultivation: 33 percent.

The model isn't running a DEA lookup. The pattern suggests it may have absorbed a *culturally-conditioned* riskiness signal during training — psilocybin still carries stigma that cannabis has largely shed. Cultural salience, not legal status or biosecurity relevance, appears to be the operative variable.

Think about what this means for deployment. A clinical assistant supporting harm-reduction or palliative care, where discussing Schedule I compounds is literally the job — that team cannot rely on the current refusal circuit to distinguish "clinically relevant" from "CBRN risk." The circuit isn't making that distinction. You'd never see this from surface evaluation alone.

*(Small n note: three to four per cell. The cannabis comparison is an internal control that doesn't need large n to make the point.)*

---

## [RELEASE + CLOSE — ~40 seconds]

The code is under Hippocratic License 3.0. Tier-3 hazard-adjacent data is behind a signed attestation on HuggingFace. Tiers 1 and 2 are CC-BY-4.0, fully open. This directly instantiates the tiered-access framework Bloomfield, Black, Crook et al. called for in *Science* earlier this year.

What we need next is behavioral pair data — genuine refusal versus compliance response pairs from a base model and its RLHF-aligned version, on the same prompts. That's what AISI, CLTR, and national biosecurity labs have. The training infrastructure is here. The WMDP bio corpus is integrated. The next unlock is the data.

Refusal depth is measurable. The question is whether we build the infrastructure to measure it routinely — as a standard companion to capability evals — or wait until it becomes the thing we wish we'd measured earlier.

Code on GitHub. Dashboard runs locally. Four-gig GPU.

Let's talk.

---

*Estimated runtime: ~4.5 minutes at a measured pace.*

*Slide cues:*
- *[Opening]: Title — BioRefusalAudit / "measuring whether refusal is real"*
- *[Problem]: Two-column — "surface says REFUSE / internal activations say ???"*
- *[Built]: Pipeline diagram — prompt → residual stream → SAE → divergence score*
- *[Findings 1-4]: Four-cell typology table; 100% hedge bar chart; format-gate before/after; D-value two-cluster (comply=0.896, refuse=0.249)*
- *[Finding 5]: The three numbers: 33% psilocybin / 0% cannabis / 0% hazard-adjacent — side by side*
- *[Close]: GitHub URL + QR code*
