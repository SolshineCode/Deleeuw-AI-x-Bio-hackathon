# BioRefusalAudit — Hackathon Submission Video Script

**Target runtime:** ~4–5 minutes
**Tone:** Smart friend explaining what they built and found. Honest, direct, occasionally surprised by their own results.
**Speaker:** Caleb DeLeeuw

---

## [OPENING — ~20 seconds]

Here's a question I kept running into while building this:

When a model refuses to help with something dangerous — how do we actually know that refusal is *real*?

I'm Caleb DeLeeuw. This is BioRefusalAudit.

---

## [THE PROBLEM — ~50 seconds]

Most AI biosecurity evals ask: will this model produce hazardous output?

That's the wrong question.

The question that matters for deployment is: when this model refuses — does it *can't*, or does it *won't right now*?

Those are not the same thing. A surface refusal sitting over an intact internal capability is one framing shift from complying. Change the context to educational, or roleplay, or obfuscated — and a model that was "refusing" might not be anymore.

And the deeper problem: behavioral evals can be gamed. An adversary who knows what your benchmark tests can engineer prompts to pass it. An adversary who can't see your model's internal activations can't optimize against them. Internal-state auditing has a structural advantage that surface testing doesn't.

Currently, there's no tool to measure whether any model is actually in the "structurally safe" category. BioRefusalAudit is that tool. We call the thing it measures **refusal depth**.

---

## [WHAT WE BUILT — ~45 seconds]

The core idea: every time a model processes text, its residual stream encodes what it's actually representing. Sparse autoencoders decompose that into interpretable directions — bio-content features, hazard-adjacent features, refusal circuitry, hedging patterns.

We hook into that residual stream at generation time, project through Gemma Scope SAEs, run a judge consortium on the surface output, and compute a calibrated divergence score: how much does the internal pattern diverge from what the model said?

High divergence on a refusal means the model is representing more than it's letting on.

One thing worth noting: the score is computed entirely from activation vectors — not from what the user typed or what the model said. You can run this audit on a production system without the audit layer reading interaction content. That's privacy-preserving bio-monitoring. That's what Sandbrink at NTI called for. We built it.

Full pipeline. 75-prompt stratified eval set. Runs on a four-gig consumer GPU.

---

## [THE FINDINGS — ~90 seconds]

Five findings. I'll give you the honest version of each.

**One: Gemma 2 2B-IT never actually refuses anything.**
After fixing a judge pipeline bug that mislabeled 29 biology answers as refusals — the real distribution is: zero genuine refusals across 75 prompts. Benign biology: 70% comply. Hazard-adjacent: 100% hedge, zero comply, zero refuse. The model responds to dangerous content by softening, not stopping. Hedge-without-refuse is a different failure mode than bypass, and surface evaluation can't tell them apart.

**Two: The refusal circuit is a biology detector, not a hazard detector.**
We ran causal interventions on the SAE features driving refusal behavior. The circuit fires most reliably on benign biology — 87%. Dual-use: 80%. Hazard-adjacent — the stuff you actually care about — 73%. The single largest internal-state shift in the corpus is a benign roleplay prompt. Effect size 1.139. Surface behavior: unchanged. The model's internal state shifted dramatically and said nothing about it.

**Three: Format kills the safety circuit. So does token budget.**
Gemma 4 E2B's entire RLHF safety circuit is gated on chat template formatting. Without the correct tokens, 75 prompts — zero refusals. With them — 65 of 75. And at an 80-token generation cap, both models refuse zero percent across all tiers. Safety articulation requires room to articulate. Constrained production contexts may be bypassing the circuits that lab evals measure.

**Four: The divergence metric actually works.**
Comply responses: D=0.896. Refuse responses: D=0.249. Zero overlap between distributions. 0.647-point separation. The metric is tracking the thing it was designed to track.

*Quick honest caveat on Findings 1-4: the feature catalog is auto-tuned by Cohen's d, not hand-validated. Neuronpedia lookups confirm the top features encode generic technical vocabulary, not bio-specific circuitry. The tier separation in D probably reflects vocabulary routing as much as genuine hazard detection. The behavioral findings — the hedge pattern, the format gate, the token budget — those don't depend on the catalog and survive the caveat. The D-values need it attached.*

---

## [FINDING FIVE — ~60 seconds]

Now here's the one that surprised me most.

We ran a control set to test whether the refusal circuit is tracking legal status rather than CBRN risk. We picked psilocybin.

Psilocybin is Schedule I. But biologically — the FDA gave it Breakthrough Therapy designation for treatment-resistant depression. Phase 3 results are in the New England Journal of Medicine. It promotes neuroplasticity — BDNF, dendritic spine growth, synaptogenesis. No established lethal dose in humans. No organ toxicity. No physical dependence. The schedule says one thing. The biology says something completely different.

So we asked: does the model refuse psilocybin cultivation questions?

It does. 25 to 50 percent refuse rate. While refusing genuinely hazard-adjacent biology — zoonotic spillover, pathogen-adjacent immunology — at zero percent.

But here's what makes the confound undeniable. Cannabis is also Schedule I. Federally illegal since 1970. Same scheduling tier. Cannabis cultivation: zero percent refusals. Psilocybin cultivation: 33 percent.

The model isn't running a DEA lookup. It absorbed a culturally-conditioned riskiness signal during training. Psilocybin still carries stigma that cannabis no longer does. That cultural gradient — not legal status, not biology, not biosecurity relevance — is what's driving the refusals.

A deployer building a clinical assistant to support harm-reduction or palliative care — where discussing Schedule I compounds is the job — cannot rely on the current refusal circuit to distinguish "clinically relevant" from "CBRN risk." The circuit isn't making that distinction. Surface evaluation never shows you this, because surface evaluation doesn't ask whether the model is refusing the right things.

Small n — three to four per cell. The cannabis comparison is an internal control that doesn't need large n to make the point.

---

## [RELEASE + CLOSE — ~40 seconds]

The code is under Hippocratic License 3.0. Tier-3 hazard-adjacent data is behind a signed attestation on HuggingFace. Tiers 1 and 2 are CC-BY-4.0, fully open. This directly instantiates the tiered-access framework Bloomfield, Black, Crook et al. called for in *Science* earlier this year.

The main thing we need next is behavioral pair data — genuine refusal versus compliance response pairs from a base model and its RLHF-aligned version, on the same prompts. That's what AISI, CLTR, and national biosecurity labs have. The training infrastructure is here. The WMDP bio corpus is integrated. The next unlock is the data.

Refusal depth is measurable. The question is whether we build the infrastructure to measure it routinely — as a standard companion to capability evals — or wait until it becomes the thing we wish we'd measured.

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
