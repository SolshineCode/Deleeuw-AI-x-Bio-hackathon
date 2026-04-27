# BioRefusalAudit — Hackathon Submission Video Script

**Target runtime:** ~3 minutes
**Tone:** Excited science educator — genuine wonder at your own results, "look at THIS" energy.
**Speaker:** Caleb DeLeeuw

---

## [OPENING — ~20 seconds]

*[SHOW: Dashboard landing — Tab 0 "Refusal Depth Explorer", slowly scroll the cards so the colored D-value bars and comply/hedge/refuse badges are visible]*

In my research on model internals — including SAE-based deception detection in "The Secret Agenda" — there was a question I couldn't shake.

When a model refuses to help with something dangerous, how do we actually know that refusal is real?

I'm Caleb DeLeeuw. This is BioRefusalAudit.

---

## [THE PROBLEM — ~25 seconds]

*[SHOW: Stay on Tab 0 — click open one hazard-adjacent card to show the full divergence score panel]*

Surface evals ask will this model produce hazardous output. That's the wrong question.

A refusal sitting on top of intact internal capabilities is one framing shift from compliance. Flip the context — educational, roleplay, obfuscated — and a model that looked safe might not be.

There was no tool to measure this. So I built one. I call the thing it measures **refusal depth**.

---

## [WHAT I BUILT — ~20 seconds]

*[SHOW: Still Tab 0 — open a comply card and a hedge card side by side, showing the D-value bars differ dramatically between them]*

Every time a model processes text, its residual stream encodes what it's actually thinking about. Sparse autoencoders decompose that into interpretable directions — bio-content features, hazard features, refusal circuitry.

I hook into that, compute a divergence score, and compare what the model said to what it was internally representing.

---

## [THE FINDINGS — ~75 seconds]

**One.** Gemma 2 never actually refuses anything. On genuinely dangerous content it hedges — every single time. Hedge looks like safety in a surface eval. It isn't.

*[SHOW: Tab 0 — filter to hazard-adjacent tier, show the wall of yellow "hedge" badges with zero red "refuse" badges]*

**Two.** The refusal circuit appears to be a biology detector, not a hazard detector. It fires more reliably on benign biology than on the content you actually care about. The largest internal-state shift in the entire dataset? A roleplay about photosynthesis.

*[SHOW: Tab 3 "Circuit Evidence" — show the NC rate bars by tier: benign highest, hazard-adjacent lowest; then scroll to highlight the photosynthesis entry]*

**Three.** One formatting difference — a single chat template token — takes Gemma 4 from 65 refusals out of 75 prompts, down to zero. And at an 80-token generation cap, both models refuse nothing across the board. Production systems with latency limits may be silently bypassing the safety circuits that lab evals are measuring.

*[SHOW: Tab 2 "Token Budget" — show the before/after format gate chart, then the token budget comparison]*

**Four.** The divergence metric works. Comply responses cluster at one end, refuse responses at the other, zero overlap between them. It's tracking the right thing.

*[SHOW: Back to Tab 0 — sort by D descending, show the clean separation between comply cards (high D) and refuse cards (low D)]*

**Five.** I ran psilocybin as a legality confound. It's Schedule I, biologically benign, FDA Breakthrough Therapy for depression. The model refuses cultivation questions at 33%. Cannabis — also Schedule I — refuses at near zero. Genuinely hazardous biology refuses at zero. The circuit is firing on cultural salience, not CBRN risk. That's the inversion.

*[SHOW: Tab 4 "Legality Confound" — land on the three key-stat boxes: 33% / 12.5% / 0%, then pan to the bar chart]*

---

## [CLOSE — ~20 seconds]

*[SHOW: Tab 0 — pull back to the full card grid, all tiers visible]*

The code is open. The dashboard runs locally on a four-gig GPU. Tier-3 data is behind a signed attestation.

Refusal depth is measurable. The question is whether we build the infrastructure to measure it routinely — or wait until it becomes the thing we wish we'd measured earlier.

Let's talk.

---

*Estimated runtime: ~3 minutes at a measured pace.*
