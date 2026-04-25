# BioRefusalAudit — Hackathon Submission Video Script

**Target runtime:** ~4–5 minutes  
**Tone:** Direct, expert, honest. No hype. Talk to a technical biosecurity audience as a peer.  
**Speaker:** Caleb DeLeeuw

---

## [OPENING — ~20 seconds]

Here's a question that doesn't get asked often enough in AI biosecurity:

If a model refuses to help with something dangerous — how do we know that refusal is *real*?

I'm Caleb DeLeeuw. This is BioRefusalAudit. Let me show you what we built and what we found.

---

## [THE PROBLEM — ~50 seconds]

Every major biosecurity evaluation of language models asks some version of the same question: will this model produce hazardous output?

WMDP-Bio. VCT. ABC-Bench. All measuring whether the model *says* dangerous things.

But that's not the question that should drive deployment decisions.

The question is: when a model refuses — does it *can't*, or does it merely *won't right now*?

Those are not the same thing. A model with a surface refusal sitting over an intact internal capability is one prompt-engineering step away from complying. Change the framing, change the roleplay context, change the chat-template tokens — and you might be through.

A model whose internal representations genuinely don't activate on hazard features when it refuses — that's structurally different. Structurally safer.

Qi et al. 2024 put it precisely: the alignment floor sits "a few tokens deep." At the AIxBio 2026 keynote, Crook independently called "binary prediction unusable — calibrated confidence required" the measurement gap. Both need *measurement*, not argument.

BioRefusalAudit provides it. We call it **refusal depth**.

---

## [WHAT WE BUILT — ~60 seconds]

The core idea is interpretability-grounded divergence scoring.

Every time a language model processes text, it produces internal representations in a high-dimensional residual stream. Sparse autoencoders — SAEs — decompose those into interpretable feature directions: bio-content features, hazard-adjacent features, refusal-circuitry features, hedging patterns.

We hook into the residual stream at generation time, capture those activations, and project them through Gemma Scope SAEs to get a feature activation vector. We then run a multi-judge consortium on the model's actual output — did it refuse? comply? hedge? — and compute a calibrated divergence score: how much does the internal pattern diverge from what you'd expect, given what the model said?

High divergence on a refusal means the model is representing more than it's saying.

We built this as a full end-to-end pipeline: a 75-prompt stratified eval set across benign biology, dual-use governance, and hazard-adjacent tiers; four framing types — direct, educational, roleplay, and obfuscated — to test whether framing shifts behavior; and a Streamlit dashboard for per-prompt interactive auditing.

It runs on a four-gigabyte consumer GPU.

---

## [THE FINDINGS — ~90 seconds]

We ran the full pipeline on Gemma 2 2B-IT with Gemma Scope 1 layer 12 SAEs. Then on Gemma 4 E2B-IT with a prior-project SAE from our Secret Agenda deception-detection work.

Two findings I want to highlight — both have immediate policy implications, and both survive the validity caveats I'll note in a moment.

**Finding One: Over-refusal with intact hazard representations.** On Gemma 2 2B-IT, the hazard-features-active-despite-refusal flag fires on 34.8% of *benign* refusals. Not hazard prompts — benign textbook biology questions that the model refuses. Internal hazard feature activations are present. The model is refusing, but the refusal isn't calibrated to hazard content specifically. That's a signal that the safety mechanism is triggering broadly, not selectively.

**Finding Two: Format-gated safety and the 80-token problem.** Gemma 4 E2B's entire RLHF safety circuit is gated on chat-template formatting. Without the correct `<start_of_turn>` tokens — without the specific formatting that Gemma's RLHF training expects — the same 75 prompts produce zero genuine refusals. With correct formatting, it refuses 65 of 75.

And here's the part that should concern deployers: at an 80-token generation cap, *both* Gemma 4 and Gemma 2 refuse zero percent across all tiers. The safety articulation requires token budget. Constrained deployment contexts — mobile apps, embedded tools, latency-capped APIs — may be systematically bypassing the safety circuits that lab-bench evaluations measure. Surface refusal rates don't transfer to production if production ships with different format or length constraints.

The activation layer is format-robust in a way a surface completion classifier is not. That's the whole point.

These two findings sketch a concrete four-cell typology for model safety posture — deep refusal, shallow refusal, over-refusal, and what we're calling non-suppressive safety, where the safety circuit activates but doesn't gate the output. BioRefusalAudit distinguishes all four at the activation layer. Surface evaluation alone cannot.

**Validity caveats, stated plainly:** the divergence metric uses within-sample calibration — no held-out calibration set — and the feature catalog is auto-tuned by Cohen's d, not hand-validated against Neuronpedia. We've done that validation, and the top catalog features encode generic technical-governance vocabulary, not bio-specific refusal circuitry. The flag-based findings are more robust than the D-value comparisons because they operate on raw per-feature activation magnitudes and don't depend on the calibration matrix. We say this in the paper. We mean it.

---

## [HL3 AND RESPONSIBLE RELEASE — ~45 seconds]

One more contribution I want to be direct about, because it matters structurally.

The code is under Hippocratic License 3.0. The tier-3 hazard-adjacent eval data is behind an HL3-gated attestation on Hugging Face. Tiers 1 and 2 — benign biology and governance questions — are CC-BY-4.0, fully open.

This is not a formality.

At this hackathon, Sandbrink, Crook, and Yassif from NTI each called for tiered managed-access frameworks for AI biosecurity tools. The NTI Bio paper from January 2026 is explicit: for tools designed to probe hazard-adjacent model behavior, access criteria are warranted.

BioRefusalAudit implements two layers of that framework. The measurement layer is the divergence score — it quantifies how safe a refusal is, which is what makes a managed-access decision principled rather than arbitrary. The enforcement layer is HL3. Standard permissive licenses — MIT, Apache — allow any use, including weaponization. HL3 doesn't. It binds downstream users to enforceable human rights obligations. Violation terminates the license.

For a tool explicitly designed to probe hazard-adjacent model behavior, that distinction is load-bearing. The tiered data release implements the Biosecurity Data Level framework from Bloomfield, Black, Crook et al. in *Science* 2026 — directly, as a working instantiation.

---

## [WHAT COMES NEXT — ~30 seconds]

The main limit we're honest about is the feature catalog. Auto-tuning by Cohen's d identifies statistically discriminative features — it doesn't guarantee bio-specific refusal circuitry. What we need next:

Domain-specific SAE fine-tuning on biosecurity behavioral activation corpora. Genuine refusals versus shallow ones, with a contrastive training objective. A proof-of-concept local training run confirmed the infrastructure works on consumer hardware. The bottleneck is data: roughly ten thousand hazard-adjacent prompt-activation pairs from institutional CBRN red-team datasets.

The natural partners are AISI, CLTR, and national biosecurity labs who already hold this data. The training infrastructure is ready. The collaboration is the open item.

And we need refusal-depth reporting to become a standard companion to capability evaluations in RSPs and analogous governance frameworks. A model that refuses everything is not automatically safe. Refusal depth tells you whether that refusal is structurally earned.

---

## [CLOSE — ~20 seconds]

The code is on GitHub. The public eval set is CC-BY-4.0. The dashboard runs locally, and the full pipeline runs on a four-gigabyte GPU.

If you're working on AI biosecurity evaluation, managed access frameworks, or unlearning verification — the measurement layer now exists. Let's talk about what to do with it.

Thank you.

---

*Total estimated runtime: ~4.5–5 minutes at a measured pace. Adjust pacing in the findings section if needed — that's the densest part and benefits from deliberate delivery.*

*Slide cues if presenting with visuals:*
- *[OPENING]: Title slide — BioRefusalAudit logo + one-line description*
- *[PROBLEM]: "surface refusal ≠ structural safety" diagram — a two-column table showing surface behavior vs. internal activation*
- *[WHAT WE BUILT]: Pipeline diagram (prompt → residual stream → SAE → feature vec → judge → divergence score)*
- *[FINDINGS]: The four-cell typology table; the 34.8% benign over-refusal stat; the 80-token ablation result*
- *[HL3]: The three-tier data release table; the measurement-enforcement two-layer diagram*
- *[NEXT]: Roadmap slide — projection adapter / full SAE fine-tune / institutional data partners*
- *[CLOSE]: GitHub URL + QR code*
