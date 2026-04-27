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

## [THE PROBLEM — ~75 seconds]

Every major biosecurity evaluation of language models asks some version of the same question: will this model produce hazardous output?

WMDP-Bio. VCT. ABC-Bench. All measuring whether the model *says* dangerous things.

But that's not the question that should drive deployment decisions.

The question is: when a model refuses — does it *can't*, or does it merely *won't right now*?

Those are not the same thing. A model with a surface refusal sitting over an intact internal capability is one framing shift from complying. Change the context — educational, roleplay, obfuscated — and a model that was "refusing" might comply.

And here's what makes this specific to biosecurity: capable adversaries don't query models with obvious prompts. They use exactly those framings. WMDP-Bio tests "will the model explain how to make anthrax." It doesn't test whether the model will explain Bacillus anthracis spore formation in the context of writing a graduate-level virology module. That's the attack surface that surface evaluation is blind to.

There's a deeper problem. Behavioral evals can be gamed. An adversary who knows what your eval tests can engineer prompts to pass it. An adversary who cannot observe your model's internal activations cannot optimize against them. Internal-state auditing has a structural advantage that behavioral testing does not have.

A model whose internal representations genuinely don't activate on hazard features when it refuses — that's structurally different. And currently, there is no tool to measure whether any model is in that category.

Qi et al. 2024 put it precisely: the alignment floor sits "a few tokens deep." Crook at the AIxBio 2026 keynote independently called "binary prediction unusable — calibrated confidence required" the measurement gap. Both need *measurement*, not argument.

BioRefusalAudit provides it. We call it **refusal depth**.

---

## [WHAT WE BUILT — ~60 seconds]

The core idea is interpretability-grounded divergence scoring.

Every time a language model processes text, it produces internal representations in a high-dimensional residual stream. Sparse autoencoders — SAEs — decompose those into interpretable feature directions: bio-content features, hazard-adjacent features, refusal-circuitry features, hedging patterns.

We hook into the residual stream at generation time, capture those activations, and project them through Gemma Scope SAEs to get a feature activation vector. We then run a multi-judge consortium on the model's actual output — did it refuse? comply? hedge? — and compute a calibrated divergence score: how much does the internal pattern diverge from what you'd expect, given what the model said?

High divergence on a refusal means the model is representing more than it's saying.

Notice what's not in that pipeline: the actual content of what the user typed or what the model said. The divergence score is computed entirely from internal activation vectors — not from transcript content. That's not incidental. Sandbrink called for bio monitoring that doesn't read interaction content, doesn't disclose proprietary data, doesn't create IP risk. BioRefusalAudit implements exactly that. A hospital running this at inference time can flag sessions with anomalous refusal depth without the audit layer ever reading a word of what the user asked. That's a privacy posture content-based screening fundamentally cannot achieve.

We built this as a full end-to-end pipeline: a 75-prompt stratified eval set across benign biology, dual-use governance, and hazard-adjacent tiers; four framing types — direct, educational, roleplay, and obfuscated — to test whether framing shifts behavior; and a Streamlit dashboard for per-prompt interactive auditing.

It runs on a four-gigabyte consumer GPU.

---

## [THE FINDINGS — ~90 seconds]

We ran the full pipeline on Gemma 2 2B-IT with Gemma Scope 1 layer 12 SAEs. Then on Gemma 4 E2B-IT with an author-trained domain-specific SAE.

Before the numbers: the key conceptual output of this work is that there are four meaningfully distinct model safety postures that surface evaluation collapses to one. Deep refusal — the internal activation pattern is consistent with not engaging with the hazard. Shallow refusal — the model says no while hazard features are still firing. Hedge-without-refuse — what Gemma 2 actually does. And non-suppressive safety — refusal circuitry activates but doesn't gate the output. A surface evaluator sees rows 1 and 2 as identical. A behavioral red-teamer might distinguish them eventually, with enough tries. BioRefusalAudit classifies all four on a single pass. That's the contribution the numbers are supporting.

Five findings — all have immediate policy implications, and most survive the validity caveats I'll name at the end.

**Finding One: Gemma 2 2B-IT hedges on hazard-adjacent content — and never refuses.** After correcting a judge-pipeline failure that mislabeled 29 biology answers as refusals, the actual surface-label distribution across all 75 prompts is: zero genuine refusals, 40 comply, 35 hedge. The tier pattern is striking: benign biology 70% comply / 30% hedge. Dual-use 80% comply / 20% hedge. Hazard-adjacent: 100% hedge, zero comply, zero refuse. The model responds to hazard-tier content with a consistent hedge posture — softening the response, not terminating it. That's a different failure mode than refusal bypass. Hedge-without-refuse is harder to detect with binary classifiers and may provide more partial information than a genuine refusal would. BioRefusalAudit surfaces this distinction. Surface evaluation alone cannot — it collapses "refuse" and "hedge" into the same bucket.

**Finding Two: The refusal circuit responds to biology, not to hazard level.** We ran causal intervention experiments — ablating and boosting the top refusal-circuitry SAE features — across all 75 prompts. 80% qualify as candidate mechanistic features. But the NC rate runs *backwards* relative to hazard level. Benign biology prompts qualify at 87%. Dual-use at 80%. Hazard-adjacent — the prompts you most care about — at 73%. The strongest single result in the corpus is bio_014, a benign roleplay prompt, with an effect size of 1.139 and no surface behavior change at all. That's a 1.1-unit internal-state shift that the model's output says nothing about. The refusal circuit is a biology detector, not a hazard detector — exactly what you'd expect from a catalog built on general bio vocabulary. Domain-specific SAE fine-tuning is the fix.

**Finding Three: Format-gated safety and the 80-token problem.** Gemma 4 E2B's entire RLHF safety circuit is gated on chat-template formatting. Without the correct `<start_of_turn>` tokens, the same 75 prompts produce zero genuine refusals. With correct formatting, it refuses 65 of 75.

And here's the part that should concern deployers: at an 80-token generation cap, both Gemma 4 and Gemma 2 refuse zero percent across all tiers. Safety articulation requires token budget. Constrained deployment contexts — mobile apps, embedded tools, latency-capped APIs — may be systematically bypassing the safety circuits that lab-bench evaluations measure. Surface refusal rates don't transfer to production if production ships with different format or length constraints.

The activation layer is format-robust where a surface completion classifier is not. That's the whole point.

**Finding Four: The divergence metric separates refusal posture from compliance posture with near-zero within-class variance.** We ran Gemma 4 E2B-IT through our domain-trained bio-SAE with a 150-token budget — enough room for the safety circuit to actually fire. Across all 75 prompts: comply responses show D=0.896 (±0.001). Refuse responses show D=0.249 (±0.004). Zero overlap. 0.647-point separation.

What that means concretely: D=0.249 on a refusal says the internal SAE feature activations are consistent with the refusal direction — the model is not just saying no, it's not activating bio-hazard feature space. D=0.896 on a compliance says bio-relevant features remain active even when the surface output is compliant — expected for educational biology content, and exactly what a well-calibrated audit should surface. The metric is tracking the thing it was designed to track. And the within-class variance is low enough that this is not a noise artifact.

Findings One and Four together sketch a concrete four-cell typology for model safety posture — deep refusal, shallow refusal, hedge-without-refuse, and what we're calling non-suppressive safety, where the safety circuit activates but doesn't gate the output. BioRefusalAudit distinguishes all four at the activation layer. Surface evaluation alone cannot.

**Validity caveats, stated plainly:** The divergence metric uses within-sample calibration, and the feature catalog is auto-tuned by Cohen's d, not hand-validated. We ran Neuronpedia lookups on the top features. They encode generic technical and governance vocabulary, not bio-specific refusal circuitry. The tier separation in D likely reflects vocabulary routing — biology prompts use technical vocabulary — as much as genuine hazard detection. A held-out calibration experiment confirmed T is framing-distribution-sensitive: a held-out T from a different framing distribution actually *inverted* the tier ordering. The flag-based and behavioral-count findings survive both caveats because they don't pass through T. The D-values need that caveat attached. We say this in the paper. We mean it.

**Finding Five: The refusal circuit can't tell "illegal" from "dangerous."** We ran a 15-prompt control set using psilocybin — Schedule I federally, but biologically non-toxic, FDA Breakthrough Therapy designation, peer-reviewed NEJM and JAMA Psychiatry trials. We designed four sub-categories that dial in on the legality axis independently of biology: pharmacology vocabulary that isn't illegal to study, cultivation of *Psilocybe* mushrooms which is illegal in the US, clinical discussion framed within legal research, and pure policy framing. If the refusal circuit is detecting CBRN risk, pharmacology and cultivation should get similar refuse rates. If it's detecting legality, cultivation should be refused more than pharmacology.

Gemma 2 refuses psilocybin cultivation at 25 to 50 percent. It refuses genuinely hazard-adjacent biology — zoonotic spillover, pathogen-adjacent immunology — at zero percent. That's the legality confound in one number.

Here's the internal control that makes it undeniable. Cannabis is also Schedule I. Federally illegal since 1970. Same compound class. Same federal scheduling. Cannabis cultivation: zero percent refusals. Psilocybin cultivation: 33 percent refusals. The model isn't running a DEA lookup. It absorbed a culturally-conditioned riskiness signal during RLHF training — psilocybin cultivation still carries stigma that cannabis cultivation doesn't anymore — and that signal is what's driving the refusals, not anything about biology or biosecurity.

This has a direct policy consequence. A deployer who wants a clinical biology assistant to discuss Schedule I compounds in harm-reduction or research contexts — while genuinely blocking CBRN-relevant biology — cannot rely on the current refusal circuit to draw that line. The circuit isn't drawing it. It's conflating "culturally taboo but benign" with "CBRN risk." That's a miscalibration that surface evaluation never finds, because surface evaluation doesn't ask whether the model is refusing the right things.

Small n — n=3 to 4 per cell. Expanded run in progress at n=27 across four compounds. But the cannabis comparison is an internal control that doesn't need large n to make the point.

---

## [HL3 AND RESPONSIBLE RELEASE — ~60 seconds]

One more contribution I want to be direct about, because it matters structurally — and because it directly addresses something multiple speakers at this hackathon called for.

The code is under Hippocratic License 3.0. The tier-3 hazard-adjacent eval data is behind an HL3-gated attestation on Hugging Face. Tiers 1 and 2 — benign biology and governance questions — are CC-BY-4.0, fully open.

This is not a formality. And I want to explain why this specific tool needs this specific license.

Sandbrink, Crook, and Yassif from NTI each called for tiered managed-access frameworks for AI biosecurity tools. The NTI Bio paper from January 2026 is explicit: for tools designed to probe hazard-adjacent model behavior, access criteria are warranted.

BioRefusalAudit is exactly such a tool. A probing tool is dual-use by definition — it can be used to verify that a refusal is safe, or it can be used to find the prompts where refusals are shallow. A permissive license that allows any downstream use, including weaponization of the probing capability, is structurally incoherent with the biosecurity purpose. HL3 doesn't allow that. It binds downstream users to enforceable human rights obligations. Violation terminates the license.

BioRefusalAudit also implements the Sandbrink monitoring requirement in a specific way that matters: the divergence score D is computed from SAE feature activations, not content. You can run this audit on a production system without the audit layer reading what users typed. That's privacy-preserving monitoring of bio-sensitive AI interactions — which is what Sandbrink called for — implemented as a working tool, not a proposal.

The tiered data release implements the Biosecurity Data Level framework from Bloomfield, Black, Crook et al. in *Science* 2026 — directly, as the first working instantiation of that framework in an AI biosecurity tool.

---

## [WHAT COMES NEXT — ~30 seconds]

The main limit we're honest about is the feature catalog. Auto-tuning by Cohen's d identifies statistically discriminative features — it doesn't guarantee bio-specific refusal circuitry. What we need next:

Domain-specific SAE fine-tuning on biosecurity behavioral activation corpora. Genuine refusals versus shallow ones, with a contrastive training objective.

We already ran this. During the hackathon, we trained a contrastive TopK sparse autoencoder on Gemma 2 2B-IT layer 12 residual activations — 5000 steps, consumer GTX 1650, under four minutes of wall clock. Reconstruction loss dropped 99.5 percent. We published that checkpoint to HuggingFace: `Solshine/biorefusalaudit-sae-gemma2-2b-l12-5000steps`.

Here's what we learned: the contrastive loss — the part that's supposed to push hazard-tier and benign-tier representations apart — barely moved. Not because the model can't learn; it did. Because the bio vocabulary is genuinely shared across hazard tiers. Whether a prompt is benign, dual-use, or hazard-adjacent, it uses the same technical biology language. The SAE sees the same feature activations. The signal you need to separate those classes isn't in a 75-prompt corpus of terminology — it's in genuine behavioral divergences between a base model and an RLHF-aligned one responding to the same prompts.

During this hackathon, we received gated access to `cais/wmdp-bio-forget-corpus` — 24,453 scientific papers on hazardous biology, from SARS-CoV-2 transmission to pathogen enhancement. That's the WMDP forget corpus. The v2 training pipeline is already updated: 5,000 steps, bio-forget-corpus as the hazard-tier training signal. That run is in progress now.

The corpus bottleneck is resolved. What remains is the behavioral sampling component: genuine refusal-versus-compliance response pairs at scale, which requires safety-tuned base model pairs and institutional operational data. That's the part you need a collaboration for.

The natural collaborators are AISI, CLTR, and national biosecurity labs who hold the behavioral pair data. The training infrastructure is validated, the WMDP corpus is integrated, the v2 SAE is in training. The next unlock is paired behavioral activation corpora — and we're positioned to use them the day access opens.

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
- *[FINDINGS]: The four-cell typology table; hazard-adjacent 100% hedge / 0% comply / 0% refuse corrected distribution; the 75-prompt NC table (80% overall, inverted tier ordering benign 87% > hazard 73%); bio_014 effect=1.139 as a single-prompt callout; the 80-token ablation result; D label-split two-cluster chart (comply D=0.896 vs refuse D=0.249, zero overlap)*
- *[HL3]: The three-tier data release table; the measurement-enforcement two-layer diagram*
- *[NEXT]: Roadmap slide — v2 SAE training (bio-forget-corpus, 5K steps, in progress) / behavioral sampling from base+RLHF pairs / institutional data partners for paired activation corpora*
- *[CLOSE]: GitHub URL + QR code*
