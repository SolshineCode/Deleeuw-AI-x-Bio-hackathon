# Strategic Intelligence Report: AIxBio Hackathon 2026
## Judges, Organizers, Sponsors — Research Profiles & Positioning Guide

**For:** Caleb DeLeeuw (BioRefusalAudit submission)
**Purpose:** Make the submission irresistible to each specific evaluator by speaking to their exact research gaps, citing their own work, and framing our contribution as the thing they have been saying is missing.

---

## How to Use This Document

Each section profiles one person: what they have published, what open problems they have explicitly named, what language they use, and exactly how BioRefusalAudit plugs into their agenda. The final section gives tactical networking and outreach scripts.

Read this before writing the writeup abstract, the dashboard copy, the Discord outreach, and the Phase 5 mentor messages. Every word facing these people should land in their frame, not ours.

---

## 1. Jasper Götting — PRIMARY JUDGE (SecureBio Head of Research, AI)

### Who he is

PhD virologist (Hannover Medical School, sequencing and transmission of DNA viruses in transplant patients). Moved into biosecurity research at Convergent Research (far-UVC roadmapping, clinical metagenomic sequencing), then joined SecureBio's AI and Biotechnology Risks team. Now Head of Research on the AI side. Primary author on the Virology Capabilities Test (VCT). Foresight Institute Fellow 2025. Verified email at securebio.org.

### His specific published work we must cite

**Virology Capabilities Test (VCT)** (Götting, Medeiros, Sanders, Li, Phan, Elabd, Justen, Hendrycks, Donoughe — arXiv April 2025, virologytest.ai). 322 multimodal questions co-developed with 57 expert virologists covering fundamental, tacit, and visual knowledge. o3 reaches 43.8% accuracy, outperforming 94% of expert virologists on questions within their sub-areas of expertise. Expert virologists with internet access average 22.1% on questions in their own specialty. Cited in Claude 3.7 Sonnet, Claude 4, GPT-4.5, o3-mini, o4-mini, and Gemini 2.5 Pro model cards. Referenced in US House Energy and Commerce Committee hearing on biosecurity and AI.

VCT's stated limitation (from the paper and SecureBio's 2025 retrospective): "It remains an open question how model performance on benchmarks translates to changes in the real-world risk landscape; addressing this uncertainty is a key focus of our 2026 efforts."

VCT is also explicitly framed as measuring what models *can* do, not whether their safety behaviors are robust. The paper proposes VCT as a proxy for capability, not a measure of safety training depth.

**SecureBio 2025 retrospective (Feb 2026):** Explicitly states four 2026 priorities:
1. "Mitigation strategies that measurably reduce risk in deployed systems"
2. "How model performance on benchmarks translates to changes in the real-world risk landscape"
3. Jailbreaking mitigations: "helping to characterize how safety systems fail under pressure and what kinds of mitigations appear most promising"
4. "If you're working on adjacent problems (especially evaluation standards, mitigation tools, or safety audit readiness), we'd love to compare notes and collaborate"

### The exact gap BioRefusalAudit fills for him

Götting's team has built the best capability evals in the field (VCT, WCB, MBCT, HPCT, ABC-Bench, ABLE). They explicitly acknowledge the unanswered question: does benchmark performance tell us about real-world risk? BioRefusalAudit provides a complementary signal: **when a model refuses on VCT-style queries, is the refusal robust at the internal computation level, or is the capability intact and the refusal superficial?**

Frame it to him as: "VCT tells you how capable the model is. BioRefusalAudit tells you how *safe* the refusals of a capable model actually are."

Also directly relevant: his background in virology sequencing means he understands the concept of assaying internal state rather than just observing surface phenotype. DNA viruses in transplant patients — he spent his PhD measuring what's really happening internally, not just what the surface behavior looks like. That's exactly what we're doing with SAE activations.

### Exact language to use when referencing his work

- "Building on the VCT's finding that o3 outperforms 94% of expert virologists, a natural next question is: when frontier models refuse VCT-style queries, how robust are those refusals internally? BioRefusalAudit provides the interpretability layer that answers this."
- "SecureBio's 2025 retrospective identified 'how model performance on benchmarks translates to real-world risk' as a key 2026 focus. Refusal-depth divergence is exactly this bridge: it operationalizes the distinction between surface capability and robustness of safety training."
- "Our unlearning shallowness metric (S4) directly addresses SecureBio's mitigation work: pretraining data filtering may suppress outputs without ablating capabilities. BioRefusalAudit detects this distinction."

### Discord outreach script (Phase 5)

> "Hi Jasper — I'm building a tool for Track 3 that I'd love your take on. The basic idea: VCT and your other evals measure what a model can do. We're measuring whether a model's refusals on bio-adjacent queries are robust at the internal-representation level, or just surface-layer output suppression. The differentiator is Gemma Scope 2 SAE activations + a divergence score between surface behavior and internal feature patterns. It's directly aimed at your 2026 open question about benchmark-to-real-world translation. Would you be willing to glance at our feature taxonomy and eval set schema and tell me if we're measuring the right things? Even 10 minutes of feedback would be huge."

---

## 2. Coleman Breen — JUDGE (SecureBio Senior AI Policy Researcher)

### Who he is

PhD candidate in Statistics at UW-Madison, specializing in statistical evaluation of genetic sequence ML models. Previously: AI resident at Google[x] (agricultural AI moonshot), Research Scientist at SecureBio building biosecurity-relevant model evaluations. Now Senior AI Policy Researcher at SecureBio. Johns Hopkins Center for Health Security Emerging Leaders in Biosecurity Fellow. Background straddles statistics, genomics, and AI.

### His specific relevant positions and work

Coleman's SecureBio bio says he "worked on the AI team at SecureBio as a research scientist to help develop biosecurity-relevant model evaluations." His transition from bench-level genomics (statistics for genetic sequence ML) to policy research is the key frame. He's a translator between technical evaluation methodology and policy.

He contributed to SecureBio's work on: pretraining data filtering topic lists and training datasets for frontier labs; AI-bio jailbreaking mitigations. On LinkedIn he shared SecureBio's work on the biological data governance framework (Biosecurity Data Levels, Science 2026 paper with Bloomfield, Black, Crook et al.).

### The exact gap BioRefusalAudit fills for him

Coleman lives at the intersection of: statistical rigor in ML evaluation + biosecurity policy. He will respond to two things: (1) methodological rigor in how we define and measure refusal depth, and (2) policy-actionable outputs. The policy brief (S8) is written for him as much as for Cassidy Nelson. Our divergence metric is a *statistically grounded* operational definition of refusal robustness — something he can point to in a model card framework.

The framing that lands hardest: "Responsible Scaling Policies currently require labs to report capability evals (WMDP, VCT). They should also require reporting interpretability-grounded refusal-depth metrics. Here's what that metric looks like."

### Exact language to use

- "Existing RSP frameworks require capability-based thresholds. BioRefusalAudit proposes a complementary interpretability-based threshold: refusal-depth divergence below a calibrated bound. This is the kind of metric that can be integrated into pre-deployment evaluation protocols."
- "Our statistical calibration procedure (Section 6.3) follows best practices from ML evaluation methodology: held-out positive controls, threshold calibration, cross-model validation. The result is a reproducible, auditable metric."
- "Coleman's work on jailbreaking mitigations at SecureBio maps directly to our S3 stretch: we show that divergence score predicts jailbreak success rate, providing a cheap proxy for red-team prioritization without requiring exhaustive adversarial search."

---

## 3. Jason Hoelscher-Obermaier — JUDGE AND ORGANIZER (Apart Research Director of Research)

### Who he is

PhD in experimental quantum physics (likely University of Vienna). 9+ years AI research engineering. Co-director of Apart Research, runs Apart Lab, organizes Apart Sprints. Research interests: safety evaluations of large language models, AI interpretability, alignment. Work spans: knowledge graphs, text summarization, NLP benchmarking, multi-agent risk taxonomies, interpretability probes for adversarial attacks, work for ARC. Published on multi-agent AI risk and evaluation methodology. Deep community involvement: Vienna AI alignment group, European Network for AI Safety. Mentored multiple Apart fellows. Self-describes as "always up for AI safety research collaborations."

His GitHub shows: apartresearch/specificityplus (ACL paper "Detecting Edit Failures in LLMs: An Improved Specificity Benchmark"). His LessWrong posts include mentoring work on interpretability of adversarial attacks using linear probes on classifier activations.

### What Apart Lab prioritizes in submissions

Based on the track record of Apart-to-publication projects: DarkBench (benchmark + eval contribution, ICLR Oral Spotlight), Women-in-AI-Safety mech-interp red-teaming winner, deception detection hackathon sandbagging work, Judge-using-SAE-Features. The pattern: technically grounded, interpretability-adjacent, clear safety motivation, reusable artifact (benchmark or tool), paper-shaped writeup.

Jason specifically looks for: (1) Is this evaluation methodology defensible? (2) Does it produce a reusable artifact? (3) Is there a clear follow-on research trajectory? (4) Is the person capable of publication-quality work?

### The exact gap BioRefusalAudit fills for him

Jason will evaluate this through the Apart Lab Studio lens: is this a hackathon project he can help the author develop into a published paper? The answer is emphatically yes if we deliver: (1) a novel metric with clear formalization, (2) cross-model results that constitute a finding, (3) a reusable eval set, (4) a writeup that is already 80% of a publishable paper.

Also: his quantum physics background means he genuinely appreciates calibration procedures and clean measurement methodology. The divergence score's formal definition and calibration procedure will land well with him.

### Exact language to use in the writeup and Discord

- "This project was designed as a continuation of Apart's deception detection research lineage, bringing the same SAE-based interpretability methods to the bio-safety domain where surface evaluation alone is insufficient."
- Lead the acknowledgments section with Apart Research and name Jason specifically if he provides any Phase 5 feedback.
- Frame the writeup structure explicitly: "Section 5 (Results) presents findings suitable for the NeurIPS Mechanistic Interpretability Workshop or ICLR Building Trust Workshop." Apart Lab Studio judges want to know you're thinking about the paper, not just the hackathon.

### Discord intro

> "Hi Jason — submitting to Track 3 with a project that sits exactly at the intersection of your mech-interp interpretability work and biosecurity evals. The core: cross-model SAE-based measurement of whether bio refusals are computationally deep or surface-layer. Clean metric formalization, multi-model comparison (Gemma 3 270M-12B + Llama 3.1 8B), reusable eval set, full Apart-style writeup. Already scoped for NeurIPS Mech-Interp Workshop follow-on. Happy to share the draft and methodology notes if useful."

---

## 4. Kevin Esvelt — KEYNOTE SPEAKER (MIT Media Lab, Sculpting Evolution Group)

### Who he is

Associate professor MIT Media Lab. Invented CRISPR-based gene drive. Co-founder of SecureBio (stepped back Sep 2024 to MIT). Co-author of multiple papers on information hazards, synthetic biology biosecurity, SecureDNA cryptographic screening. Author of "Delay, Detect, Defend: Preparing for a Future in Which Thousands Can Release New Pandemics" (Geneva Centre for Security Policy, 2022). Published in PLOS Pathogens on inoculating science against information hazards (2018). 2026 book chapter on information hazards in biotechnology and AI (RAND).

He coined the Delay/Detect/Defend framework that organizes the entire biosecurity community's intervention logic.

### His specific language and positioning

Esvelt's Delay/Detect/Defend framework explicitly calls for a "Detect" layer. In the original GCSP paper: delay capability proliferation, detect anomalous events early, defend populations. His information hazard work centers on: true information whose release enables harm. His work at MIT focuses on making sure beneficial research doesn't inadvertently disseminate dangerous information.

He recently co-authored work on information hazards with RAND (Jan 2026) discussing the role of publishers, funders, IRBs, and regulators in managing dual-use information.

### The exact positioning for BioRefusalAudit

We are squarely a "Detect" contribution in his framework, but at the AI-system layer rather than the pathogen/surveillance layer. We are detecting shallow refusals — AI safety behaviors that appear safe but conceal intact dangerous capabilities. This is an LLM information hazard detection problem: the LLM appears to not be hazardous (surface refusal) but internally it may be.

The framing: "BioRefusalAudit is a Detect intervention at the AI-system layer: it detects when a model's apparent safety behavior conceals an intact capability to disseminate hazardous information."

Also connect to his information hazard work: we are helping laboratories and biosecurity institutions detect AI systems that present information hazard risk through shallow refusals, which is directly in his lineage of practical biosecurity tooling.

### In the writeup (Introduction, Section 2.4)

- "Esvelt's Delay/Detect/Defend framework has organized biosecurity interventions at the pathogen and infrastructure layer. We extend the Detect component to the AI-system layer: BioRefusalAudit detects when a model's surface refusal behavior misrepresents its internal capability, flagging the specific prompts where a deployed LLM poses information hazard risk regardless of its apparent safety behavior."
- Cite his GCSP paper and the PLOS Pathogens information hazards piece.
- Do NOT try to shake hands with Esvelt at the hackathon to pitch. He is the keynote; let the work speak. If he asks questions in Q&A or office hours, have a 2-sentence version ready: "We're the Detect layer for LLMs. When a model refuses a VCT-style query, we tell you if the refusal is computationally deep or just output suppression."

---

## 5. Jonas Sandbrink — SPEAKER (Sentinel Bio Entrepreneur-in-Residence, formerly UK AISI)

### Who he is

DPhil Oxford (Nuffield Department of Medicine), biosecurity and AI. Previously: founded and led UK AISI's chemical and biological evaluations team. Currently: Entrepreneur-in-Residence at Sentinel Bio, building verified researcher credentials for accessing sensitive biological data and tools. His LLM-vs-BDT paper (arXiv 2306.13952) is the foundational framing for distinguishing large language model risk from biological design tool risk. Advised UK Cabinet Office, Google DeepMind, Nuclear Threat Initiative on biosecurity.

### His specific work to cite

Sandbrink's LLM/BDT paper argues: LLMs lower barriers for non-experts by providing dual-use information; BDTs expand capabilities of sophisticated actors. LLMs "will in particular lower barriers to biological misuse" through information access, not capability design. This is our exact domain.

His current work at Sentinel Bio on verified researcher credentials is directly relevant to our HL3.0 gated dataset: both are access-control frameworks for sensitive bio-relevant tools and data. He will understand and appreciate the HL3.0 tiering immediately.

### How BioRefusalAudit lands for him

Sandbrink's LLM/BDT distinction says LLMs are dangerous because they lower information barriers. BioRefusalAudit asks: for LLMs that appear to have been safety-trained to not lower these barriers (i.e., they refuse), how deep is that training? Are the information barriers actually raised, or just superficially papered over?

This is exactly the question his framework raises but doesn't answer. His paper motivates the existence of safety training; we evaluate whether that safety training is real.

The Sentinel Bio credential work: our tool could in principle be a component of a credential-based access system. An institution wanting access to sensitive bio tools could run BioRefusalAudit on their candidate LLM advisor to verify that refusals are robust before granting access.

### Exact language

- "Sandbrink's foundational LLM/BDT framework motivates safety training as a barrier to information access. BioRefusalAudit provides the first tool for empirically verifying whether that barrier is structurally robust or superficial — a question his framework implicitly raises but evaluation tooling has not addressed."
- "Our approach to dataset tiering under Hippocratic License 3.0 is informed by Sandbrink's current work on verified researcher credentials at Sentinel Bio: both are access-control frameworks that enable research while managing dual-use risk."

---

## 6. Cassidy Nelson — SPEAKER (CLTR Director of Biosecurity Policy)

### Who she is

Dual-trained physician-scientist. DPhil in biology from Oxford. MD from University of Queensland. MPH. Director of Biosecurity Policy at Centre for Long-Term Resilience. Advised WHO, Bipartisan Commission on Biodefense, RAND Europe. Research focus: early warning systems for Disease X / novel pathogens. Member of UK Engineering Biology Responsible Innovation Advisory Panel. Alumna of JHU Center for Health Security Emerging Leaders in Biosecurity.

### Her policy frame

Cassidy works at the policy-to-practice interface: she translates technical biosecurity findings into actionable policy recommendations. Her work on early warning and Disease X means she thinks about surveillance systems, notification protocols, and institutional preparedness. At CLTR, she focuses on long-term resilience — not just near-term threats but structural governance.

### How the policy brief (S8) lands for her

She will respond to: (1) a clear policy recommendation that is immediately actionable, (2) a governance hook to existing frameworks (RSPs, pre-deployment testing requirements), (3) language accessible to non-technical senior policymakers.

The policy brief should make this case: "We propose that frontier model developers add a refusal-depth metric to their pre-deployment biological capability assessments, alongside existing VCT/WMDP-style capability measures. This requires one additional interpretability-based eval that can be run with open-source tooling (BioRefusalAudit) at no marginal cost to labs already doing bio evals. The metric becomes the interpretability layer in their Responsible Scaling Policies."

Frame it as a structural intervention: not "test model X," but "require all frontier developers to report refusal-depth scores when they exceed biological capability thresholds."

### Exact language for the policy brief

- "We propose a new class of biosecurity pre-deployment requirement: interpretability-grounded refusal-depth disclosure. When a model reaches or exceeds the VCT capability threshold (Götting et al., 2025), developers should be required to report not only capability scores but refusal-depth divergence scores on a standardized bio-adjacent eval set. BioRefusalAudit provides the open-source tooling to do this."
- "This proposal aligns with CLTR's longstanding recommendation for structural, threat-agnostic biosecurity governance: rather than regulating individual model behaviors, we propose governance at the evaluation infrastructure layer."
- Cite her CLTR bio and her Disease X/pandemic preparedness policy work generally (no specific paper to cite but reference the CLTR framework).

---

## 7. Steph Guerra — SPEAKER (RAND AIxBio Portfolio Lead)

### Who she is

Molecular biologist. AIxBio portfolio at RAND Center on AI. Co-author on Nature Biotechnology "A call for built-in biosecurity safeguards for generative AI tools" (Wang, Zhang, Bedi, Guerra, et al. 2025). Previously affiliated with NIST (Affiliation 4 in the Nature Biotech paper).

### Her specific work

The Nature Biotechnology paper (2025, doi 10.1038/s41587-025-02650-8) calls for built-in biosecurity safeguards for generative AI tools. It argues that AI-driven tools could be misused to engineer pathogens, toxins, or destabilizing biomolecules, and that AI science agents could amplify risks by automating experimental design.

The paper's core argument: safeguards should be built into the tools themselves, not retrofitted. This is the "SafetyByDesign" position.

### How BioRefusalAudit maps to her work

BioRefusalAudit is precisely the kind of built-in safeguard evaluation infrastructure the Nature Biotech paper calls for. We are not asking whether a model is capable — we are auditing whether its built-in safety training is robust, which is the prerequisite for trusting any built-in safeguard.

### Exact language

- "Our approach directly implements the evaluation infrastructure called for by Guerra et al. (2025) in their Nature Biotechnology call for built-in biosecurity safeguards: rather than assuming that a model's refusal behavior reflects genuine internal safety, we provide the auditing tool to verify this assumption."
- Cite Wang et al. (2025) as the authoritative statement of why built-in safeguards need to be auditable, not just present.

---

## 8. Oliver Crook — KEYNOTE SPEAKER (Oxford MRC Fellow, Kavli Institute)

### Who he is

MRC Career Development Fellow, Kavli Institute for Nanoscience Discovery, Oxford. Todd-Bird Junior Research Fellow at New College. Develops computational and statistical methods at the intersection of Bayesian ML, structural biology, and biosecurity. Works on genomic language models, protein design methods, and interpretability of biological AI. Technology and Security Policy Fellow at RAND. RAND biosecurity report co-author.

### His specific published work

- Co-author on "Strengthening nucleic acid biosecurity screening against generative protein design tools" (Science, 2025, 390:82-87). This paper showed that AI-redesigned protein sequences evaded current screening tools, and they developed patches improving detection rates.
- Co-author on "Securing dual-use pathogen data of concern" (arXiv 2602.08061, 2026) introducing the Biosecurity Data Levels (BDL) framework — 5-tier classification of pathogen data by risk level, published with Oliver Crook, Bloomfield, Black, et al.
- Co-author on "Biological data governance in an age of AI" (Science, Feb 2026) — the follow-up to the BDL framework.
- Oxford Chemistry profile: "We work on genomic and protein language models, sequence design methods and interpretability approaches."

### The hook

Crook is an Oxford ML researcher who does interpretability of biological AI models. He literally works on interpretability of genomic language models. BioRefusalAudit uses interpretability methods (SAEs) to audit safety behavior of generalist LLMs in the bio domain.

More directly: his Science 2025 paper showed that protein design tools could evade screening. The mechanism of evasion is capability without detection. BioRefusalAudit addresses this from the LLM side: safety training without robustness is analogous to screening without sensitivity.

The BDL framework (5 tiers from BDL0 to BDL4) maps directly to our 3-tier eval set structure. We should cite the BDL framework when introducing our tiering and note the alignment.

### Exact language

- "Our three-tier eval set structure (benign_bio, dual_use_bio, hazard_adjacent_category) is informed by Bloomfield, Black, Crook et al.'s Biosecurity Data Level (BDL) framework (Science, 2026). We operate at the LLM-interaction layer of BDL2-3 risk, evaluating whether safety training provides robust barriers at these tiers."
- "Crook et al. (Science, 2025) demonstrated that AI-designed protein sequences can evade nucleic acid screening tools. Analogously, we show that LLM safety training can exhibit surface refusal while retaining internal bio-capable representations — a form of 'safety evasion' that current capability-only evals miss."
- Cite the Strengthening Nucleic Acid Screening paper prominently in the Related Work section.

---

## 9. James Black — SPEAKER (Fourth Eon Bio Founding Scientific Advisor, JHU Scholar)

### Who he is

MD (Oxford BA Medical Sciences, Imperial MBBS), PhD in Computational Cancer Genomics (UCL). Founding Scientific Advisor at Fourth Eon Bio and Founder of Firewall Bio. Contributing Scholar at JHU Center for Health Security focusing on technical assessments of AI-enabled biological risks. Expert in biosecurity, evolutionary biology, and genotype-phenotype relationships.

### His specific published work and positions

- Co-author on "Biological data governance in an age of AI" (Science, Feb 2026) — BDL framework with Bloomfield, Crook, Pannu, et al.
- Co-author on "Open-weight genome language model safeguards: Assessing robustness via..." (cited in the Securing Dual-Use Pathogen Data paper).
- JHU Center's response to NIST AI Executive Order RFI explicitly argues that "biosecurity evaluations should determine whether AI systems provide more access to dangerous information or capabilities than alternative public sources."
- Fourth Eon Bio co-direction: benchmarks for AI-generated sequences, evaluation of AI models for hazardous function prediction.

### How BioRefusalAudit maps to his work

James Black sits at the exact intersection of computational biology and biosecurity evaluation. His Fourth Eon Bio work is explicitly about benchmarks for AI model evaluation. BioRefusalAudit is the interpretability-grounded complement to Fourth Eon's sequence-level screening benchmarks.

His open-weight genome language model safeguards work (assessing robustness) is methodologically parallel to ours: we are both asking "how robust are the safeguards, really?" His answer is at the genomic sequence level; ours is at the LLM behavior level.

The JHU NIST RFI response argues that bio evals should measure whether AI provides more access than alternative sources. Our divergence score is a proxy for exactly this: a model with high divergence (appears to refuse but internal representations are activated on hazard features) is *more* dangerous than its surface behavior suggests.

### Exact language

- "Fourth Eon Bio's evaluation work focuses on whether AI-designed sequences can evade screening. Our approach is the LLM-facing complement: we evaluate whether AI safety training can be bypassed not by sequence redesign but by the shallow nature of the safety training itself."
- "This work directly implements the evaluation criterion articulated in the JHU Center for Health Security's NIST AI EO response: determining whether AI systems provide more access to dangerous information than alternative sources. A model with high refusal-depth divergence does exactly this — it appears safe while retaining internal hazard representations."
- Cite the "Open-weight genome language model safeguards: Assessing robustness" paper as a methodological parallel (same question, different layer of the stack).

---

## 10. Gene Olinger — SPEAKER (Galveston National Lab Director)

### Who he is

Director of the Galveston National Laboratory. Professor of Microbiology and Immunology at UTMB. John Sealy Distinguished University Chair in Tropical and Emerging Virology. BSL-4 lab director — he runs the highest biosafety level research in the US.

### His frame

Olinger operates at the extreme high-containment end of bio research. His frame is: what does the wet-lab biosecurity community need? He will evaluate hackathon projects from the perspective of "would a real-world biosecurity institution with serious containment capabilities actually use this?"

### How to speak to him

Olinger is not an AI interpretability person. He is a virologist and lab director. Do not lead with SAEs. Lead with: "A biosafety officer at your institution is evaluating which open-weight LLM to allow as a research assistant. BioRefusalAudit tells them not just whether the model refuses bio-adjacent queries, but whether those refusals are computationally deep. That's the deployment decision tool they need."

The dashboard is the artifact that lands with him. Demo should show: clear risk categories, clear refusal-depth scores, clear recommendation ("Model A: deep refusals, low divergence, lower deployment risk; Model B: shallow refusals, high divergence, jailbreak-susceptible").

Do not overclaim to him. He will know immediately if the bio framing is wrong. Let the security use case speak; don't try to impress him with the ML.

---

## 11. Organizational Agenda Alignment — Summary Matrix

| Org | Their explicit 2026 gap | How BioRefusalAudit fills it | Best citation hook |
|---|---|---|---|
| **SecureBio** | Benchmark-to-real-world translation; jailbreaking mitigation characterization | Refusal-depth bridges capability to real risk; divergence predicts jailbreak surface | VCT paper (Götting et al. 2025) |
| **Fourth Eon Bio** | AI model evaluation for hazardous function prediction; benchmark datasets | Interpretability layer under their sequence-level evals; LLM advisor safety audit | Science 2025 screening paper (Black + Crook) |
| **Sentinel Bio** | Screening gap closure; verified credentials for sensitive tool access | Refusal-depth as access-control criterion; HL3.0 dataset mirrors credential approach | Sandbrink arXiv 2306.13952 |
| **CLTR** | Policy-actionable governance mechanisms for AI-bio convergence | RSP-compatible refusal-depth reporting requirement | Policy brief (S8) |
| **Apart Research** | Benchmark-style hackathon-to-publication projects; interpretability + safety | Reusable eval set + novel metric + Apart deception detection lineage | DarkBench, Apollo Sandbagging work |
| **Cambridge Biosecurity Hub / CBAI** | DNA screening enhancement; LLM component of AI-assisted design workflows | Future extension to LLM advisors in DNA design pipelines | Crook BDL framework (Science 2026) |
| **BlueDot Impact** | Practitioner tools for underresourced institutions | Dashboard + policy brief are precisely this | McGurk RFP talk Apr 23 |
| **Measuring AI Progress** | Tracking how AI capability changes bio risk over time | Cross-model scaling story is a measurement contribution | Forecasting LLM-Enabled Biorisk (GovAI) |

---

## 12. Critical Insight: SecureBio's Explicit 2026 Invitation

This quote from the SecureBio 2025 retrospective (Feb 2026) is as close to an explicit invitation as you will get:

> "It remains an open question how model performance on benchmarks translates to changes in the real-world risk landscape; addressing this uncertainty is a key focus of our 2026 efforts."
>
> "If you're working on adjacent problems (especially evaluation standards, mitigation tools, or safety audit readiness), we'd love to compare notes and collaborate."

BioRefusalAudit directly addresses their open question. The divergence metric is the bridge between "model does well on VCT" and "is deploying this model actually unsafe." Every sentence you write to or about SecureBio should reference this gap and claim we're filling it.

This also means the Phase 5 Discord outreach to Jasper Götting or Coleman Breen should open with that quote and your response to it. Not sycophantic — direct. "You wrote that benchmark-to-real-world translation is your 2026 focus. Here is a metric that operationalizes exactly that bridge."

---

## 13. The Open-Weight Safeguard Robustness Thread

One of the strongest threads connecting multiple judges is the Science 2025 paper: Black, Crook et al. showed that nucleic acid screening failed because AI-designed sequences could evade homology-based detection. Their solution was patches that improved detection of synthetic homologs.

BioRefusalAudit is the same argument at the LLM layer: LLM safety training can fail because it produces surface refusal without deep capability removal. The solution is an interpretability-grounded audit tool.

The shared structure: **AI-designed evasion → superficial safety → interpretability-based detection → patches/mitigations**.

Use this explicitly in the writeup abstract and intro. The framing: "Just as Götting/Black/Crook demonstrated that AI-designed sequences can evade nucleic acid screening, we show that safety training can produce surface-level refusals over intact bio-capable internal representations. BioRefusalAudit is the detection layer for this form of evasion."

This thread connects: Götting (VCT → what evasion looks like at the output layer), Black (Science 2025 → sequence-level evasion), Crook (same paper + BDL framework), and your own Secret Agenda paper (SAE auto-labels don't detect strategic deception). You are the person who has read all these papers and synthesized the cross-layer argument. That is the distinctive intellectual contribution.

---

## 14. The Argument Judges Will Find Most Compelling

Based on reviewing all their work, the single argument most likely to resonate across all five judges and the key organizer:

**"Current bio safety evals measure whether a model will provide hazardous output. BioRefusalAudit measures whether a model that won't provide hazardous output *can't* — or merely *doesn't right now*. The distinction matters for every deployment decision, every unlearning verification, and every benchmark-to-real-world translation. We provide the first cross-model tool to operationalize this distinction."**

Every judge has a piece of this:
- Götting: benchmark-to-real-world translation is his explicit 2026 focus
- Breen: statistical rigor + policy-actionable output
- Jason H-O: novel metric + Apart Lab publication trajectory
- Black: same question as his Science 2025 screening robustness paper
- Esvelt: Detect layer for information hazards
- Sandbrink: his LLM paper motivates the need; we answer it
- Crook: interpretability of biological AI is literally his lab's focus
- Nelson: policy brief targeting her audience
- Guerra: built-in safeguard audit is her explicit call-to-action

No single prior work addresses all of these. You are the author who has read all of it.

---

## 15. Networking Tactical Plan

### Apr 23 (McGurk Talk, 12:00 PM PDT)
- Attend on Zoom. Have the Phase 5 questions ready (see PRD). Take notes in `logs/phase_5_mcgurk_notes.md`. Ask at minimum: (1) what is the typical first-grant size for a solo-PI hackathon project, (2) does Coefficient fund interpretability-grounded evaluation work specifically or only bio infrastructure.
- After the talk: post in the hackathon Discord that you attended and what your key takeaway was. This signals to organizers you are engaged.

### Apr 23 evening — Discord outreach
- Post project summary to #biosecurity-hackathon or equivalent channel. Mention: Track 3, Gemma Scope 2 as infrastructure, SecureBio's 2026 benchmark-to-risk translation gap as the problem, the divergence metric as the method. Ask for a collaborator (someone who can handle the dashboard or cross-model experiments in parallel). Ask for mentor feedback on the feature taxonomy.
- DM Jasper Götting with the script from Section 1. Do not DM all judges simultaneously — too much. Götting is the primary judge and he explicitly invited collaboration.

### Apr 24 (Hackathon Day 1)
- Attend any keynote Zooms. Ask one good question during Esvelt's keynote — specifically: "How would the Detect component of Delay/Detect/Defend apply to evaluating the robustness of LLM safety training, as opposed to pathogen detection?" This puts your project concept in his head before he sees your submission.

### Apr 25-26 (Hackathon Days 2-3)
- Submit early if possible (judges often give more attention to early clean submissions than rushed late ones).
- Post a Twitter/LinkedIn update on hackathon Day 2 with the headline finding from the cross-model run. Keep it direct, unpolished per your style. "Ran BioRefusalAudit across Gemma 3 (270M-4B) and Llama 3.1 8B. [Finding]. Dataset and code open-source. Paper in the hackathon submission."

### Post-hackathon
- If Jason H-O expresses interest in Apart Lab Studio, respond within 48 hours.
- If Götting or Breen responds in Discord, send them the full draft writeup promptly and ask for one specific feedback item — not a general review. Specific ask = more likely response.
- RFP submission: use the McGurk talk learnings. Submit by May 11 regardless of polish level. A submitted draft beats a polished non-submission.

---

## 16. Citations Cheatsheet for Maximum Judge Recognition

Each cite below is recognized by its associated judge; using it signals you are in their literature:

| Citation | Whose literature | Where to use |
|---|---|---|
| Götting et al. (VCT, arXiv 2025) | Jasper Götting | Introduction, Related Work, S4 unlearning |
| SecureBio 2025 Retrospective (Feb 2026) | Götting + Breen | Introduction (their open question) |
| Wang/Guerra et al. Nature Biotech (2025) | Steph Guerra | Introduction (built-in safeguards call-to-action) |
| Black/Crook et al. Science (2025, 390:82-87) | James Black, Oliver Crook | Related Work, framing AI evasion analogy |
| Bloomfield/Black/Crook et al. Science (Feb 2026, BDL) | Oliver Crook, James Black | Eval set tiering section |
| Sandbrink arXiv 2306.13952 (LLM/BDT) | Jonas Sandbrink | Introduction scope-setting |
| Esvelt GCSP 2022 (Delay/Detect/Defend) | Kevin Esvelt | Introduction framing |
| Esvelt PLOS Pathogens 2018 (information hazards) | Kevin Esvelt | Information hazard discussion |
| DeLeeuw et al. arXiv 2509.20393 (Secret Agenda) | You / apart deception lineage | Method (SAE negative result as design principle) |
| Lieberum et al. arXiv 2408.05147 (Gemma Scope) | Apart community, Gemma Scope users | Method |
| McDougall et al. 2025 (Gemma Scope 2, LessWrong) | Apart community | Method (headline infrastructure) |
| He et al. arXiv 2410.20526 (Llama Scope) | Apart community | Method (cross-architecture) |
| Li et al. (WMDP, 2024) | All | Eval set positioning |
| DarkBench (ICLR 2025 Oral) | Jason H-O / Apart lineage | Related Work (Apart track record) |
| Forecasting LLM-Enabled Biorisk (GovAI) | Measuring AI Progress sponsors | Track 2 cross-track relevance |

---

*End of Strategic Intelligence Report. Use this as a living reference throughout the build and during submission. Re-read Section 12 before writing the abstract. Re-read Sections 1 and 14 before any Discord outreach.*
