# Safety and Responsible Use

This document is the release and responsible-use policy for BioRefusalAudit. It applies to the code in this repository, the public eval set (tiers 1+2), and the gated eval set (tier 3).

## What this tool is for

BioRefusalAudit is a biosecurity and AI-safety research instrument. It is built for:

- **Biosecurity research** — measuring whether a deployed model's refusal behavior on bio-hazard-adjacent prompts is backed by internal representations, or is only a surface-layer output filter.
- **AI safety research** — extending the deception-detection methodology of prior work (arXiv:2509.20393) into a specific, safety-critical domain.
- **Red-team defense** — giving model deployers, auditors, and governance bodies a tool to distinguish structurally safer models from structurally riskier ones that happen to refuse fluently.
- **Governance and policy** — producing evidence for deployment decisions, eval-set design, and audit frameworks.

## What this tool is NOT for

BioRefusalAudit must not be used for:

- **Weaponization or uplift** — extracting, surfacing, or leveraging hazardous biological information for offensive purposes.
- **Surveillance** — profiling researchers, students, or practitioners based on their interactions with models.
- **Jailbreak development** — using the divergence metric as a targeting signal to construct prompts that elicit hazardous outputs.
- **Any activity prohibited by Hippocratic License 3.0** — including, but not limited to, violations of the UN Guiding Principles on Business and Human Rights, the UN Universal Declaration of Human Rights, or applicable international humanitarian law.

Use of this tool to harm people, facilitate bioweapon development, enable population-scale surveillance, or defeat legitimate safety training is a license violation and will be treated as such.

## Why HL3-FULL for the code

Quoting the README rationale verbatim:

> This tool was built to strengthen biosecurity, not to weaken it. Standard permissive licenses (Apache, MIT) allow any use including weaponization. HL3 provides a legally enforceable human rights requirement that aligns with how every organization involved in this work — SecureBio, Fourth Eon Bio, Sentinel Bio, CLTR — actually thinks about responsible AI. We think it's the right choice for a biosecurity tool, and we think that licensing signal matters to the community this tool is for.

Expanding on the standard [firstdonoharm.dev](https://firstdonoharm.dev) framing: HL3-FULL binds downstream users to the UN Guiding Principles on Business and Human Rights, the Universal Declaration of Human Rights, ILO core labor standards, and the UN Declaration on the Rights of Indigenous Peoples. For a tool whose explicit purpose is to probe hazard-adjacent model behavior, a permissive license would put no enforceable floor under downstream use. HL3-FULL puts one there.

See [docs/HL3_RATIONALE.md](docs/HL3_RATIONALE.md) for the longer argument, including the relationship to the Biosecurity Data Level (BDL) framework.

## Data release tiering

The eval set is released in tiers, matching the structure described in the README:

| Tier | Content | License | Location |
|---|---|---|---|
| 1 — `benign_bio` | Innocuous biology control prompts | CC-BY-4.0 | This repo; HF `SolshineCode/biorefusalaudit-public` |
| 2 — `dual_use_bio` | Legitimate research framings, hazard-adjacent | CC-BY-4.0 | This repo; HF `SolshineCode/biorefusalaudit-public` |
| 3 — `hazard_adjacent_category` | Category-level hazard probes | HL3-Gated | Will be published at HF `SolshineCode/biorefusalaudit-gated` (gated dataset not yet created) |

Tiers 1 and 2 are fully open. Tier 3 is access-controlled behind an attestation gate on a **separate** Hugging Face dataset repo, so tier-3 prompts never land in this repository or its history.

Model outputs generated from tier-3 prompts are redacted before any public artifact (reports, figures, paper tables) is produced. See `biorefusalaudit/reporting/redaction.py`.

## Tier-3 access attestation

Access to `SolshineCode/biorefusalaudit-gated` (once published) requires an attestation with the following elements:

1. **Identity and affiliation** — your name and the organization or independent research context you are working in.
2. **Purpose** — a one-paragraph statement that your intended use is biosecurity research, AI safety research, or defensive red-teaming.
3. **Non-reproduction** — agreement that specific tier-3 prompts will not be reproduced in public artifacts (papers, posts, repos, model outputs) without explicit author permission.
4. **Downstream obligation** — agreement that any onward sharing propagates the same attestation requirement to downstream recipients.
5. **HL3.0 acknowledgment** — agreement to the Hippocratic License 3.0 Full terms as applied to the dataset.

The attestation is binding. Breach is treated as a license violation under HL3.0 and handled accordingly.

## Disclosure contact

For security disclosures, responsible-use questions, suspected misuse, tier-3 access questions, or concerns about specific model findings:

- **Email:** caleb.deleeuw@gmail.com

This is the author's direct contact per the repository's `CLAUDE.md`. Please prefer email over GitHub issues for anything sensitive (hazard-relevant prompts, specific feature coordinates, suspected misuse); issues are public.

## Incident reporting

If you observe what you believe to be a misuse of BioRefusalAudit or its data:

1. **Do not reproduce the suspected misuse publicly.** Do not post tier-3 prompts, specific feature coordinates, or reconstructions of hazard-adjacent outputs.
2. **Email the disclosure contact above** with a description of what you observed, the URL or artifact in question if public, and any timestamps or identifiers.
3. Expect an acknowledgment within 7 days. For time-sensitive incidents, mark the subject line `[URGENT]`.

If you observe what you believe is a bug or false positive in the divergence metric itself, a GitHub issue is fine; for anything with hazard-adjacent content, email instead.

## HL3 license: anticipated questions

**Q: Does HL3 create friction that prevents legitimate biosecurity researchers from using your tool?**

Yes, there is friction — and that is intentional. HL3-FULL requires users to attest to the UN Guiding Principles on Business and Human Rights and prohibits use for harm. A biosecurity researcher at a national lab, university, or AI safety institute can attest to this in under five minutes. The friction is proportional: inconvenient enough to deter bad actors who want deniability, not so heavy that a research team with an IRB and an institutional email address would stop. The alternative — MIT or Apache — provides no friction at all, explicitly permits offensive use, and would allow the divergence metric to be used as a targeting tool to identify shallow-refusal models for adversarial exploitation. We regard that as an unacceptable risk for a tool whose explicit purpose is to probe hazard-adjacent model behavior.

**Q: What if an institution's legal team is unfamiliar with HL3?**

HL3 is backed by firstdonoharm.dev and has been adopted by several AI safety tools. It is SPDX-registered as `HL3-FULL`. The HL3 team provides guidance for institutional legal review at firstdonoharm.dev. For research use, the license is more permissive than GDPR compliance or standard IRB requirements — it adds human-rights conditions, not bureaucratic overhead. We are happy to provide a plain-English summary for any institution that requests one.

## Scope notes

This policy governs BioRefusalAudit. It does not govern the underlying models (Gemma 3 family, Llama 3.1 8B), the SAE releases (Gemma Scope 2, Llama Scope), the judge APIs (Gemini, Claude, OpenRouter), or any external data source. Use of those is governed by their respective terms.
