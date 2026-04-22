# Why Hippocratic License 3.0

This is the longer version of the licensing argument summarized in [README §Safety and responsible use](../README.md#safety-and-responsible-use). Two choices need justification: HL3-FULL for the code, and HL3-Gated for the tier-3 data.

## HL3-FULL for the code

Quoting the README:

> This tool was built to strengthen biosecurity, not to weaken it. Standard permissive licenses (Apache, MIT) allow any use including weaponization. HL3 provides a legally enforceable human rights requirement that aligns with how every organization involved in this work — SecureBio, Fourth Eon Bio, Sentinel Bio, CLTR — actually thinks about responsible AI. We think it's the right choice for a biosecurity tool, and we think that licensing signal matters to the community this tool is for.

Three reasons, more precisely:

1. **Permissive licenses put no enforceable floor under downstream use.** Apache, MIT, and BSD are deliberately neutral on what the software is used for. That neutrality is appropriate for general-purpose tooling. It is not obviously appropriate for a tool whose explicit design goal is to probe hazard-adjacent model behavior. HL3-FULL binds downstream users to the UN Guiding Principles on Business and Human Rights, the Universal Declaration of Human Rights, the ILO Declaration on Fundamental Principles and Rights at Work, and the UN Declaration on the Rights of Indigenous Peoples. That is the floor this tool needs.
2. **The biosecurity and AI-safety community already operates on these norms.** The organizations that sponsor, fund, publish, and use this kind of work — SecureBio, Fourth Eon Bio, Sentinel Bio, CLTR, AISI-equivalents in multiple jurisdictions — already think about responsible use in human-rights terms. Choosing HL3 makes the license text match how the community actually reasons about these artifacts, instead of relying on informal norms that a permissive license cannot express.
3. **Licensing is a signal.** A biosecurity tool released under MIT sends one message. The same tool released under HL3-FULL sends a different message — about who the author thinks the audience is, what kind of reuse is and isn't welcome, and what the author will stand behind. For a tool specifically designed to probe hazard-adjacent behavior, the HL3 signal is the correct one.

## HL3-Gated for the tier-3 dataset

Tier-3 data — the `hazard_adjacent_category` prompts — is released under HL3.0 on a **separate** gated Hugging Face dataset (`SolshineCode/biorefusalaudit-gated`, to be published), distinct from the CC-BY-4.0 public tiers 1+2.

This is a direct application of the Biosecurity Data Level (BDL) framework proposed by Bloomfield, Black, Crook et al. (Science, 2026), which argues for tiered access control over biosecurity-relevant data as a structural precedent. The framework's core insight: not all biosecurity-relevant content should share the same release posture. Innocuous controls and legitimate-research framings can be fully open; category-level hazard probes cannot responsibly be released on the same terms. BioRefusalAudit adopts that tiering directly. Tiers 1 and 2 are CC-BY-4.0 because they are appropriate for a CC-BY-4.0 release; tier 3 is HL3-Gated because it is not.

The gate is an attestation of biosecurity/AI-safety research intent plus a no-public-reproduction agreement. See [SAFETY.md](../SAFETY.md#tier-3-access-attestation) for the exact attestation elements, and [LICENSE_HL3_DATASET.md](../LICENSE_HL3_DATASET.md) for the data-side license summary.

## This is a considered choice, not a default

The author has released other work under permissive open-source licenses elsewhere — this is not a policy applied uniformly across projects. HL3 was chosen here because **this tool is itself a biosecurity artifact**: its purpose is to probe and measure hazard-adjacent behavior in deployed language models. The license should reflect what the artifact is for and who it is built for, and permissive licensing does not reflect that. HL3-FULL does.

If you are evaluating this tool for a use case where HL3-FULL is incompatible with your organization's licensing policy, email the disclosure contact in [SAFETY.md](../SAFETY.md#disclosure-contact) — there may be a path forward, and feedback on friction is welcome.
