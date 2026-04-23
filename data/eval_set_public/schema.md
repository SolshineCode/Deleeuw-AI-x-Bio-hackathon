# Eval Set Public v1 — Schema

This document describes the JSONL schema used by `eval_set_public_v1.jsonl`.

Each line of the file is a single JSON object (one prompt per line) with the following fields.

## Fields

| Field              | Type   | Required | Description |
|--------------------|--------|----------|-------------|
| `id`               | string | yes      | Stable prompt identifier of the form `bio_NNN` (zero-padded to 3 digits). IDs are unique within the file and are not reused across versions. |
| `tier`             | enum   | yes      | One of `benign_bio`, `dual_use_bio`. `benign_bio` prompts are undergraduate-textbook biology questions with no dual-use content. `dual_use_bio` prompts are biosafety-policy and research-ethics questions at a governance or meta-debate level. |
| `framing`          | enum   | yes      | One of `direct`, `educational`, `roleplay`, `obfuscated`. `direct` is a plain question. `educational` frames the question as teaching/lecture preparation. `roleplay` asks the model to adopt a tutor, advisor, or analyst role. `obfuscated` uses a lexical constraint (e.g., "without using the word X") or an indirect phrasing. The `obfuscated` framing here is lexically-constrained paraphrase, not hazard-laundering. |
| `prompt`           | string | yes      | The full prompt text as it would be presented to a model under evaluation. |
| `rationale`        | string | yes      | One sentence explaining why the prompt fits its tier. Used for auditability during dataset review. |
| `expected_surface` | enum   | yes      | One of `comply`, `hedge`, `partial`. The expected high-level surface behavior for a safety-well-calibrated model. `comply` is used for benign textbook content. `hedge` is used for policy/meta-ethics questions where a calibrated model is expected to answer substantively while noting that the topic is contested or requires framing. `partial` is used for governance/regulatory questions where a calibrated model is expected to answer the policy question but stop short of operational detail. These are target surfaces, not pass/fail labels. |
| `category`         | string | yes      | A short snake_case topic label used for stratified reporting (e.g., `photosynthesis`, `durc_policy`). Labels are not a controlled vocabulary in v1; they are author-assigned and may be consolidated in later versions. |
| `sampled_from`     | string | yes      | Inspiration source — textbook, policy document, or public review the prompt is modeled on. This is a provenance pointer for auditability; no prompt is copied verbatim from any source. |

## Content guardrails (all prompts)

Every prompt in this dataset satisfies the following constraints:

1. No specific pathogen names paired with enhancement techniques.
2. No synthesis protocols or routes of any kind.
3. No specific agent selection strategies for any malicious application.
4. Each prompt is framed as something a curious biology student or policy analyst would ask in a public lecture or graduate seminar.
5. No prompt contains actionable uplift toward harm.

All content is modeled on publicly available educational sources (Alberts *Molecular Biology of the Cell*, Stryer *Biochemistry*, Campbell *Biology*, Janeway's *Immunobiology*, Goodman & Gilman's, Guyton & Hall *Physiology*) or public biosafety/biosecurity policy documents (WHO *Laboratory Biosafety Manual* 4th edition, CDC/NIH *BMBL* 6th edition, NIH DURC policy, 42 CFR Part 73 and 9 CFR Part 121, NIH Guidelines for Research Involving Recombinant or Synthetic Nucleic Acid Molecules, NSABB reports, NASEM *Gene Drives on the Horizon*, OSTP and WHO policy documents, UN Biological Weapons Convention text).

## License

The dataset is released under **CC-BY-4.0**.
