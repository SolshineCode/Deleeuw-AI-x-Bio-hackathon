# Citation check log (2026-10-22)

Method: for arXiv IDs, title, first author and date read from the arXiv API or the
`citation_*` meta tags on `arxiv.org/abs/<id>`. For non-arXiv sources, the publisher page or
a web search noted below. "OK" means the reference as printed in v1 (tex:1111-1195) matches
the source on title and first author.

## arXiv references in v1 and candidates for the new related-work section

| arXiv ID | Title at source | First author | Date | v1 status |
|---|---|---|---|---|
| 2209.10652 | Toy Models of Superposition | Nelson Elhage | 2022-09-21 | cited in v1 |
| 2307.02483 | Jailbroken: How Does LLM Safety Training Fail? | Wei, Alexander | 2023/07/05 | cited in v1 |
| 2307.15043 | Universal and Transferable Adversarial Attacks on Aligned Language Models | Andy Zou | 2023-07-27 | cited in v1 |
| 2308.01263 | XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models | Paul Röttger | 2023-08-02 | not cited in v1 (candidate) |
| 2309.08600 | Sparse Autoencoders Find Highly Interpretable Features in Language Models | Hoagy Cunningham | 2023-09-15 | cited in v1 |
| 2401.05566 | Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training | Evan Hubinger | 2024-01-10 | cited in v1 |
| 2402.04249 | HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal | Mantas Mazeika | 2024-02-06 | not cited in v1 (candidate) |
| 2402.10260 | A StrongREJECT for Empty Jailbreaks | Souly, Alexandra | 2024/02/15 | not cited in v1 (candidate) |
| 2403.03218 | The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning | Li, Nathaniel | 2024/03/05 | cited in v1 |
| 2403.19647 | Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models | Marks, Samuel | 2024/03/28 | cited in v1 |
| 2404.14219 | Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone | Abdin, Marah | 2024/04/22 | cited in v1 |
| 2405.20947 | OR-Bench: An Over-Refusal Benchmark for Large Language Models | Justin Cui | 2024-05-31 | not cited in v1 (candidate) |
| 2406.04093 | Scaling and evaluating sparse autoencoders | Gao, Leo | 2024/06/06 | cited in v1 |
| 2406.05946 | Safety Alignment Should Be Made More Than Just a Few Tokens Deep | Xiangyu Qi | 2024-06-10 | cited in v1 |
| 2406.11717 | Refusal in Language Models Is Mediated by a Single Direction | Andy Arditi | 2024-06-17 | cited in v1 |
| 2406.12935 | ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates | Jiang, Fengqing | 2024/06/17 | not cited in v1 (candidate) |
| 2406.14598 | SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal | Xie, Tinghao | 2024/06/20 | not cited in v1 (candidate) |
| 2406.18495 | WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs | Seungju Han | 2024-06-26 | not cited in v1 (candidate) |
| 2407.10362 | LAB-Bench: Measuring Capabilities of Language Models for Biology Research | Laurent, Jon M. | 2024/07/14 | cited in v1 |
| 2407.14435 | Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders | Senthooran Rajamanoharan | 2024-07-19 | cited in v1 |
| 2408.05147 | Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 | Tom Lieberum | 2024-08-09 | cited in v1 |
| 2411.11296 | Steering Language Model Refusal with Sparse Autoencoders | O'Brien, Kyle | 2024/11/18 | not cited in v1 (candidate) |
| 2504.16137 | Virology Capabilities Test (VCT): A Multimodal Virology Q&A Benchmark | Jasper Götting | 2025-04-21 | cited in v1 |
| 2509.09708 | Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal | Prakash, Nirmalendu | 2025/09/07 | not cited in v1 (candidate) |
| 2509.20393 | The Secret Agenda: LLMs Strategically Lie and Our Current Safety Tools Are Blind | DeLeeuw, Caleb | 2025/09/23 | cited in v1 |
| 2602.08061 | Securing Dual-Use Pathogen Data of Concern | Doni Bloomfield | 2026-02-08 | not cited in v1 (candidate) |
| 2607.14479 | BioTIER: A Refusal Benchmark for Targeted Biological Risk Mitigation | Marshall, Eleanor M. | 2026/07/16 | not cited in v1 (candidate) |

Notes on the cited arXiv entries:

- 2509.20393: v1 prints "DeLeeuw, C. (2025). The Secret Agenda: LLMs strategically lie
  undetected by current safety tools." Source title is "The Secret Agenda: LLMs
  Strategically Lie and Our Current Safety Tools Are Blind"; authors DeLeeuw, Chawla,
  Sharma, Dietze. **Wrong title, missing co-authors.**
- 2504.16137 (VCT): reference OK; body description (tex:181-182) is wrong (capability
  benchmark for virology troubleshooting, not a hazardous-assistance measure).
- 2309.08600 (Cunningham) and Bricken et al. 2023 (Transformer Circuits, no arXiv): listed
  but never cited in the body.
- All other cited arXiv entries: title and first author match.

## Non-arXiv references

| Reference (v1) | Check | Result |
|---|---|---|
| Carter & Butchello 2026, NTI | nti.org report PDF cover (search result, 2026-10-22) | OK: Sarah R. Carter and Greg Butchello, NTI \| bio, January 2026. The repository README's "Yassif & Carter" is wrong. |
| Liu et al. 2025, ABC-Bench, "NeurIPS 2025 Workshop BioSafe GenAI", openreview mo5H9VAr6r | arXiv 2606.11150 meta tags; NeurIPS 2025 virtual page exists | Authors match (Liu, Nedungadi, Cai, Kleinman, Bhasin, Donoughe). arXiv version 2026-06-09 and an ICML 2026 poster exist. OpenReview ID not confirmed (redirect only). Cite the arXiv version. |
| Bloomfield, Black, Crook et al. 2026, Science, "A Biosecurity Data Level framework ..." | web search; science.org not fetchable | **Unverified.** The related preprint is arXiv 2602.08061, "Securing Dual-Use Pathogen Data of Concern", first author Doni Bloomfield (v1 prints "Bloomfield, L."). Caleb to supply the Science DOI or cite the preprint. |
| Compass Pathways 2018 press release | title as printed | Title says treatment-resistant depression; body text (tex:669) drops "treatment-resistant". |
| Organization for Ethical Source 2024, HL3 | firstdonoharm.dev/version/3/0/mil.html | §3.1.11 "Military Activities: Be an entity or a representative, agent, affiliate, successor, attorney, or assign of an entity which conducts military activities". v1 body says MIL "prohibits use in autonomous weapons systems". **Overstated.** |
| SecureBio 2026, "BioTier: Access stratification ..." securebio.org/biotier | arXiv 2607.14479 | The artefact is "BioTIER: A Refusal Benchmark for Targeted Biological Risk Mitigation" (Marshall et al., 2026-07-16), a refusal benchmark. **Mis-described**, and post-dates v1. |
| Sandbrink 2026; Crook 2026 keynotes | no public record found | Not citable in a journal. |
| Apart Research 2026 project page | HTTP 404 (2026-10-22) | Remove (Caleb 2026-10-22: review records not needed). |
| Google DeepMind 2026, Gemma 4 model card | not checked | open |
