# Project Progress Reconstruction

**Audit date:** 2026-09-16  
**Scope:** Repository history, documentation, prompts, code, manifests, retained outputs, and generated reports through the current local checkout.  
**Important current-state qualification:** The local V4 artifacts were last summarized on 2026-08-24 and do not include the results of any later cluster resume that has not been copied back into this repository.

## 1. Executive summary

This project asks whether large-language-model-derived semantic feature norms can replace or approximate the human Leuven feature data used to train the Integrated Semantics Control with Context Inference (ISC-CI) model. The work developed through four main methodological versions:

1. **V1** implemented an atomic concept-by-feature judging pipeline and planned held-out reconstruction tests and downstream feature expansion. It was an implementation/protocol stage; no retained completed V1 experiment was found.
2. **V2** used Qwen2.5-72B-Instruct to judge every one of 1,995 existing Leuven feature columns for every one of 293 concepts. Three prompt variants, all served by the same model, produced independent atomic first-pass calls. The final ISC-CI comparison used the 385-feature schema retained by the human preprocessing and converted every nonzero resolved V2 value to a positive. V2 recovered human positives well (recall `.864`) and strongly recovered human object geometry (`r=.821`), but precision was only `.232` and density was `26.6%`, compared with `7.1%` for the human matrix.
3. **V3** changed the task from judging supplied features to freely generating features, better matching the first stage of Leuven norm collection. Twenty simulated participants per concept and prompt generated at most ten features. After prompt-specific consolidation and the human-style recurrence cutoffs, V3 produced only 94-121 trainable contexts and very sparse matrices (`2.2-2.3%`). Human object geometry was weak (`r=.060-.122`). The models nevertheless recovered some downstream ISC-CI behavior, showing that useful structure was present but that spontaneous production did not recover the human matrix.
4. **V3.1** changed prompts only, attempting to elicit less optimized and more participant-like responses. Prompt B was the clear improvement: retained contexts rose from 115 to 175 and human object-geometry correlation rose from `.090` to `.241`. The improvement was not universal across prompts or behavioral simulations, and V3.1 remained outside the human run-to-run ceiling.
5. **V4** combines broad, high-recall feature discovery with atomic concept-by-feature completion. Discovery is complete: V3, V3.1, and seven new V4 prompt families were pooled into 133,081 candidate propositions. The intended 38,992,733-cell judgment matrix is not complete in the local snapshot. It contains 2,508,408 valid reusable resolutions, with matrices, primary ISC-CI training, and paper simulations still pending. A post-hoc prompt-C cascade was validated and approved to resume production; it preserved more than 99.6% of positive cells in both complete V2 and a V4 pilot while reducing estimated calls.

The strongest current conclusion is therefore qualified: LLM-derived norms preserve meaningful semantic and behavioral structure, but none of the completed LLM conditions is interchangeable with human norms. V2's high recall and geometry show that an LLM can recognize much of the structure represented in a human-seeded vocabulary. V3/V3.1 show that spontaneous generation is a much harder retrieval problem than endorsing a supplied property. V4 is the still-unfinished test of whether broad discovery plus completion can combine those strengths.

## 2. Overall research objective

ISC-CI learns from episodes in which one semantic feature defines a context, one or two objects sharing that feature form the support set, and positive and negative query objects must be classified. The paper's training environment was constructed from 293 Leuven concepts and 385 retained human feature contexts. The model learns both to infer the feature/context and to predict which other objects belong in that context.

The project's central research objective is:

> Can LLM-derived semantic feature data train ISC-CI models that recover the representations and behavioral phenomena obtained when ISC-CI is trained on human Leuven norms?

The validation strategy deliberately goes beyond native training accuracy. It compares fixed-context representations, membership outputs, input-matrix geometry, and the paper's induction and similarity simulations against:

- four independently retrained human-data models, which define a human self-ceiling;
- four released ISC-CI checkpoints, which test whether the training reconstruction is faithful; and
- matched LLM-derived conditions trained with the same architecture and training procedure.

**Evidence: Directly documented.** See `ISCCI_2026Jan_ArXiv.pdf`, `ISC-CI_LLM_validation/README.md`, `ISC-CI_LLM_validation/DECISIONS.md`, and `ISC-CI_LLM_validation/reports/RESULTS.md`.

## 3. Chronological development

### Stage 0: Human ISC-CI baseline and validation reconstruction

#### Research question

Can the original human-trained ISC-CI procedure be reconstructed closely enough that human-versus-LLM differences can be attributed to feature data rather than a failed training implementation?

#### Motivation

A meaningful LLM-data comparison requires a model self-ceiling and a successful reproduction of the released human checkpoints.

#### Method

- Human source: 293 Leuven concepts and 1,995 raw/consolidated feature columns.
- Human binarization: a cell is positive when the count is strictly greater than 3; a feature is retained when it is positive for strictly more than 3 objects. This yields 385 features and 8,056 positive cells.
- Model architecture: released ISC-CI architecture with a frozen 64-dimensional pretrained context-independent semantic embedding, a 128-dimensional context layer, and a 128-dimensional context-dependent layer.
- Training: seeds 0-3; 400 epochs; 1,024 sampled episodes per epoch; executed batch size 128; one- and two-shot supports; Adam learning rate `.001`; task-loss weight `.5`; 409,600 sampled episodes per run.
- Evaluation: all 293 singleton supports, 1,024 sampled unordered support pairs, all 293 query objects, 512 fixed contexts for context RDMs, and 128 fixed contexts for context-dependent RDMs. Evaluation seed: `20260804`.
- Simulations: the two paper notebooks were reproduced in a deterministic command-line pipeline covering five induction datasets, seven induction phenomena, nine similarity domains, asymmetry, paired choices, and the similarity-context LCA.

The decision log explains one reconstruction choice: released checkpoint metadata says `batch_size=1024`, but retained metrics show eight batches of 128 examples per 1,024-episode epoch. Retraining follows the executed batch size of 128.

#### Key results

| Comparison | Context RDM | Context-dependent RDM | Membership rank | Binary agreement | Probability MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Human retrain self-ceiling | `.982` | `.810` | `.892` | `.952` | `.056` |
| Released vs retrained human | `.986` | `.840` | `.911` | `.952` | `.055` |

The released models were at or slightly above the human-retrain self-ceiling on all five measures. Mean final-50-batch loss was `1.454` for released models and `1.444` for retrained models; query accuracy was `.931` and `.932`, respectively.

**Evidence: Supported by results.** `ISC-CI_LLM_validation/reports/RESULTS.md` and `ceiling_summary.csv`.

#### Interpretation at the time

1. **Direct empirical result:** The reconstructed human training runs behave like the released checkpoints within ordinary human-run variation.
2. **Interpretation:** Later differences between human and LLM conditions are unlikely to be artifacts of an incorrect ISC-CI training reconstruction.

#### Question answered

**Question:** Is the reconstructed ISC-CI pipeline a valid basis for comparing human and LLM feature matrices?  
**Answer:** Yes. Released-versus-retrained agreement is at least as high as human retrain self-agreement.

#### New questions raised

With training reconstruction established, the project could test whether LLM-derived matrices fall within or outside the human self-ceiling and which parts of the human structure they recover.

---

### Stage 1: V1 atomic feature-judgment prototype

#### Research question

Can an LLM reconstruct Leuven concept-by-feature cells using isolated atomic judgments, and can the resulting pipeline expand the Leuven feature schema to new DRM items?

#### Motivation

The initial project needed an auditable way to ask an LLM about one concept-feature pair at a time without leaking dataset labels, false-memory outcomes, or surrounding matrix context.

#### Method

- Three archived prompt variants asked whether and how strongly a supplied feature applied to one target concept.
- All calls used the same fixed user payload: target word, feature ID, feature text, a 0-4 scale, and four obvious few-shot examples.
- Three first-pass prompt variants were followed by disagreement-triggered adjudication.
- Planned validation included cell holdout and category-stratified word holdout, followed by DRM expansion.
- Temperature was intended to be `0`; calls were stateless and atomic.

Exact prompts are retained in:

- `leuven_expansion/prompts/archive_v1/feature_judge_prompt_A_v1.txt`
- `leuven_expansion/prompts/archive_v1/feature_judge_prompt_B_v1.txt`
- `leuven_expansion/prompts/archive_v1/feature_judge_prompt_C_v1.txt`
- `leuven_expansion/prompts/archive_v1/feature_adjudicator_prompt_v1.txt`

#### Key results

No completed V1 result artifact was found. The original `todo.md` still lists the mock run, vLLM smoke test, held-out validations, feature-level inspection, and production DRM expansion as unfinished. The initial implementation had 98 passing unit tests, but that establishes software behavior, not scientific validity.

**Evidence: Directly documented.** Git commit `7374e70`, `todo.md`, and `scratchpad.md`.  
**Status:** Implemented protocol; no retained completed experiment.

#### Interpretation at the time

1. **Direct empirical result:** None available.
2. **Interpretation:** The code established the atomic, auditable architecture later used by V2, but no claim about recovery quality can be made for V1.

#### Question answered

**Question:** Did generic applicability prompts reconstruct the Leuven matrix?  
**Answer:** Unclear. The repository contains the implementation and prompts but no completed validation output.

#### New questions raised

The archived V1 prompts mixed truth, applicability, salience, and spontaneous endorsement. This motivated a more explicit production-norm framing in V2.

---

### Stage 2: V2 full atomic reconstruction over existing Leuven features

#### Research question

Given the existing Leuven feature vocabulary, can an LLM recover the human concept-feature structure well enough to train a comparable ISC-CI model?

#### Motivation

V1's generic applicability wording did not cleanly distinguish objective truth from what people produce in a feature-norm task. V2 introduced conservative spontaneous-production prompts and executed the full Leuven cross-product.

#### Method

##### Inputs and coverage

- Concepts: all 293 Leuven words.
- Supplied feature vocabulary: all 1,995 columns in `data/leuven_combined_features_consolidated.csv`.
- Atomic pairs: `293 x 1,995 = 584,535`.
- ISC-CI comparison: the 385 features retained by human preprocessing; three V2-empty tasks were dropped, leaving 382 executable V2 contexts.

##### Model and judges

- Model: `Qwen2.5-72B-Instruct` for every first-pass and adjudication call.
- The "three judges" were **not three models**. They were three separate prompt variants A/B/C served by the same model.
- Every prompt variant evaluated every one of the 584,535 pairs. The vote file contains 584,535 rows per judge variant.
- First-pass calls were stateless and did not see other judges' outputs. They differed by system prompt but shared the same model, deterministic temperature, user message, scale, and few-shot examples.
- Temperature: `0.0`; no sampling seed was supplied. `top_p` was not explicitly passed and therefore used the backend default.
- The retained manifest does not record an exact model revision or checkpoint hash; it records only `Qwen2.5-72B-Instruct`. This prevents byte-for-byte model reconstruction from the manifest alone.

##### Exact prompt references

- A, extended production-frequency rubric: `leuven_expansion/prompts/feature_judge_prompt_A_v2_production.txt`
- B, concise conservative rubric emphasizing that false positives are worse: `leuven_expansion/prompts/feature_judge_prompt_B_v2_production.txt`
- C, checklist emphasizing first-to-mind salience and rejecting generic truths: `leuven_expansion/prompts/feature_judge_prompt_C_v2_production.txt`
- Adjudicator: `leuven_expansion/prompts/feature_adjudicator_prompt_v2_production.txt`

All system prompts explicitly say that zero means typical participants would not spontaneously produce the feature, even when it is technically true. However, the common user message supplied a generic scale beginning with `0 = feature does not apply`, and the few-shot examples contrasted obvious truth and falsehood. This is a real instruction mismatch within the executed prompt stack.

##### Agreement and adjudication

Adjudication was triggered by any of the following:

- a parse error or fewer than three valid votes;
- first-pass range at least 2;
- a 0-versus-3-or-4 extreme disagreement;
- a minority vote with confidence at least `.80` when rounded votes differed; or
- at least two `ambiguous=true` votes.

When no adjudication was needed, rounded unanimity returned the first value; otherwise a small disagreement was resolved by the arithmetic mean.

When adjudication was triggered, the adjudicator saw:

- the target word, feature ID, and feature text;
- all three first-pass values;
- all three confidence scores;
- all ambiguity flags; and
- all first-pass rationales.

The code loops over three adjudicator indices, but all three calls use the same prompt, model, user message, temperature `0`, and no seed. `VLLMClient` caches on exactly those fields. The retained artifact confirms that all three adjudicator rows are identical in value, confidence, and rationale for all 79,750 adjudicated cells. Thus the nominal three-adjudicator panel is **effectively one deterministic adjudicator decision repeated three times**, not three independent adjudicators. The resolution code then takes the mean of agreeing copies, which is the same single decision.

##### Final binary rule

The finalized ordinal resolution was used directly: `0` became absent and **any value greater than zero became present**. No post-hoc frequency, confidence, or object-count filter was applied to V2 cells. This follows the recorded investigator decision for V2.

This differs from the human target in two important ways:

1. The V2 system prompt estimates spontaneous production, while the final Leuven matrix reflects a separate applicability-rating stage.
2. Human cells are positive only when the count is `>3` (effectively 4/4 raters), while V2 treats any nonzero resolved value as positive.

These criterion differences are plausible contributors to low precision and high density; they are not proof of a pure model yes-bias.

##### Implementation details

- Up to 64 concept-feature pairs were processed concurrently with `ThreadPoolExecutor`.
- Within each pair, A, B, and C were called sequentially and independently.
- vLLM used four 80 GB GPUs, tensor parallelism 4, maximum model length 4,096, maximum 256 sequences, and chunked prefill.
- Structured JSON schema was used for first-pass calls; one same-prompt retry followed a failed parse/call.
- Outputs were flushed incrementally and resume used the presence of three first-pass votes.
- The final artifact has all 1,753,605 expected first-pass rows but 584,532 rather than 584,535 resolution rows. The three missing resolutions are outside the retained 385-feature validation subset, which is complete at 112,805 cells. This likely reflects the historical resume rule skipping cells once three votes existed even if a resolution had not yet been written.

#### Key results

##### Human recovery and density

Across the 382 shared executable features:

| Metric | V2 result |
| --- | ---: |
| Human-positive recall | `.864` |
| Precision | `.232` |
| F1 | `.366` |
| Balanced accuracy | `.822` |
| MCC | `.376` |
| V2 positive cells | `29,759` |
| V2 matrix density | `26.6%` |
| Human matrix density | `7.1%` |
| Input object-RDM Spearman vs human | `.821` |

The all-385-feature calibration view gives the closely related locked-rule recall `.858`, precision `.232`, MCC `.374`, and density `.264`. The difference is whether the three V2-empty contexts are excluded.

##### ISC-CI comparison to human retrains

| Measure | Human self-ceiling | V2 vs human |
| --- | ---: | ---: |
| Context RDM | `.982` | `.886` |
| Context-dependent RDM | `.810` | `.773` |
| Membership-logit rank | `.892` | `.748` |
| Binary membership agreement | `.952` | `.780` |
| Membership probability MAE | `.056` | `.248` |

V2 reached 95.4% of the human self-ceiling for context-dependent RDM correlation and 83.8% for membership-rank correlation.

##### Paper simulations

V2 mean induction correlation was `.554`, exceeding the retrained-human mean `.498`; mean in-domain similarity correlation was `.511` versus `.559` for human. V2 asymmetry was `.464`, context non-monotonicity agreement `.659`, thematic agreement `.440`, and similarity-context direction agreement `.967`.

##### Newly computed V2 judge diagnostics

The following summaries were computed during this audit from the retained vote, resolution, and human matrix CSVs. They were not previously reported in the generated research reports.

| First-pass prompt | Positive rate, all 1,995 features | Positive rate, retained 385 | Precision vs strict human cell | Recall vs strict human cell |
| --- | ---: | ---: | ---: | ---: |
| A | `18.8%` | `29.8%` | `.215` | `.898` |
| B | `13.5%` | `23.5%` | `.257` | `.846` |
| C | `25.9%` | `36.5%` | `.181` | `.923` |

On the retained 112,805 cells:

- 77.3% had exactly identical ordinal A/B/C values.
- 86.1% had binary unanimity: 71,250 all-negative and 25,904 all-positive.
- The final matrix contained 29,759 positives; 25,900 of these came from all-three-positive first-pass cells. Only 3,859 final positives came from mixed binary patterns.
- Overall adjudication rate was 79,750/584,532 = 13.64%; within the retained schema it was 20,559/112,805 = 18.23%.
- All 239,250 stored adjudicator rows represent three exact copies for each of 79,750 cells.

These diagnostics show a real prompt effect: B was most conservative and had the best precision/F1; C endorsed the most cells and had the highest recall but lowest precision. They also show that adjudication/aggregation was not the sole source of overendorsement: 87.0% of final positives were already positive under all three first-pass prompts.

#### Interpretation at the time

1. **Direct empirical result:** V2 recovered most human-positive cells and strongly preserved human object geometry, but marked many additional cells positive and remained outside human self-variation on downstream measures.
2. **Interpretation:** Atomic recognition of a supplied feature is much easier for the model than reproducing the sparse human matrix. The retained documents describe this as overendorsement. The evidence supports several nonexclusive explanations: prompt-specific endorsement tendencies, the nonzero-positive aggregation rule, criterion mismatch between production prompts and applicability data, the human matrix's strict 4/4 cutoff, and genuine ambiguity in many supplied features. The repository does not isolate one explanation as definitive.

#### Question answered

**Question:** Does full atomic LLM judgment over a human-created feature vocabulary recover human semantic structure?  
**Answer:** Partially. It achieves high recall and strong object geometry, but low precision and substantial density inflation prevent equivalence to human norms.

#### New questions raised

- Is the feature vocabulary itself doing much of the work?
- Can an LLM spontaneously discover the human feature dimensions without being shown them?
- Does the gap reflect feature retrieval, applicability completion, thresholding, or prompt criterion?
- Can broad coverage be retained without V2's density inflation and large computational cost?

---

### Interlude: the unexecuted V3 applicability branch

On 2026-06-01, the repository briefly introduced prompts named `v3_applicability`. These accurately described the original Leuven second-stage task as predicting how many of four independent raters would endorse a supplied feature. A positive-verifier prompt was also added. On 2026-07-31, these prompts were deleted and the name V3 was reassigned to free generation.

No retained result artifact for the applicability-prompt branch was found. It must therefore be treated as an abandoned or unexecuted design, not as the empirical V3 result.

**Evidence: Directly documented in Git history.** Commits `ba543a2`, `2fd324d`, and `fe73e05`.

This naming history matters because later V4 documents describe V2 as an applicability pipeline, while the frozen V2 prompt files explicitly ask for spontaneous-production likelihood. The scientific intention and executed wording are not identical.

---

### Stage 3: V3 free feature generation

#### Research question

Can Qwen recreate the human feature-generation task itself, discovering a useful semantic feature inventory without seeing existing Leuven feature columns?

#### Motivation

V2 used human-created feature dimensions, so its success did not establish that an LLM could discover those dimensions. The original Leuven procedure began with free feature production by at least 20 participants. V3 shifted from supplied-feature judgment to one-word free generation.

#### Method

- Model: Qwen2.5-72B-Instruct.
- Concepts: all 293 Leuven words.
- Prompt variants: A original-style, B concise, C structured perspectives.
- Simulated participants: 20 responses per concept per prompt, for 17,580 planned and valid responses.
- Experimental unit: one concept x one prompt x one simulated participant call.
- Sampling: temperature `.8`, paired seeds across A/B/C, base seed `20260801`, maximum 500 tokens.
- Response: zero to ten free feature phrases in JSON; no Leuven columns, categories, human values, or other words were shown.
- Generation produced 175,796 valid feature rows. Prompt A returned fewer than ten features in only 4/5,860 responses; B and C returned exactly ten in every valid response.
- Prompt variants were kept as separate conditions rather than pooled.

Exact prompts:

- `leuven_expansion/prompts/feature_generation_prompt_A_v3_original.txt`
- `leuven_expansion/prompts/feature_generation_prompt_B_v3_concise.txt`
- `leuven_expansion/prompts/feature_generation_prompt_C_v3_structured.txt`

##### Consolidation

- Generated phrases were normalized and consolidated within prompt only.
- Lexical variants were grouped; semantic candidates used normalized `all-MiniLM-L6-v2` embeddings at `.85` and 293-object generation-profile cosine at `.50`.
- Complete-link constraints prevented transitive chaining.
- Substantive modifiers, conjunctions, alternatives, negation, and CJK text received safeguards.
- A cluster was counted once per response.
- Object-cluster cells became positive at strict response count `>3`; clusters were retained as tasks when positive for `>3` objects.
- Seven retained semantic/profile clusters were manually reviewed and passed.
- V3 clusters became new native ISC-CI task dimensions. They were **not mapped onto the 385 human feature labels** for training. Comparison occurred through shared object geometry and downstream behavior.

##### ISC-CI validation

Each A/B/C binary matrix trained four ISC-CI seeds with the human reconstruction's architecture, frozen embedding, episode sampler, optimizer, 400 epochs, and fixed probes.

#### Key results

| Prompt | Phrase types | Consolidated clusters | Retained tasks | Positive cells | Density | Object RDM vs human |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 A | 15,977 | 12,271 | 94 | 607 | `2.2%` | `.060` |
| V3 B | 14,669 | 11,584 | 115 | 756 | `2.2%` | `.090` |
| V3 C | 15,294 | 12,221 | 121 | 817 | `2.3%` | `.122` |

Compared with V2, free generation sharply decreased trainable feature/context coverage, matrix density, and recovery of human input geometry. Precision and recall against individual human feature cells were not computed and are not directly defined because V3 used a different, generated feature vocabulary.

##### ISC-CI comparison

| Prompt | Context RDM | Context-dependent RDM | Membership rank | Binary agreement | Probability MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| V3 A | `.864` | `.644` | `.636` | `.922` | `.110` |
| V3 B | `.887` | `.642` | `.656` | `.923` | `.110` |
| V3 C | `.887` | `.639` | `.657` | `.923` | `.107` |

High binary agreement was interpreted cautiously because V3 matrices were very sparse and shared negative predictions can inflate agreement.

##### Paper simulations

| Prompt | Mean induction r | In-domain similarity r | Asymmetry r | Context nonmono | Thematic | Context direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 A | `.387` | `.473` | `.192` | `.738` | `.759` | `.875` |
| V3 B | `.450` | `.478` | `.357` | `.728` | `.543` | `.842` |
| V3 C | `.448` | `.505` | `.299` | `.725` | `.586` | `.947` |

Prompt C had the strongest broad profile, B was strongest for asymmetry, and A retained a thematic advantage.

##### Consolidation audit

Changing the embedding threshold from `.80` through `.90`, or removing semantic merges while retaining current lexical signatures, did not change the 94/115/121 task counts. The `.50` profile guard prevented clearly invalid merges. The more important risk was lexical normalization, which sometimes erased modality, frequency, or derivational distinctions. The clearest retained error merged `can be washed` with `washed frequently`.

#### Interpretation at the time

1. **Direct empirical result:** Free generation produced valid, behaviorally useful ISC-CI conditions, but native object geometry was weak and the inventories contained only 24-31% as many trainable contexts as the 385-feature human matrix.
2. **Interpretation:** The model often knows that a feature applies when shown it, as V2 demonstrated, but does not reliably retrieve and repeat that feature across independent free-generation responses. Recognition/applicability knowledge and spontaneous production are therefore empirically distinct in this pipeline.

#### Question answered

**Question:** Does free LLM feature generation recover the human Leuven feature space?  
**Answer:** No, not closely. It recovers some downstream behavior but substantially under-recovers recurring feature dimensions and human object geometry.

#### New questions raised

- Can prompt wording make generations more participant-like and increase recurrence rather than surface novelty?
- Is poor coverage caused by consolidation or by generation behavior?
- Would restoring a separate applicability/completion stage recover features known but not spontaneously generated?

---

### Stage 4: V3.1 prompt-only generation follow-up

#### Research question

Can prompt wording alone increase recurring, human-like feature production while keeping model, sample size, sampling, consolidation, training, and evaluation fixed?

#### Motivation

V3 produced polished ten-feature lists, many surface phrase types, few recurring trainable contexts, and weak human geometry. The V3.1 prompts attempted to reduce optimized/encyclopedic list construction and emphasize first-to-mind order and individual-participant variability.

#### Method

- Same Qwen2.5-72B-Instruct model, 293 concepts, 20 paired participant seeds, temperature `.8`, base seed, schema, and maximum of ten features as V3.
- All 17,580 responses completed with no parse or request errors, producing 175,766 feature rows.
- A was a faithful human-instruction control.
- B emphasized first-to-mind order, ordinary recall, and stopping rather than balanced expert coverage.
- C simulated one plausible participant rather than an averaged or encyclopedic answer.
- A and C still returned exactly ten features in every response; B stopped at nine in 34 of 5,860 responses and otherwise returned ten.
- Consolidation, thresholds, ISC-CI training, probes, and simulations were locked to V3.
- Five retained semantic merges were reviewed; four passed. `driven on highways` versus `driven on roads` was rejected and split before thresholding.

Exact prompts and rationale:

- `leuven_expansion/prompts/feature_generation_prompt_A_v3_1_faithful.txt`
- `leuven_expansion/prompts/feature_generation_prompt_B_v3_1_first_to_mind.txt`
- `leuven_expansion/prompts/feature_generation_prompt_C_v3_1_individual_participant.txt`
- `leuven_expansion/prompts/V3_1_PROMPT_RATIONALE.md`

#### Key results

| Prompt | Retained tasks V3 -> V3.1 | Positive cells V3 -> V3.1 | Object RDM V3 -> V3.1 | Context-dependent RDM V3 -> V3.1 | Membership rank V3 -> V3.1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | `94 -> 88` | `607 -> 547` | `.060 -> .056` | `.644 -> .614` | `.636 -> .585` |
| B | `115 -> 175` | `756 -> 1,212` | `.090 -> .241` | `.642 -> .653` | `.656 -> .659` |
| C | `121 -> 128` | `817 -> 861` | `.122 -> .132` | `.639 -> .668` | `.657 -> .638` |

V3.1 densities remained low: A `2.1%`, B `2.4%`, and C `2.3%`.

Paper-simulation changes were mixed. V3.1-B improved in-domain similarity (`.478 -> .534`), asymmetry (`.357 -> .374`), thematic agreement (`.543 -> .621`), and context non-monotonicity (`.728 -> .731`), while induction (`.450 -> .425`) and similarity-context direction (`.842 -> .757`) declined. V3.1-C achieved the strongest free-generation induction result (`.474`) but declined on several other behaviors.

#### Interpretation at the time

1. **Direct empirical result:** Prompt B increased recurring feature coverage by 52%, more than doubled its correlation with human object geometry, and moved all five fixed-context measures toward the human retrains. A and C did not show a comparable general improvement.
2. **Interpretation:** Prompt wording can materially affect the quality of generated norms, but asking for first-to-mind ordinary recall appears more useful than merely asking for fidelity or an imagined individual. The evidence remains descriptive and model-family-specific.

#### Question answered

**Question:** Can prompt engineering alone solve V3's undercoverage?  
**Answer:** Partially. V3.1-B substantially improved coverage and geometry, but all free-generation conditions remained sparse, outside human run-to-run variability, and far below V2's `.821` input-geometry correlation.

#### New questions raised

- Would pooling broader prompt families increase recall without relying on any one prompt?
- Can generated candidates be separated from their original source concepts and completed across all objects?
- Can the completion step avoid V2's high cost and overendorsement?

---

### Stage 5: V4 high-recall discovery plus atomic completion

#### Research question

Can broad LLM feature discovery followed by concept-by-feature completion recover human semantic geometry and ISC-CI behavior better than V2's human-seeded vocabulary or V3.1's generation-only matrix?

#### Motivation

V2 showed strong recovery when features were supplied but could not test discovery and overendorsed cells under its binary rule. V3.1-B improved discovery but omitted the original Leuven applicability/completion stage and remained sparse. V4 was designed to separate feature discovery, consolidation, and matrix completion.

#### Intended and implemented method

##### Discovery

- Imported all V3 and V3.1 generation sources.
- Added seven high-recall V4 prompt families: first-to-mind, perceptual/structural, functional/affordance, taxonomic/definitional, situational/relational, encyclopedic/causal, and less-obvious/general.
- Used Qwen2.5-72B-Instruct, 293 words, 20 responses per word per prompt, temperature `.8`, maximum ten propositions, and base seed `20260801`.
- The V4 round produced 41,020 responses, 410,154 generated feature rows, and 203,962 exact phrase types.
- Rare features and singletons were retained; V3's pre-judgment recurrence cutoffs were deliberately not applied.

Exact prompt references are the seven `leuven_expansion/prompts/feature_generation_prompt_*_v4.txt` files listed in `configs/v4_discovery.json`.

##### Consolidation and candidate bank

- Pooled V3, V3.1, and V4 sources.
- Used exact proposition normalization plus `all-MiniLM-L6-v2` similarity `.85`, profile similarity `.50`, nearest-neighbor count 128, and complete-link safeguards.
- The PRD originally required review of every nontrivial merge. At investigator direction, the executed pipeline automatically accepted all 56,294 proposed merges and rejected none.
- Frozen result: 133,081 candidate propositions; inventory hash `cf0af7b1c8126e2a06ad162e916685b75a298c3cf8da9a89b8ecd9e9652da3fd`.
- Human mapping was not used to construct the candidate bank.
- The exact ordered 175-context V3.1-B inventory was frozen as a fixed-vocabulary control.

The automatic acceptance of every proposed merge is a documented deviation and a sensitivity limitation. The repository does not yet contain a completed downstream comparison showing whether these merges are too aggressive.

##### Atomic completion

The intended primary matrix contains every candidate crossed with all 293 words: `133,081 x 293 = 38,992,733` cells. Source words do not automatically become positive; provenance is hidden from atomic calls.

The PRD calls this an applicability/completion stage, but the implementation freezes and reuses the V2 **spontaneous-production** A/B/C prompts and resolver. This terminology-versus-prompt conflict should be preserved in any research report.

The run began as a full A/B/C panel. After only 2,508,408 valid resolutions completed, a validated prompt-C cascade was approved:

1. Every unresolved cell receives prompt C.
2. Prompt-C positives, ambiguous outputs, confidence below `.80`, or parse failures receive A and B plus the frozen V2 resolver/adjudicator.
3. Other high-confidence prompt-C negatives resolve directly to zero.
4. Existing complete full-panel cells remain unchanged.

##### Calibration and planned conditions

Calibration used only completed V2 judgments and five held-out word folds. The selected threshold was `resolved_value >= 1.0`, maximizing mean MCC subject to recall at least `.80`. It achieved recall `.854`, precision `.245`, F1 `.381`, MCC `.388`, density `.249`, and input object-RDM correlation `.822`. The executed V2 `>0` rule remains a control.

Planned V4 ISC-CI conditions are:

- `v4_b_locked_v2`
- `v4_b_calibrated`
- `v4_ensemble_locked_v2`
- `v4_ensemble_calibrated`

All are to use the same four-seed ISC-CI training, fixed probes, and paper simulations as V2/V3/V3.1.

#### Key results available so far

##### Discovery

- Candidate bank complete: 133,081 candidates.
- V4 round complete: 410,154 generated feature rows from seven prompt families.
- All 56,294 proposed consolidation merges accepted automatically.

##### Retrieval retrospective

Embedding shortlist values `K=25,50,75,100` failed the preregistered `.95` positive-recall and `.99` geometry gates. `K=100` retained only 74.2% of V2-positive cells. Development selection escalated to `K=275`; held-out features then retained 98.24% of positives and object geometry `.9941`, but screened 94.47% of all cells, saving only 5.53%. Embedding retrieval therefore remained post-hoc rather than the production shortcut.

##### Prompt-C cascade validation

| Benchmark | Routed to full panel | Positive recall | Context recall | Object geometry | Estimated call reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complete V2 | `.259` | `.9963` | `.9995` | `.9996` | `.435` |
| Complete V4 pilot | `.020` | `.9968` | `.9934` | `.9987` | `.647` |

Matched cascade/full-V2 ISC-CI runs had context-RDM correlation `.9917`, context-dependent RDM correlation `.9466`, membership-logit correlation `.9361`, and binary membership agreement `.8955`. The cascade was closer than ordinary V2 seed-to-seed variation on all five representational metrics. Across 184 matched paper-simulation rows, mean absolute change was `.0393` and sign agreement `.978`; context-dependent non-monotonicity decreased from `.659` to `.591`.

##### Current completion state

The local repository snapshot contains:

- 2,508,408 valid reusable resolutions (`6.43%` of 38,992,733);
- 36,484,325 remaining cells (`93.57%`);
- no finalized V4 matrix;
- no primary V4 ISC-CI checkpoints;
- no primary V4 fixed-context results; and
- no primary V4 paper-simulation results.

The report CSVs for V4 behavior, primary contrasts, pruning/completion, and representation contain headers only. The 32 local shard manifests are migration/resume sidecars and do not report completion. A later cluster run may be ahead of this local state.

#### Interpretation at the time

1. **Direct empirical result:** High-recall discovery and the candidate bank are complete; the cascade has strong retrospective/pilot fidelity; the primary V4 scientific comparison is not complete.
2. **Interpretation:** Prompt-C cascading is a plausible engineering acceleration, while embedding-only retrieval is too weak at useful K values. No conclusion about whether V4 improves on V2 or V3.1 is yet supported.

#### Question answered

**Question:** Does V4's combined discovery-and-completion method outperform prior versions?  
**Answer:** Not yet answerable. Discovery and efficiency validation are complete, but the full matrix and downstream ISC-CI analyses are pending in the local record.

#### New questions raised

- Does the completed V4 matrix improve human object geometry and downstream simulations?
- How much do completion and broader discovery each contribute?
- Do automatically accepted candidate merges materially distort the feature inventory?
- Does the cascade remain faithful on a stratified sample of the entire V4 vocabulary?
- Would results change under prompts that reproduce the original four-rater applicability criterion rather than V2's production criterion?

## 4. Detailed V2 protocol audit

### What "three judges" means

All three first-pass judges were Qwen2.5-72B-Instruct. Judge IDs A/B/C identify prompt variants, not different models. Calls were isolated by prompt and did not share chat history, but they were not independent model samples in a statistical sense: they used one model, temperature zero, no seed, and highly overlapping instructions.

### Agreement definition

The resolver first examined numeric spread, rounded-value disagreement, confidence, and ambiguity. Exact rounded agreement resolved without adjudication. Small non-triggering disagreement was averaged. There was no simple majority-vote rule over binary labels.

### Adjudicator behavior

The adjudicator saw the original first-pass values and rationales. Although the implementation stores three adjudicator rows, caching made them exact duplicates. The effective final rule for adjudicated cells was therefore the single deterministic adjudicator output. For non-adjudicated cells, the first-pass unanimity or mean remained final.

### Why V2 produced many positives: state of evidence

| Candidate explanation | Evidence | Current assessment |
| --- | --- | --- |
| Permissive wording | System prompts are explicitly conservative, especially B. The shared user scale says `0 = feature does not apply`, which conflicts with the production framing. | Mixed; not a simple permissive-prompt story. |
| Prompt/model yes-bias | C marked 36.5% of retained cells positive, A 29.8%, B 23.5%, all above the 7.1% human density. | Supported descriptively; strongest for C. |
| Judge disagreement | 86.1% of retained cells had binary unanimity. | Disagreement exists but is not the main source of positives. |
| Adjudicator inflation | Only 18.2% of retained cells were adjudicated; 87.0% of final positives were already positive under all three judges. | Not the primary explanation. |
| Nonzero aggregation | Every resolved value above zero counted positive. | Directly documented and clearly density-increasing. |
| Criterion mismatch | V2 asks about production; human cells reflect four-rater applicability. | Directly documented conflict. |
| Strict/sparse human target | Human positivity requires count `>3`; V2 positivity requires value `>0`. | Directly documented and likely important. |
| Feature ambiguity | Disagreement triggers and rationales show ambiguous/peripheral cases, but `ambiguous=true` was rare in first-pass outputs. | Plausible but not quantified as a dominant cause. |

The available evidence therefore does not justify attributing low precision to one mechanism. The strongest supported account is a combination of criterion/threshold mismatch and model endorsement behavior, with prompt C contributing more positives than A or B.

## 5. Prompt-engineering experiments

| Variant | Intended change | Model/sample | Result |
| --- | --- | --- | --- |
| V1 A/B/C | Generic applicability/commonness with varying conservatism/checklist structure | Planned Qwen atomic judgments | No retained completed evaluation. |
| V2 A | Detailed spontaneous-production frequency rubric with examples | Qwen2.5-72B, all 584,535 cells | Retained-cell positive rate `.298`, precision `.215`, recall `.898`. Newly computed. |
| V2 B | More concise; false positives explicitly worse | Same | Lowest positive rate `.235`, best precision `.257` and first-pass F1 `.394`, recall `.846`. Newly computed. |
| V2 C | First-to-mind checklist and generic-truth exclusions | Same | Highest positive rate `.365`, lowest precision `.181`, highest recall `.923`. Newly computed. |
| Historical V3 applicability | Explicitly predict 0-4 of four applicability raters | No retained run | Unexecuted/abandoned; cannot evaluate. |
| V3 A/B/C generation | Original-style vs concise vs structured free generation | Qwen2.5-72B; 20 responses x 293 words x 3 prompts | A/B/C retained 94/115/121 tasks; object RDM `.060/.090/.122`. |
| V3.1 A | More faithful human wording and response order | Same matched design | Tasks fell 94 -> 88; object RDM `.060 -> .056`. |
| V3.1 B | First-to-mind ordinary participant; discourage balanced/expert lists | Same matched design | Tasks rose 115 -> 175; object RDM `.090 -> .241`; broad representational improvement, mixed simulations. |
| V3.1 C | One individual rather than averaged participant | Same matched design | Tasks rose 121 -> 128; object RDM `.122 -> .132`; mixed downstream changes. |
| V4 seven-family ensemble | Deliberately liberal high-recall discovery across semantic relation types | Qwen2.5-72B; 20 responses x 293 words x 7 prompts | 410,154 rows and 203,962 phrase types contributed to 133,081 pooled candidates; downstream primary result pending. |

V2 is the only completed prompt set supporting cell-level precision/recall comparisons against the same human columns. V3/V3.1 use generated vocabularies, so their prompt effects are evaluated through task counts, density, object geometry, and ISC-CI behavior instead.

## 6. Methodology comparison

| Version | Main question | Feature source | Applicability judgment | Judges | Aggregation | Matrix density | Precision | Recall | Similarity recovery | Main conclusion |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Human baseline | Can ISC-CI be faithfully retrained? | Human Leuven generation plus applicability norms | Four human raters; cell positive at count `>3` | Human | Retain features positive for `>3` objects | `7.1%` | N/A | N/A | Self-ceiling context RDM `.982`; released-vs-retrained `.986` | Reconstruction is valid. |
| V1 | Can atomic LLM judgments reconstruct/expand Leuven features? | Existing human Leuven columns | Generic LLM applicability prompts | One model, three prompts | Disagreement adjudication | N/A | N/A | N/A | N/A | Implemented but not empirically completed. |
| V2 | Can LLM judgments recover the matrix when features are supplied? | Existing human Leuven columns | Executed prompts estimate spontaneous production, despite later applicability terminology | Qwen2.5-72B under A/B/C; same model adjudicator | Ordinal resolver; any final value `>0` positive | `26.6%` | `.232` | `.864` | Input object RDM `.821`; model context/context-dependent `.886/.773` | Strong recall/geometry, severe overendorsement, not human-equivalent. |
| V3 A/B/C | Can the LLM discover features through free generation? | Prompt-specific free generations | None; recurrence used as cell value | 20 Qwen responses per word/prompt | Consolidate phrases; response `>3`; objects `>3` | `2.2/2.2/2.3%` | N/A | N/A | Object RDM `.060/.090/.122`; context RDM `.864/.887/.887` | Useful but sparse; knowing exceeds spontaneous production. |
| V3.1 A/B/C | Can more participant-like prompts improve discovery? | Revised free generations | None; same recurrence rule | Same matched Qwen design | Same locked consolidation and cutoffs | `2.1/2.4/2.3%` | N/A | N/A | Object RDM `.056/.241/.132`; context RDM `.865/.892/.885` | B improves substantially; no universal prompt improvement. |
| V4 | Can broad discovery plus completion combine V2 and V3.1 strengths? | Pooled V3/V3.1 plus seven V4 families | Intended completion; implemented with frozen V2 production prompts and prompt-C cascade | Same Qwen model; C screen, routed A/B/adjudicator | Locked and calibrated conditions planned | N/A | N/A | N/A | Cascade geometry `.9996` V2 / `.9987` pilot; primary V4 N/A | Discovery complete; primary scientific comparison pending. |

## 7. Question -> experiment -> answer timeline

### Question 1

**Can an isolated LLM judge reconstruct Leuven concept-feature cells without dataset leakage?**  
|  
**Experiment:** V1 atomic A/B/C prompts, disagreement resolver, planned cell and word holdouts.  
|  
**Result:** Software implemented and unit-tested; no retained completed validation.  
|  
**Answer:** Unclear empirically.  
|  
**New question:** How should the task distinguish truth/applicability from sparse feature production?

### Question 2

**Can production-framed judgments recover the human matrix when the feature vocabulary is supplied?**  
|  
**Experiment:** V2, 293 x 1,995 cells, three prompt variants, same Qwen model, deterministic adjudication.  
|  
**Result:** Recall `.864`, precision `.232`, density `26.6%` vs human `7.1%`, object RDM `.821`.  
|  
**Answer:** Partially. Most human positives and much geometry are recovered, but many extra positives are introduced.  
|  
**New question:** Is success dependent on being given the human-created features?

### Question 3

**Can the model spontaneously produce a useful Leuven-like feature inventory?**  
|  
**Experiment:** V3 free generation, 20 simulated participants, three prompts, prompt-specific consolidation and recurrence thresholds.  
|  
**Result:** 94-121 tasks, 2.2-2.3% density, object RDM `.060-.122`; partial behavioral recovery.  
|  
**Answer:** Not closely. Spontaneous generation under-recovers recurring dimensions even when the model can recognize them when supplied.  
|  
**New question:** Can prompts elicit more human-like recurrence and less optimized list construction?

### Question 4

**Can a prompt-only change improve feature discovery?**  
|  
**Experiment:** V3.1 A/B/C with all non-prompt parameters locked.  
|  
**Result:** B increased tasks 115 -> 175 and object RDM `.090 -> .241`; A worsened and C was mixed.  
|  
**Answer:** Yes for first-to-mind prompt B, but not generally and not enough to reach human equivalence.  
|  
**New question:** Can broad discovery be followed by an independent completion step across all concepts?

### Question 5

**Can high-recall discovery plus atomic completion combine V3.1's discovery with V2's recognition strength?**  
|  
**Experiment:** V4 pools V3/V3.1 and seven liberal V4 prompt families, consolidates 133,081 candidates, and plans all 38,992,733 concept-candidate cells.  
|  
**Result:** Discovery complete; only 2,508,408 valid cells are present locally; primary matrices/models/simulations pending.  
|  
**Answer:** Not yet known.  
|  
**Engineering question:** Can completion be accelerated without changing scientific conclusions?

### Question 6

**Can a retrieval or judge cascade approximate exhaustive V2-style completion?**  
|  
**Experiment:** Retrospective K-shortlists and prompt-C routing against complete V2 plus a V4 complete-feature pilot.  
|  
**Result:** Retrieval needed K=275 and saved only 5.5%; prompt-C cascade retained 99.63-99.68% of positives, geometry `.9996/.9987`, and saved an estimated 43.5-64.7% of calls.  
|  
**Answer:** Prompt-C cascading is supported as a production acceleration; aggressive embedding retrieval is not.  
|  
**New question:** Does cascade fidelity hold across the completed, stratified V4 vocabulary and all paper simulations?

## 8. Current best-supported conclusions

1. **The ISC-CI retraining implementation is faithful.** Released and retrained human models agree at or above the human self-ceiling.
2. **LLM feature data are scientifically informative but not interchangeable with human norms.** Every completed LLM condition remains outside human run-to-run variability on important continuous measures.
3. **Supplying the feature vocabulary is a major advantage.** V2 preserves human object geometry far better than V3/V3.1 free generation.
4. **Recognition and spontaneous production are different bottlenecks.** V2's high recall contrasts sharply with V3's small recurring feature inventory.
5. **V2 overendorsement is real, but its cause is not singular.** Prompt-specific positive rates differ, yet most final positives are unanimous across A/B/C. The nonzero binary rule, production-versus-applicability mismatch, and strict human 4/4 threshold are all relevant.
6. **V3.1-B is the strongest free-generation prompt tested for structural recovery.** It increases recurring contexts and human object geometry, though some behavioral measures decline.
7. **V3 consolidation's embedding threshold was not the main cause of sparse task counts.** Lexical merging has identifiable errors, but generation recurrence remains the larger limitation.
8. **V4's primary claim is still open.** The candidate bank and acceleration analyses are complete; the exhaustive scientific comparison is not complete in the local artifacts.
9. **Prompt-C cascading is well supported as an efficiency method.** Its retrospective/pilot fidelity is high, but the final V4-wide stratified audit remains pending.

## 9. Remaining unresolved questions

- Will the completed V4 ensemble exceed V3.1-B's `.241` human object-RDM correlation and approach V2's `.821`?
- Will V4's ISC-CI representations and paper simulations move toward the human self-ceiling?
- How much improvement comes from broad discovery versus cross-object completion at fixed V3.1-B vocabulary?
- How much do the 56,294 automatically accepted V4 merges affect candidate identity and downstream structure?
- Would an actual four-rater applicability prompt, such as the abandoned historical V3 design, produce a less inflated completion matrix than the frozen V2 production prompts?
- How much of V2's low precision is model endorsement versus comparison of `>0` predictions to a strict human `>3` target?
- Do newer model families or more independent simulated participants improve generation diversity and recurrence without changing the task?
- Will prompt-C cascade fidelity remain high on a stratified full-V4 audit, particularly for V3/V3.1-derived candidates?
- The original V1 held-out validation and DRM expansion were never completed in the retained record. Are they obsolete, or should they remain formally closed as unexecuted?

## 10. Conflicts, caveats, and reconstruction limits

1. **V3 name reuse:** Git history contains a short-lived V3 applicability design, but current V3 artifacts refer to free generation. This report uses V3 for the completed free-generation experiment and labels the older branch explicitly.
2. **V2/V4 task terminology:** V2 system prompts estimate spontaneous production. V4 documentation often calls their reuse "applicability judgment." The executed prompt text takes precedence when describing what the model saw.
3. **Human/V2 binary thresholds:** Human cells use `count > 3`; V2 cells use `resolved value > 0`. Precision and density comparisons reflect both model behavior and this intentionally asymmetric decision rule.
4. **Human feature-retention prose versus execution:** The paper says features true of two items or fewer were removed. The released checkpoint records `n_obj_cutoff=3`, and the upstream preprocessing code retains features whose positive-object count is strictly `>3`; that executed rule yields the reported 385 features. This reconstruction follows the checkpoint and code, while preserving the prose discrepancy for future review.
5. **Adjudicator independence:** Code comments describe a three-adjudicator panel, but caching and the artifacts show three exact copies of one deterministic response. It should not be reported as three independent adjudicators.
6. **V2 manifest incompleteness:** The manifest lacks exact model revision, top-p, prompt hashes, and aggregate prior-run parse counts. Prompt files and code recover most of the protocol, but exact model-byte reproducibility is not guaranteed.
7. **V2 missing resolutions:** Three of 584,535 full-schema pairs lack resolution rows, despite complete first-pass votes. The 385-feature validation subset is complete, so reported ISC-CI comparisons are unaffected.
8. **V3 source row history:** The V3 manifest records 17,580 valid responses after reparsing 60 preserved candidate errors without new model calls. Raw `feature_generations.csv` retains historical rows; analyses use the valid/revalidated response set.
9. **V4 local versus cluster state:** Current local reports remain partial. Recent cluster progress must be synced and finalized before this document can state a newer V4 completion percentage.
10. **Missing generated figures:** No tracked project-result PNG/SVG/PDF figure set was found. Quantitative reporting currently lives primarily in Markdown, CSV, JSON, NPZ, and checkpoint artifacts. `ISCCI_2026Jan_ArXiv.pdf` is the source paper, not a generated project result.

## 11. Newly computed summaries in this audit

The V2 first-pass prompt table, binary-pattern counts, and effective adjudicator-duplication check were computed from existing files without modifying them:

- Human retained columns: binarize `data/leuven_combined_features_consolidated.csv` at cell count `>3`, then retain columns positive for `>3` objects.
- First-pass positivity: `feature_value > 0` in `artifacts/leuven_full_labels/leuven_full_v2/feature_votes.csv`.
- First-pass precision/recall: compare those binary values with the strict human retained-cell matrix.
- Binary patterns: pivot A/B/C by `(word_normalized, feature_id)` for the retained 385 columns.
- Final positivity and adjudication: join to `feature_resolutions.csv` on the same key.
- Adjudicator duplication: group `feature_adjudication_votes.csv` by concept-feature key and compare value, confidence, and rationale across `adjudicator_idx`.

These are descriptive diagnostics of the completed V2 artifact. They do not change the locked V2 matrix, any model, or any previously generated report.

## 12. Source map

### Project overview

- `ISCCI_2026Jan_ArXiv.pdf` - Source paper defining ISC-CI, its human Leuven training environment, architecture, and target simulations.
- `ISC-CI_LLM_validation/README.md` - Run order and inventory for the human/V2/V3/V3.1 validation pipeline.
- `ISC-CI_LLM_validation/DECISIONS.md` - Most important methodological decision log, including training reconstruction, consolidation, V2 binary conversion, model ceiling, and simulations.
- `scratchpad.md` - Initial V1 assumptions and decisions; useful for distinguishing planned from completed work.
- `todo.md` - Shows that V1 held-out validation and expansion tasks remained unfinished.
- Git history - Establishes the May 2026 V1 start, June V2 and temporary applicability prompts, July/August V3 transition, and August/September V4 development.

### V2

- `leuven_expansion/prompts/feature_judge_prompt_A_v2_production.txt` - Exact detailed V2 production-frequency prompt.
- `leuven_expansion/prompts/feature_judge_prompt_B_v2_production.txt` - Exact conservative V2 prompt.
- `leuven_expansion/prompts/feature_judge_prompt_C_v2_production.txt` - Exact checklist-style V2 prompt and later cascade screen.
- `leuven_expansion/prompts/feature_adjudicator_prompt_v2_production.txt` - Exact adjudicator prompt.
- `leuven_expansion/feature_prompts.py` - Shared user message, generic scale, few-shot examples, and adjudicator payload.
- `leuven_expansion/feature_judge.py` - First-pass call independence, retry, schema, and vote recording.
- `leuven_expansion/feature_adjudicate.py` - Exact disagreement triggers and resolution rules.
- `leuven_expansion/run_jobs.py` - Concurrency, incremental output, resume, and optional verifier implementation.
- `run_leuven_full_labels.sh` - Cluster model identity and vLLM/runtime configuration.
- `artifacts/leuven_full_labels/leuven_full_v2/manifest.json` - V2 pair count, model label, run dates, and completion summary.
- `artifacts/leuven_full_labels/leuven_full_v2/feature_votes.csv` - Complete first-pass evidence used for prompt-specific diagnostics.
- `artifacts/leuven_full_labels/leuven_full_v2/feature_adjudication_votes.csv` - Stored adjudicator outputs demonstrating deterministic duplication.
- `artifacts/leuven_full_labels/leuven_full_v2/feature_resolutions.csv` - Final ordinal V2 decisions used to build the matrix.
- `ISC-CI_LLM_validation/reports/v2_binary_recovery.csv` - Published recovery metrics against the human matrix.

### V3 / V3.1

- `V3_FEATURE_GENERATION.md` - V3 experimental contract, original Leuven task rationale, sampling, and output definitions.
- `leuven_expansion/generate_features.py` - Free-generation execution, seeding, validation, resume, and manifests.
- `leuven_expansion/prompts/feature_generation_prompt_A_v3_original.txt` - V3 original-style condition.
- `leuven_expansion/prompts/feature_generation_prompt_B_v3_concise.txt` - V3 concise condition.
- `leuven_expansion/prompts/feature_generation_prompt_C_v3_structured.txt` - V3 structured-perspectives condition.
- `leuven_expansion/prompts/V3_1_PROMPT_RATIONALE.md` - Direct explanation of why V3.1 changed the prompts and the missing applicability-stage limitation.
- `leuven_expansion/prompts/feature_generation_prompt_A_v3_1_faithful.txt` - V3.1 faithful control.
- `leuven_expansion/prompts/feature_generation_prompt_B_v3_1_first_to_mind.txt` - V3.1 first-to-mind condition and strongest structural result.
- `leuven_expansion/prompts/feature_generation_prompt_C_v3_1_individual_participant.txt` - V3.1 individual-participant condition.
- `artifacts/leuven_feature_generation/leuven_v3_qwen2_5_72b/manifest.json` - Complete V3 generation protocol and counts, including reparsing history.
- `artifacts/leuven_feature_generation/v3.1/leuven_v3.1_qwen2_5_72b/manifest.json` - Complete V3.1 generation protocol and counts.
- `ISC-CI_LLM_validation/configs/v3_validation.json` - Locked V3 consolidation, training, and evaluation parameters.
- `ISC-CI_LLM_validation/configs/v3_1_validation.json` - Confirms that V3.1 changed prompts but not substantive downstream parameters.
- `ISC-CI_LLM_validation/consolidate_v3.py` and `iscci_validation/consolidation.py` - Phrase normalization, clustering, manual review, count matrices, and task retention.
- `ISC-CI_LLM_validation/reports/RESULTS.md` - Main human/V2/V3 quantitative report and interpretation.
- `ISC-CI_LLM_validation/reports/v3_1/RESULTS.md` - Prompt-matched V3.1 versus V3 results and paper simulations.
- `ISC-CI_LLM_validation/reports/consolidation_audit/CONSOLIDATION_AUDIT.md` - Diagnostic analysis showing that lexical rules, not embedding threshold `.85`, are the main consolidation concern.

### V4

- `PRD-V4-High-Recall-Feature-Discovery-and-Atomic-Completion.md` - Full scientific rationale, planned contrasts, pipeline, gates, and later execution amendment.
- `V4_REPRODUCIBILITY.md` - Concise executed-protocol record for discovery, calibration, retrieval, and planned validation.
- `configs/v4_discovery.json` - Seven prompt families, source runs, generation settings, and consolidation parameters.
- `configs/v4_validation.json` - Planned V4 conditions, unchanged training/evaluation settings, and efficiency gates.
- `artifacts/v4/discovery/candidate_bank_manifest.json` - Frozen candidate count/hash and the all-merges-accepted deviation.
- `artifacts/v4/discovery/source_inventory.csv` - Exact V3, V3.1, and V4 source counts, hashes, models, prompts, and parse status.
- `artifacts/v4/discovery/candidate_source_summary.csv` - Feature rows, phrase types, and candidate contributions by source and prompt.
- `artifacts/v4/reports/V4_RESULTS.md` - Current local stage status and pending primary comparisons.
- `artifacts/v4/scratchpad.md` - Chronological V4 execution record, partial-run counts, cascade decision, and cluster recovery issues.
- `run_v4_judgments.py` and `leuven_expansion/cascade_jobs.py` - Sharding, protocol validation, resume, prompt-C routing, and completeness logic.
- `run_leuven_v4_atomic.sh` - Current 32-shard production launcher and resume/preflight contract.

### Prompt experiments

- `leuven_expansion/prompts/archive_v1/` - Archived generic applicability prompts that preceded V2.
- Git commit `ba543a2` - Recoverable exact text of the temporary four-rater V3 applicability prompts.
- `leuven_expansion/prompts/V3_1_PROMPT_RATIONALE.md` - Best direct account of the intended V3-to-V3.1 prompt manipulation.
- `leuven_expansion/prompts/feature_generation_prompt_*_v4.txt` - Exact high-recall V4 discovery ensemble.

### Evaluation/results

- `ISC-CI_LLM_validation/iscci_validation/training.py` - Exact reconstructed training loop and checkpoint metadata.
- `ISC-CI_LLM_validation/iscci_validation/evaluation.py` - Fixed probes and representational comparison metrics.
- `ISC-CI_LLM_validation/iscci_validation/simulations.py` - Reimplementation of the paper simulations.
- `ISC-CI_LLM_validation/reports/ceiling_summary.csv` - Human self-ceiling and condition-to-human representational metrics.
- `ISC-CI_LLM_validation/reports/paper_simulation_summary.csv` - Human/V2/V3 paper-simulation summaries.
- `ISC-CI_LLM_validation/reports/v3_1/` - Cumulative V3.1 matrices, deltas, ceilings, training summaries, and simulations.
- `artifacts/v4/judgments/calibration_summary.csv` - All V2-derived threshold candidates and held-out metrics.
- `artifacts/v4/retrieval_efficiency/v2_retrospective/` - K-shortlist development/held-out results and audit simulation.
- `artifacts/v4/retrieval_efficiency/prompt_c_cascade/RESULTS.md` - Cascade cell, geometry, ISC-CI, and simulation fidelity summary.

### Figures

- No tracked generated result figures were found in the repository. The reports are table-based.
- `ISCCI_2026Jan_ArXiv.pdf` contains the source paper's model and result figures but is not a generated artifact of the LLM-validation project.

### Other useful documentation

- `artifacts/v4/todo.md` - Authoritative checklist of completed and pending V4 stages.
- `artifacts/v4/run_manifest.json` - Orchestration state as of 2026-08-24.
- `artifacts/v4/reports/report_manifest.json` - Hash-linked evidence that primary V4 result tables were still unavailable when the report was generated.
- `pyproject.toml` and `tests/` - Environment and test coverage for the current implementation; useful for software reproducibility but not substitutes for scientific result artifacts.
