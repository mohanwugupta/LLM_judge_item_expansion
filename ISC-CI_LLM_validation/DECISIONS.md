# ISC-CI LLM Validation Decision Log

## 2026-08-08: Validate V3.1 as a Prompt-Only Follow-Up

### Decision

Treat V3.1 A, B, and C as three new ISC-CI conditions and compare each with the corresponding
V3 prompt, V2, retrained human models, and released human checkpoints. Hold every
consolidation, training, fixed-context evaluation, and paper-simulation parameter equal to the
locked V3 protocol. Do not retune thresholds after observing V3.1.

### Source and Integrity

The source is
`artifacts/leuven_feature_generation/v3.1/leuven_v3.1_qwen2_5_72b/generated_features_long.csv`.
Its production manifest records Qwen2.5-72B-Instruct, 293 words, prompts A/B/C, 20 simulated
participants per word and prompt, temperature 0.8, paired seeds, 17,580 of 17,580 valid
responses, and zero parse or request errors. Consolidation uses the same strict
`response_count > 3` and `positive_objects > 3` cutoffs as V3.

`configs/v3_1_validation.json` deliberately duplicates every substantive value in
`configs/v3_validation.json`; only the protocol identifier differs. The comparison therefore
tests the V3.1 prompts, not a changed threshold or training procedure.

### Consolidation Review

Prompt variants remain separate. At the primary threshold, five semantic/profile clusters
would be retained and all were reviewed. Four were equivalent and passed. The proposed merge
of `driven on highways` with `driven on roads` was rejected because a highway is a narrower
road type. The consolidation code splits rejected semantic clusters back into lexical groups
before response counts and the training cutoff are calculated. The broader `driven on roads`
feature remains a retained singleton; the highway feature does not meet the task cutoff.

The final V3.1 task counts are A=88, B=175, and C=128. These counts and positive-cell totals
are identical at embedding thresholds 0.80, 0.825, 0.85, 0.875, and 0.90, so the result is not
driven by the selected embedding threshold.

### Reuse and Provenance

`artifacts/validation_v3/` remains unchanged. The V3.1 run copies it to
`artifacts/validation_v3_1/`, preserves its prior manifest as
`base_validation_manifest.json`, and reuses checkpoints only when matrix hash, epoch count,
and training parameters match. It then trains 12 V3.1 checkpoints (A/B/C by seeds 0-3),
evaluates all conditions on the same fixed contexts, and runs the paper simulations only for
models without completion artifacts. The cumulative manifests hash both consolidation stages
and the copied base validation.

### Interpretation Constraint

V3.1 is a prompt iteration on the same model family, not an independent replication. Report
prompt-matched descriptive deltas and four-seed variation, but do not treat the 12 V3.1 runs as
independent evidence about model-family generalization. The same no-applicability-judgment
limitation documented for V3 also applies to V3.1.

## 2026-08-04: Consolidate V3 Feature Generations

### Decision

Treat prompt variants A, B, and C as separate feature-norm conditions. Consolidate
near-equivalent feature phrases within each prompt before constructing the ISC-CI training
matrix. Do not pool responses across prompts.

### Why

V3 recreated the Leuven free feature-generation task. The released Leuven matrix used by
ISC-CI is already consolidated, whereas the V3 artifact stores exact normalized strings and
explicitly states that semantic merging was not performed. Treating every wording variant as
a distinct feature would therefore create an avoidable mismatch with the human pipeline.

### Operational Definition

1. Begin with `generated_features_long.csv`; one valid model response is treated as one
   simulated participant response.
2. Use the supplied normalized feature text as the phrase type.
3. Within each prompt, merge deterministic lexical variants and conservatively merge
   semantic near-duplicates using normalized `all-MiniLM-L6-v2` embeddings plus the
   similarity of their 293-object generation profiles.
4. Require every cross-pair in a proposed multi-phrase cluster to satisfy the semantic and
   profile criteria. This complete-link constraint prevents transitive chaining.
5. Block embedding-only merges when one phrase adds a substantive modifier to the other
   (for example, `has a tail` versus `has a long tail`). Lexically equivalent signatures may
   still merge.
6. Ignore non-substantive frequency scaffolding (`often`, `usually`, `typically`, and similar
   terms) when forming lexical signatures, but preserve conjunctions and substantive
   alternatives. Thus `metal and wood` remains distinct from `metal or wood`, and `made of
   wool` remains distinct from `made of wool or silk`.
7. Do not use the English sentence-embedding model to merge phrases containing CJK text.
   Such phrases retain their exact-string identities and are flagged in the assignments.
8. If one response contains two phrases assigned to the same cluster, count the cluster once
   for that response.
9. Construct object-by-cluster response counts, binarize using the paper's strict
   `response_count > 3` rule, and retain clusters positive for strictly more than three objects.
10. Train separate ISC-CI models for A, B, and C using the released architecture, frozen
   semantic embedding, episode sampler, loss, optimizer, 400 epochs, and seeds 0-3.

### Primary and Sensitivity Specifications

The primary embedding threshold is `0.85`; object-profile cosine must be at least `0.50`.
Embedding thresholds `0.80`, `0.825`, `0.85`, `0.875`, and `0.90` are reported as a
sensitivity analysis with the profile threshold held fixed. The exact parameters live in
`configs/v3_validation.json`.

The automated method is intentionally conservative. It is a reproducible approximation to
the human consolidation applied before release of the Leuven matrix, not a claim that sentence
embedding proximity proves logical equivalence. Every assignment, cluster member, within-
cluster similarity, source hash, and model revision is written to the consolidation artifact.
This allows a future manual or LLM-adjudicated consolidation to replace the current mapping
without changing the downstream training protocol.

All embedding/profile clusters retained by the primary training cutoff were manually reviewed
and recorded in `configs/v3_consolidation_manual_review.csv`. The consolidation command
fails if that reviewed set no longer exactly matches the retained semantic clusters. During
method development, the review caught and rejected a merge of `made of wool` with `made of
wool or silk`; preserving conjunctions and substantive qualifiers prevents that merge in the
locked implementation.

Nearest-neighbor candidates are computed by exact, chunked PyTorch cosine products. An
initial FAISS implementation was rejected after the locally installed native binary crashed
on macOS (`exit 139`); no FAISS-derived assignments were produced or retained.

### Revisit Triggers

- Manual review finds systematic over-merging or under-merging.
- An LLM adjudication run becomes available for candidate phrase pairs.
- The original Leuven consolidation instructions or unconsolidated source become available.
- Validation conclusions change materially across the recorded threshold sensitivity range.

## 2026-08-04: Follow Executed Checkpoint Batch Size

The released checkpoint dictionary records `batch_size=1024`, but every epoch in its retained
metrics has batch indices 0-7 and accuracy increments of `1/128`. Thus the model actually
executed eight batches of 128 examples over each 1,024-episode epoch. Retraining follows the
executed behavior (`batch_size=128`) rather than the inconsistent metadata field. It uses 400
epochs, matching the released checkpoint's `epoch=400`, for 409,600 sampled episodes. The
paper describes this approximately as 500,000 episodes.

## Reconstructed V2 Rule

V2 remains locked to the 385 features retained by the original human Leuven preprocessing.
The finalized adjudicator value is converted directly to binary (`0` is absent; any nonzero
value is present), with no additional rating or object-frequency threshold. Three human-schema
features have no V2 positives and cannot generate support episodes, leaving 382 V2 tasks. This
reproduces the retained prior run log (`human_tasks=385`, `llm_tasks=382`) and the user's
instruction that the adjudicator decision is final.

## Recovery Note

The earlier validation source and reports disappeared from this Git-ignored directory before
the V3 analysis. Reconstruction uses the clean upstream repository snapshots, released model
checkpoint metadata, the retained V2 run log, and surviving compiled ceiling modules. New
runs record source hashes and protocol manifests so this can be detected and audited.

## 2026-08-04: Define Comparability Against a Human Self-Ceiling

Comparability is evaluated on fixed task-label-independent probes rather than each model's
native training accuracy. The evaluation uses all 293 singleton supports, 1,024 sampled
unordered support pairs, and all 293 real query objects for every support context. Context
RDMs use 512 fixed contexts and context-dependent RDMs use 128 fixed contexts. The sampling
seed is `20260804`, and the same probes are used for every model.

Five measures are retained: context RDM Spearman correlation, median context-dependent RDM
Spearman correlation, membership-logit Spearman correlation, binary membership agreement,
and membership-probability mean absolute error. Human self-ceiling values are calculated from
all pairs of the four independently retrained human models. Candidate-to-human values average
all 16 pairs between a candidate condition's four seeds and the four retrained human seeds.
The four released checkpoints are also compared with all four retrained human models as an
independent reconstruction check.

Binary agreement is not interpreted alone because the V3 matrices are sparse. Input-matrix
object RDMs and the continuous/rank output measures are reported to distinguish shared
negative decisions from genuinely similar representational geometry. These choices are
implemented in `iscci_validation/evaluation.py`; exact probes and model outputs are retained
under `artifacts/validation_v3/evaluation/`.

## 2026-08-04: Recreate the Paper Simulations

The analyses in the two released notebooks were moved into a deterministic command-line
runner without changing their substantive calculations. It evaluates every human, V2, V3,
and released checkpoint on the five induction datasets, seven induction phenomena, nine
similarity domains, asymmetric similarity, thematic and context-dependent paired choices,
and the similarity-context LCA simulation.

Necessary execution adaptations are limited to explicitly selecting numeric columns for
pandas 2, mapping behavioral labels through `leuven_singular_to_plural.csv`, and assigning
stable per-model/per-row random seeds to the stochastic LCA. The LCA retains 100 simulations,
500 steps, and a 100-step burn-in. Per-model outputs and completion records are preserved
under `artifacts/validation_v3/paper_simulations/`; the manifest records notebook paths and
adaptations.

Prompt C is the current broad default because it has the strongest overall V3 profile, not
because it wins every benchmark. Prompt B is better for asymmetric similarity, while prompt A
is better for thematic arguments and slightly better for context-dependent non-monotonicity.
This choice must be revisited if additional participants, models, prompts, consolidation
rules, or behavioral benchmarks materially alter that profile.

## 2026-08-04: Audit the V3 Consolidation Rules

A post-validation diagnostic sweep compared exact-only, conservative lexical, current
lexical, stricter, primary, and relaxed embedding/profile rules. It is recorded under
`reports/consolidation_audit/` and does not modify the locked primary matrices or models.

The embedding threshold is not responsible for the small V3 task inventory. With the current
lexical signatures, removing semantic merges leaves A/B/C at 94/115/121 tasks, identical to
the primary counts. Embedding thresholds from 0.80 through 0.90 also leave those task counts
unchanged when the profile threshold remains 0.50. The semantic/profile rule contributes only
two primary positive cells, both in C.

The 0.50 profile guard remains necessary. Relaxing or removing it admits non-equivalent and
sometimes contradictory pairs. The rule requiring revision is automatic lexical merging:
it can erase modality and frequency distinctions or conflate derivational roles. The clearest
retained error pools `can be washed` with `washed frequently`; another cluster pools `musical
instrument` with `used for musical instruments`, although that second error is currently
subthreshold and does not change a positive object.

Therefore, keep the current embedding/profile/complete-link safeguards, but do not treat the
primary lexical consolidation as final. A follow-up should use safe inflectional normalization
plus manual adjudication for modality, frequency, and derivational changes, then retrain those
matrices as a reported sensitivity condition without replacing the original primary results.
