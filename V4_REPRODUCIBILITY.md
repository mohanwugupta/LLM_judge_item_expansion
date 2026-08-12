# V4 Reproducibility Record

## Scope

V4 combines liberal V3.1-style feature discovery with exhaustive V2 atomic applicability judgments. Generation nominates candidate propositions; only the atomic panel determines candidate-by-object cells. The primary condition is never retrieval-pruned.

## Frozen decisions

- Leuven vocabulary: the existing ordered 293-word human matrix.
- Existing discovery sources: completed V3 and V3.1 Qwen2.5-72B outputs, linked by manifests and hashes.
- New discovery: seven independently configured high-recall prompt families, 20 responses per word, at most 10 propositions per response.
- Fixed-vocabulary control: the exact ordered 175 retained V3.1-B clusters, with a separate inventory hash.
- Candidate inclusion: valid singletons and rare phrases remain eligible; no `response_count > 3` cutoff before atomic judging.
- Candidate consolidation: lexical/embedding/profile rules nominate possible merges. Every multi-phrase merge needs an explicit `pass` or `reject`; similarity never authorizes a merge by itself.
- Atomic panel: the existing V2 A/B/C prompts, Qwen2.5-72B model identity, temperature 0, adjudication logic, and ordinal 0-4 resolver.
- Independence: atomic calls contain one target word and one candidate. They exclude source words, generation frequency, prompt/model provenance, other candidates, human values, and downstream results.
- Locked decision rule: `resolved_value > 0`.
- Calibrated decision rule: `resolved_value >= 1`, frozen before V4 judgments or model evaluation.
- Context retention: `positive_object_count > 3`, applied after exhaustive judging.
- ISC-CI: frozen semantic embeddings, existing architecture and sampler, 1,024 episodes per epoch, batch size 128, 400 epochs, seeds 0-3, evaluation seed 20260804, and unchanged paper simulations.
- Efficient retrieval: posthoc only. Features are split into development and held-out test sets before selecting K.

## Calibration result

Calibration used all 112,805 completed V2 cells for 293 words and the 385 retained human features. Splits were five-fold by word. The selected `>=1` rule met the recall gate:

| Metric | Mean held-out value |
| --- | ---: |
| Positive precision | 0.2452 |
| Positive recall | 0.8542 |
| Positive F1 | 0.3810 |
| Balanced accuracy | 0.8260 |
| MCC | 0.3884 |
| Matrix density | 0.2487 |
| Word-vector cosine | 0.4527 |
| Human input-object RDM correlation | 0.8221 |

The frozen machine-readable decision is `artifacts/v4/judgments/judgment_threshold.json`.

## V2 retrieval retrospective

The originally proposed K values 25, 50, 75, and 100 did not meet the 95% positive-recall and 0.99 geometry gates. On held-out features, K=100 retained 74.2% of V2-positive cells and had object-geometry correlation 0.925. A configured automatic escalation selected K=275 on development features.

On untouched held-out features, K=275:

- retained 98.24% of V2-positive cells;
- retained 100% of Leuven-positive cells, which is expected because original human source words are always forced into the V2 retrospective shortlist;
- preserved object geometry at 0.9941 and feature geometry at 0.9976;
- shortlisted 94.47% of cells, only a 5.53% initial-judgment reduction.

Using judge C as the cheap first-stage judge and routing positive, ambiguous, or confidence-below-.80 cells to the full panel retained 97.89% of V2 positives, preserved object geometry at 0.9935, and reduced estimated calls by 45.89%. This is promising for a cascade, but it must be rerun against the completed V4 matrix before being recommended. Call fractions are token/cost/runtime proxies because V2 did not preserve per-request telemetry.

## Current blocker

The seven-family V4 discovery job must run before the pooled bank can freeze. The imported V3/V3.1 candidate pool produced a preliminary 11,003 proposed multi-phrase merges, but this list will change when V4 generation is imported and should not be reviewed yet. After generation, rerun the bank builder; it will regenerate the proposals and populate `configs/v4_candidate_merge_review.csv` for explicit `pass` or `reject` decisions. The exact V3.1-B 175 bank is already exported for smoke testing.

## Execution order

```bash
sbatch run_leuven_v4_generation.sh
python build_v4_candidate_bank.py --config configs/v4_discovery.json --manual-review configs/v4_candidate_merge_review.csv --output-dir artifacts/v4/discovery
python run_v4_judgments.py --candidate-bank artifacts/v4/discovery/candidate_bank.csv --leuven-words data/leuven_combined_features_consolidated.csv --v2-manifest artifacts/leuven_full_labels/leuven_full_v2/manifest.json --output-dir artifacts/v4/judgments --shard-count 32 --dry-run
sbatch run_leuven_v4_atomic_smoke_test.sh
sbatch run_leuven_v4_atomic.sh
sbatch run_leuven_v4_atomic_finalize.sh
python run_v4_pipeline.py --config configs/v4_validation.json --resume
```

## Traceability

`run_v4_pipeline.py` writes the resolved configuration, source hashes, stage manifest, log, generated report, and reproduction status under `artifacts/v4/`. Matrix and model stages fail closed on bank, judgment, calibration, semantic-embedding, or training-configuration hash mismatches.
