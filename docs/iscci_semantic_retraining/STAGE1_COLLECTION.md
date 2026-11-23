# Full-Domain V2 and V3 Collection

This is **collection only**, not training. It implements the collection portion
of `STAGE1_DATASET_CORRECTION_PLAN.md` in a new namespace. Existing scientific
code, 32-class labels, feature matrices, checkpoints and results are unchanged.
Both V3 and V3.1 A/B/C were explicitly requested. No V4 ensembles are collected.

## What Is Ready and What Still Needs Input

The smoke test can run immediately after setup. Production intentionally refuses
to spend GPU calls until its input contract is satisfied:

1. `inputs/objects.csv` has all **350 original objects**, size/category metadata,
   and 350 distinct display names. 266 independently verified old identities and
   58 original fruit/vegetable names have source evidence. **26 names need review**,
   including numbered homonyms and original category conflicts. Review/edit
   `display_name`, set `identity_reviewed=true`, and record `identity_evidence`.
   Do not merge object IDs, infer identities from nearest embeddings, or change
   original names/category/size/index. Separate concepts need distinct, meaningful
   display names. Do not show auxiliary categories to Qwen as extra context.
2. V2 additionally requires the **original 2541 feature descriptions**. The file
   `inputs/v2_features.csv` has the correct original zero-based feature/task IDs,
   but descriptions are deliberately blank. Fill `feature_text`, `verified=true`
   and `evidence` using a verified original mapping. Do not paste the downstream
   1995 features into this template by position or ask Qwen to invent the mapping.
   This unresolved dictionary is a genuine V2 prerequisite; the code does not
   claim the production V2 run is presently ready.

V3 production does **not** depend on the V2 dictionary. Once object names are
reviewed it can proceed independently. Review inputs before the first production
request; changing scientific inputs afterwards requires a new output version.
`prepare` never overwrites an existing edited input file.

## Implementation

Entry point: `python -m leuven_expansion.stage1_collection`.
GPU launcher: `run_stage1_collection.sh`. It copies the established Conda/vLLM
setup: Qwen2.5-72B, four 80-GB GPUs, 8 CPUs, 32GB host RAM, TP4, 24-hour limit.
No runtime estimate or convergence claim is implied by that walltime.

Default output: `artifacts/isc_stage1_collection_20261122_v1/`.
New inputs are derived from the pinned original ISC source and are included as
small tracked templates under `configs/stage1_collection/`. Collection does not
require restoring ignored old training artifacts or passing legacy training gates.

- **V3 generation:** frozen original V3 and V3.1 A/B/C prompts, 20 responses per
  object per prompt, temperature .8, seed20260801 and existing paired seed rule.
  350 x 6 x 20 = **42000 response units**. No human feature schema is shown.
- **V2 applicability:** existing independent A/B/C calls and original V2
  adjudication/resolver, without a new cascade or positive-verification filter.
  350 x 2541 = **889350 cells**, at least 2668050 first-pass calls plus retries
  and any three-call adjudication panels. Scores and review flags are preserved;
  binary export uses the existing approved `final_feature_value > 0` rule, not
  the human frequency cutoff. Original feature task IDs remain unchanged.
- **V3 task classification:** after generation is finalized, classify distinct
  exact-phrase candidates once, not once per object. The new 33-class collection
  reconstruction restores Associated object as collection ID32. **This is not
  proof of historical numeric task correspondence.** Outputs are provisional,
  carry uncertainty/out-of-inventory flags, and do not authorize training.
- **No hidden retention rule:** generation finalization performs only the existing
  transparent phrase normalization. It preserves raw text, membership, version,
  prompt and participant ID, with no `>3` response/object filter or embedding
  merge. All six conditions remain identifiable. Final semantic consolidation,
  source-only matrices and original-grouping verification are still separate
  steps after collection. These are candidate labels, not final training matrices.

Collection is a fresh, revision-fingerprinted full-domain run. **It does not
automatically import old responses**, whose actual weight revision was not
verified. This costs more than an approved reuse run but avoids silently mixing
unverified model versions. Existing code/prompts are reused; future old-data
reuse would need explicit identity, prompt and weight-provenance checks. The
previous 84-row estimates were conditional reuse workloads, not this run's cost.

## Cluster Setup and Smoke Test

These are commands to run after this code has been transferred/pushed to the
cluster checkout. Nothing is submitted by local preparation. Activate the same
`PromptControlText` environment used by the working jobs. The collection requires
`numpy`, `pandas`, `jsonschema`, `openai`; local model fingerprinting reuses an
existing repository helper. GPU-serving packages stay in the existing cluster
environment. Dependency check:

```bash
python -c 'import numpy, pandas, jsonschema, openai'
mkdir -p /scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs
python -m leuven_expansion.stage1_collection --stage prepare
sbatch --export=ALL,COLLECTION_STAGE=smoke run_stage1_collection.sh
```

Smoke uses bee, monkey and apple: one participant per each of six prompt
conditions, six synthetic applicability cells, and two synthetic task labels.
These are API/schema checks, not human behavioral tests or label-accuracy gates.
It writes only `smoke/` and `smoke_result.json`, never production judgments.
It may run before identity/dictionary review; passing smoke does not waive them.
The model fingerprint hashes actual local weight shards. Smoke has to pass with
the same fingerprint/source before production starts.

Inspect remaining prerequisites on the login node, with no model calls:

```bash
python -m leuven_expansion.stage1_collection --stage preflight --task generate
python -m leuven_expansion.stage1_collection --stage preflight --task judge
```

An unready preflight prints the blockers and exits2. `prepare` prints readiness
but exits0 because the review templates were successfully created.

## Production

After reviewing the 26 object identities, submit generation. Four zero-based
shards with `%1` run one four-GPU job at a time; increase concurrency only when
appropriate for the cluster allocation:

```bash
sbatch --array=0-3%1 --export=ALL,COLLECTION_STAGE=generate run_stage1_collection.sh
```

V2 can run separately **after** the feature dictionary passes preflight:

```bash
sbatch --array=0-31%1 --export=ALL,COLLECTION_STAGE=judge run_stage1_collection.sh
```

The shard count is derived from `SLURM_ARRAY_TASK_COUNT`; array IDs must be the
complete zero-based range. No stale fixed32 setting is inherited. Keep the
same shard count for retries and resubmit only failed/interrupted indices. For
example, retry V2 indices2and7 with the **original**32-way partition:

```bash
sbatch --array=2,7%1 --export=ALL,COLLECTION_STAGE=judge,SHARD_COUNT=32 run_stage1_collection.sh
```

Completed requests are skipped after protocol/payload verification. Failures
remain pending and are retried on the next invocation. A timeout does not require
starting the completed requests again. Each shard has its own transactional
SQLite database; no shared WAL database across GPFS workers. Do not run two
processes for the same shard. Completed shards still incur server startup if
resubmitted, but perform no new requests. Save GPU time by resubmitting only
unfinished indices. Changed inputs/model/prompts or shard counts fail closed.
Model fingerprinting scans the actual weight files and can take time.

Inspect committed progress without loading Qwen:

```bash
python -m leuven_expansion.stage1_collection --stage status
```

Per-shard `status.json` is updated during collection; it stays incomplete until
the full assigned unit stream finishes. Status reports committed successes and
errors, not inferred negatives. Error details are retained in the shard database
and raw request journal.

After **all** generation shards succeed, finalize on a CPU/login session:

```bash
python -m leuven_expansion.stage1_collection --stage finalize --task generate
sbatch --array=0-3%1 --export=ALL,COLLECTION_STAGE=classify run_stage1_collection.sh
```

After all classifier/V2 shards succeed:

```bash
python -m leuven_expansion.stage1_collection --stage finalize --task classify
python -m leuven_expansion.stage1_collection --stage finalize --task judge
```

Finalization rejects missing/incomplete/misassigned shards, altered databases,
duplicate/missing units, protocol mismatches and insufficient global coverage.
Completed output CSVs are checksummed and not overwritten. Invalid/truncated
responses remain in the raw request journal and are never converted to zero.
Uncertain valid labels and provisional adjudication decisions are exported with
their flags; collection completion is not scientific readiness or human review.

## Bring Results Back

Stop/wait for all collection jobs before exporting. The archive includes input
reviews, protocol/model fingerprints, raw request logs, shard databases, statuses,
and any finalized results, not Qwen weights or training checkpoints. Partial
exports are permitted but explicitly identify incomplete stages.

```bash
python -m leuven_expansion.stage1_collection --stage export \
  --archive artifacts/isc_stage1_collection_20261122_v1_export_01.tar.gz
python -m leuven_expansion.stage1_collection --stage verify-export \
  --archive artifacts/isc_stage1_collection_20261122_v1_export_01.tar.gz
```

Use your existing cluster transfer method to download that archive into the
local `artifacts/` directory, retaining its filename. Run the same `verify-export`
command locally before inspecting/extracting it. It checks every member/hash and
rejects traversal paths or symlinks. Do not force-add raw outputs to ordinary Git;
the archive is already under the ignored artifacts directory. A future LFS input
bundle can be made after review. Never extract over historical artifacts.

There is no automatic ISC/ISC-CI training command or submission dependency here.
After the data return, validate coverage, consolidation and task grouping against
the Stage 1 correction plan before training anything.
