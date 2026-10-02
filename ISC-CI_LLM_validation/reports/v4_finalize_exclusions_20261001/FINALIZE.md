# V4 Finalization With Declared Failures: 2026-10-01

## Investigator Decision

The investigator explicitly changed the remaining-retry plan: the 127 unresolved
adjudications may remain failed, and V4 should be finalized without another GPU
retry run. This supersedes the retry requirement in the earlier recovery guide.
No human behavioral results were used to select these exclusions.

## Policy

- Saved-response recovery still resolves the 1,878 exact-ID formatting cases,
  preserving original CSVs and raw responses. The finalize job applies that plan
  automatically when it is present, inside the same CPU allocation.
- Finalization accepts at most **127** missing scores explicitly recorded as
  `adjudicator_failed`. It does not accept out-of-range scores, missing resolution
  rows, duplicates, wrong words/IDs/shards, changed protocol hashes, or invalid
  first-pass vote coverage.
- Failed cells are saved in `artifacts/v4/judgments/unresolved_cells.csv` with
  candidate ID, object, feature, shard, and reason. They are not zero-filled and
  are absent from the valid `resolved_feature_values.csv`.
- To keep training matrices genuinely complete, the matrix builder excludes each
  **entire candidate column** with any failed cell. The current 127 cells affect
  **64 of 133,081 candidates**; 133,017 candidates remain before the unchanged
  positive-object retention rule. The 64 exclusions overlap **zero** fixed-B
  candidates, so that control remains the original 175-feature inventory.
- The frozen candidate bank is unchanged. Every valid judgment is preserved in
  shard files and the valid consolidated values, including known values for
  excluded candidates. Those columns are omitted only from training matrices.
- The judgment manifest lists counts, excluded candidate IDs, failure-audit hash,
  and policy. `complete: true` means finalization outputs are accounted for;
  `judgments_complete: false` and
  `status: finalized_with_unresolved_exclusions` explicitly disclose failed
  judgments. Matrix manifests carry the same exclusions and any fixed-B overlap.
- Original shard sidecars remain unchanged. Their `complete: false` and old
  invalid-score counters may be stale after recovery. The opt-in finalizer checks
  actual CSV values/keys/votes and the frozen partition, not those flags alone.
- The generic Python finalizer remains strict by default. The Slurm finalization
  launcher now passes `--max-unresolved-cells 127`. Its CPU resources are unchanged,
  and it never loads a model or requests GPUs.

## One Cluster Job

The GitHub deployment includes a small, tracked recovery-input archive in
`configs/v4_exact_id_recovery_inputs.tar.gz`. The finalize launcher unpacks it
into the recovery directory only when the plan is absent. No large repaired CSVs,
backups, or unrelated human-behavior changes are required for this deployment.
The normal path from the cluster project directory is:

```bash
git pull --ff-only
sbatch run_leuven_v4_atomic_finalize.sh
```

The portable bundle remains an alternative:

Transfer the new `cluster_finalize_bundle.tar.gz`, enter the cluster project
directory, and extract there. This bundle contains the changed finalizer,
matrix-builder/reporting code, finalization Slurm launcher, recovery helper,
original small recovery plan/evidence, and this guide. It excludes large local
CSV backups and application journals.

```bash
tar -xzf /path/to/cluster_finalize_bundle.tar.gz
sbatch run_leuven_v4_atomic_finalize.sh
```

No production-array resubmission or separate recovery job is needed. Do not run
any shard writers concurrently. Apply to the original cluster CSVs (or a cohort
already repaired with its matching application journals), rather than syncing
locally repaired CSVs alone and leaving the recovery journal behind.

If the cluster CSVs already contain exactly the remaining 127 failures and the
formatting-recovery plan is absent, the finalizer can proceed directly. If they
still contain all 2,005 failures and no plan is supplied, the 127-cell cap correctly
blocks finalization. Do not increase the cap to silently discard unrecovered
formatting cases. The bundled plan is hash-checked against the audited original
cohort before application. A source mismatch remains an error.

## Verification and Scope

Focused tests cover strict-default rejection, accepted failed adjudications,
audit-file integrity, no zero filling, whole-column removal through the full
matrix builder, stale completeness flags, cap overrun, out-of-range rejection,
and protocol mismatch. Original cross-product, cascade, matrix, and reporting
regressions are also run. All model calls in tests are mocked.

Verification: **58 focused tests passed**, shell syntax validation passed, and
the changed tracked code passed `git diff --check`.

The production finalization is deliberately left for the cluster. This update
does not claim V4 is already finalized or trained locally. The earlier full-data
QA verified 1,878 repairs, 127 remaining failures, all 38,992,733 resolution rows,
and original/updated source hashes. The present amendment changes how those
remaining failures are handled, not the judgments themselves.
