# Leuven v3 free feature generation

## Experimental contract

V3 recreates the first, free-generation stage of the Leuven norming study. It
does not reconstruct cells from the existing exemplar-by-feature matrix.

The original procedure asked each participant to provide preferably 10
different features for each stimulus word. Instructions requested a mixture of
physical or perceptual features, functional features, and background
information. Participants could return fewer than 10 when they could not think
of more. At least 20 participants generated features for every word.

Source: De Deyne et al. (2008), *Exemplar by feature applicability matrices and
other Dutch normative data for semantic concepts*, Behavior Research Methods,
40, 1030–1048, doi:10.3758/BRM.40.4.1030.

The human task used paper booklets containing 6–10 one-word response sheets.
V3 preserves the one-sheet response unit as one independent model call. A call
receives only the stimulus word and returns zero to ten freely generated
English feature phrases. Existing Leuven feature columns, category metadata,
human values, other words, and downstream model information are not sent.

The initial experiment uses one model and three prompt conditions:

- **A, original-style:** closely follows the original Leuven written task.
- **B, concise:** gives the same task with minimal instruction scaffolding.
- **C, structured:** asks the model to consider perceptual, functional, and
  background perspectives before selecting its response.

All three conditions have the same stimulus message, maximum of 10 features,
JSON schema, model, sampling parameters, and response count. They differ only
in the free-generation instructions. The output records `prompt_variant` on
every response and feature row so conditions can be evaluated separately.

## Sampling design

- Models per run: 1
- Prompt conditions: A, B, and C
- Default simulated participants per word per prompt: 20
- Default temperature: 0.8
- Default base seed: 20260801
- Sampling seed: stable hash of base seed, word, and replicate number
- Prompt comparison: the same seed is paired across A/B/C for each word and
  replicate
- Resume key: word, prompt condition, replicate, model, sampling seed, and
  prompt hash
- Semantic synonym merging: deliberately not performed during collection

The nonzero temperature is required when collecting multiple responses. The
client cache includes the sampling seed, so repeated prompts do not collapse to
one cached answer.

## Cluster submission

The dedicated launcher submits only v3:

```bash
sbatch run_leuven_v3_generation.sh
```

Defaults produce 20 responses for every item under each of the three prompts
using one Qwen2.5-72B-Instruct model. For the 293 Leuven words, this is 17,580
calls. Submit only after the smoke test passes. To run a different model or
sampling configuration later, provide a unique run label and model
configuration through exported variables:

```bash
sbatch --export=ALL,V3_RUN_LABEL=my_model,V3_RESPONSES_PER_WORD=20,V3_TEMPERATURE=0.8,MODEL_DIR_NAME=model-directory,SERVED_MODEL_NAME=served-model run_leuven_v3_generation.sh
```

`MODEL_PATH` can also be supplied explicitly. Each `V3_RUN_LABEL` writes to a
separate directory under:

```text
artifacts/leuven_feature_generation/leuven_v3_<run label>/
```

The input CSV must contain the actual Git LFS content on the cluster. The runner
fails before starting vLLM if it sees an LFS pointer. The launcher also executes
a no-model preflight that validates the response count, unique seeds, prompt and
schema hashes, and any existing resume manifest before loading the model.

## Cluster smoke test

Run this first:

```bash
sbatch run_leuven_v3_smoke_test.sh
```

The smoke test uses the same model server, prompts, schema, runner, output
derivation, and completion checks as the full experiment. It selects three
evenly spaced words from the input and collects two responses under each prompt,
for 18 calls total. Its default output is isolated by SLURM job ID:

```text
artifacts/leuven_feature_generation/leuven_v3_smoke_<job id>/
```

The job succeeds only when all expected A, B, and C responses are valid and all
six output files exist. `V3_MAX_WORDS`, `V3_RESPONSES_PER_WORD`, and
`V3_RUN_LABEL` can be overridden with `sbatch --export=ALL,...` when repeating a
targeted smoke run.

## Outputs

- `feature_generations.csv`: one preserved response per word, prompt, and
  replicate
- `generated_features_long.csv`: one row per generated feature token, labeled
  by prompt
- `generated_feature_frequencies.csv`: exact normalized-string frequencies
  computed separately by word and prompt
- `parse_errors.csv`: invalid or failed responses for resumable retry
- `manifest.json`: locked prompt texts and hashes, schema, model, seed, selected
  words, and run configuration
- `run.log`: collection log

Exact-string frequencies are an audit aid, not the final human comparison.
Following the original study, synonym consolidation, minimal stemming, removal
of redundant quantifiers, and splitting conjunctive features require a separate
documented preprocessing stage after the raw responses are frozen.

Prompt quality should ultimately be ranked by recovery of the human feature
norms after that common preprocessing/matching stage, not by raw feature count.
Keeping all three prompt conditions in one run prevents model configuration or
input ordering from becoming a prompt-condition confound.

## Protocol guard

`leuven_expansion.validate_features` now accepts only v2 atomic prompts. Passing
v3 to that word-by-feature pipeline is rejected. V3 must use:

```bash
python -m leuven_expansion.generate_features --help
```

This prevents the previous applicability task from being mislabeled as v3.
