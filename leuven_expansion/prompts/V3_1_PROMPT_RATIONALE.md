# Leuven Feature Generation V3.1

V3.1 retains the published free feature-generation task while testing three mechanisms:

- **A, faithful:** close control for the published Leuven instructions.
- **B, first-to-mind:** discourages systematic, optimized, expert-style lists and preserves
  production order.
- **C, individual participant:** discourages the model from approximating an aggregate answer
  and permits natural participant-to-participant differences without requesting novelty.

## Why Change V3

Qwen V3 returned exactly 10 features for every B/C response and for 5,856 of 5,860 A
responses. Exact phrase types were A=15,977, B=14,669, and C=15,294. Prompt A had the most
surface diversity but retained only 94 ISC-CI tasks; B retained 115 and C retained 121. Prompt
C had the strongest broad behavioral validation profile, but all prompts repeated polished
generic templates and remained outside the human model's run-to-run ceiling.

The V3.1 prompts keep "preferably 10" because this was part of the human task. They instead
remove selection language such as "the clearest" and make spontaneous order, stopping, and
individual rather than aggregate responding explicit. They do not request creativity, rare
features, or non-overlap with previous responses because those instructions would destroy the
meaning of independent production frequency.

## Required Pipeline Distinction

The original Leuven model matrix was not a thresholded free-generation matrix. The published
procedure was:

1. At least 20 participants generated features for each word.
2. Responses were manually normalized, with synonyms and inflectional variants merged,
   redundant quantifiers removed, and multi-property responses split where appropriate.
3. Features generated at least four times across a semantic category were selected.
4. Separate participants judged every selected feature's applicability to every exemplar in
   the relevant category/domain; four raters completed each domain matrix.

V3 skipped step 4 and treated generation recurrence as applicability. V3.1 generation should
therefore be evaluated both as a production norm and as the inventory-building phase of a
separate applicability experiment. Prompt improvement cannot substitute for that phase.

## Sampling Recommendations

- Keep at least 20 independent responses per word and prompt for direct continuity; use 30-40
  when cluster cost permits because newer models may still have concentrated response modes.
- Compare model families, not just seeds from one model. Between-model diversity is part of
  the sensitivity design, but each model must be reported separately before pooling.
- Preserve paired seeds across A/B/C and use the same decoding settings within a model.
- Start with temperature `0.8` for continuity, then smoke-test `1.0` as a declared decoding
  sensitivity. Do not select temperature after observing ISC-CI validation results.
- Do not show previous responses or ask later calls to avoid duplicates. That would make calls
  dependent and invalidate production frequencies.
- Preserve response order (`feature_rank`) for checking whether later positions become generic
  filler.

Primary methodological source: De Deyne et al. (2008), *Exemplar by feature applicability
matrices and other Dutch normative data for semantic concepts*, Behavior Research Methods,
40, 1030-1048, https://doi.org/10.3758/BRM.40.4.1030.
