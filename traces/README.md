# Complete Reviewer Traces

This directory is the default destination for reproducible, per-run traces from
both case studies. Generated experiment directories are separate from the
aggregate tables in `results/` and are ignored by Git.

## What each run contains

- `metadata.json`: case study, model, fault, run number, experiment ID,
  repository commit, start time, and case-specific settings.
- `queries/*.sparql`: the exact query text used to retrieve KG context.
- `subgraphs/*.ttl`: the exact Turtle returned by GraphDB and inserted into the
  model prompt. An empty file plus `loaded: false` in `events.jsonl` identifies
  a failed or unavailable retrieval.
- `prompts/*.json`: complete ordered system and user messages after KG
  augmentation and evaluation feedback have been inserted.
- `responses/*.json`: raw and parsed model responses, the rationale explicitly
  supplied by the model, token usage when available, and latency.
- `events.jsonl`: ordered retrieval, validation, trajectory, and completion
  events, including accepted/rejected status and failure reasons.
- `trajectories/*.csv`: accepted plant trajectories and, for CSTR, accepted
  digital-twin validation rollouts. Sidecar metadata identifies `nominal` or
  `fault` conditions and the trajectory source.
- `final_result.json`: final run status and metrics.
- `manifest.json`: artifact paths, sizes, and SHA-256 hashes.

“Rationale” means the explanation emitted in the model's structured answer. The
logger does not expose or claim to record hidden chain-of-thought.

## Generate the reviewer package

Run GraphDB and the selected model provider, then execute fresh nominal and
fault runs. Existing aggregate CSVs cannot reconstruct exact prompts, returned
subgraphs, or raw model responses.

From `case_studies/mixer_case`:

```bash
python mixer_case.py --fault normal --runs 1 --model gpt-4o-mini
python mixer_case.py --fault all --runs 1 --model gpt-4o-mini
```

From `case_studies/cstr_case`:

```bash
python cstr_case.py --fault normal --runs 1 --llm-model gpt-4o-mini --mode 3sigma
python cstr_case.py --fault all --runs 1 --llm-model gpt-4o-mini --mode 3sigma
```

For final paper artifacts, use the same model, prompt level or detector mode,
fault set, random-seed policy, and number of repetitions reported in the paper.
Keep successful and failed run directories so selection is auditable; accepted
trajectory CSVs are emitted only when the corresponding acceptance rule passes.

Before sending traces to a reviewer:

1. Confirm every claimed condition has the intended number of run directories.
2. Confirm grounded runs have non-empty `.sparql` and `.ttl` files.
3. Confirm each model call has matching prompt and response JSON files.
4. Confirm nominal and fault conditions have accepted trajectories where
   claimed in the paper.
5. Check `final_result.json` and `events.jsonl` for exceptions or rejected runs.
6. Inspect artifacts for sensitive information.
7. Zip the experiment directories together with this README; do not replace
   them with copied terminal output or aggregate result tables.
