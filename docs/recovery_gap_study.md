# Why does an LLM fail when recovery is possible?

Research protocol draft, 17 September 2026. This is an experiment design, not a
claim that the proposed mechanisms or method are novel. A focused literature
comparison is needed before positioning the eventual paper.

## Research question and scope

Given a fault state for which an admissible recovery action or sequence has
been demonstrated, what prevents an LLM supervisor from recovering the process?

The primary endpoint is continued productive operation. Safe shutdown is a
separate outcome, as agreed with the project owner. The LLM remains a supervisor;
the existing low-level controllers remain part of the plant dynamics.

Use both existing case studies, with different roles:

- CSTR: continuous setpoint selection, fault severity, intervention delay, and
  the size of the set of successful actions.
- Mixer: discrete path selection and translation of a valid path into actuator
  commands. A route in a graph must also pass guards and physical replay.

The first CSTR study covers a single constant setpoint triple after detection,
matching the present intervention design. This is not a claim about all possible
time-varying control policies. Mixer recovery involves a sequence of decisions.

## Evidence of recoverability

For each saved decision state, retain at least one successful reference action
or sequence and its replayed trajectory: a recovery witness. The witness must
use the same allowed actions, intervention time, objective, and horizon as the
agent. Save controller integrators, actuator state, phase, fault configuration,
and random-stream state, not just measured temperature and level.

A successful witness establishes existence under the specified model and
disturbance realization. Failure of a finite search establishes only that no
witness was found at that resolution and budget. Mark such states unresolved;
do not label them physically unrecoverable. Do not exclude them silently from
the overall benchmark: publish witness coverage and unresolved counts.

For noisy experiments, distinguish a witness on one replayed disturbance from
robust recovery across fresh disturbance realizations. Freeze candidate choices
before evaluating them on held-out noise.

Use privileged plant state only for offline reference analysis. Online baselines
must receive the same observations and tools as the LLM. A full-state reference
is an existence test, not a fair competitor with a partially observed agent.

## Physical outcome contract

Freeze the contract before comparing agents:

- Absolute physical safety limits independent of the proposed setpoints.
- A fixed production target/tolerance and a positive minimum production
  requirement, so reducing feed to zero is not counted as productive recovery.
- A recovery deadline and consecutive recovery hold time, followed by evaluation
  through the remaining operating horizon to detect relapse.
- An explicit transient allowance for a state already outside the operating
  envelope at detection, plus separate non-negotiable catastrophic limits.
- Separate outcomes for productive recovery, safe shutdown, unsafe termination,
  failure to recover by the deadline, and invalid/incomplete execution.

Start from the existing physical limits (temperature maximum 313 K and level
range 2 to 11 in the simulator's units), but audit their meaning and calibrate
the production criterion against nominal operation. These are candidate study
settings, not independently certified safety limits. A temperature operating
target of 310 K must not become easier merely because an agent lowers T_sp.

For the mixer, productive recovery means completing the intended batch transfer
within a deadline, without violating tank/actuator constraints; stopping safely
without finishing the transfer is reported separately.

Measure outcomes on plant truth. Record sensor-based validator decisions
separately, especially when introducing sensor faults. Do not use a validator
pass or selection of the expected next state as the physical endpoint.

## First pilot: no model calls required

1. Make CSTR snapshots and noise replay deterministic and independent of the
   number or order of hypothetical rollouts.
2. Start with the existing stuck-cooling scenario at its current setting.
   The current fault forces cooling opening to 0.3; it is not literally a
   fully closed valve. Expose this opening as an explicit severity parameter.
3. Capture a decision snapshot using a fixed detection rule. Hold detection
   constant initially so the experiment isolates recovery rather than diagnosis.
4. Evaluate a bounded, declared grid of T_sp, L_sp, and Fin_sp values using the
   independent physical outcome contract. Coarsen first and refine near
   successful regions; retain grid bounds, resolution, and search cost.
5. Confirm successful candidates by replay. Save their trajectories and the
   proportion of tested grid points that succeed. This proportion depends on
   the grid and coordinates; do not present it as intrinsic task difficulty.
6. Repeat after prescribed delays while the original controller continues
   evolving the plant. Candidate pilot delays: 0, 10, 30, 60, and 120 seconds.
   Broaden only if those values reveal no informative differences.
7. Add a small severity sweep and inspect whether successful action regions
   change. Do not assume recoverability decreases monotonically with delay.

Pilot output: a severity-by-delay map showing witnessed recovery and unresolved
searches, plus physical trajectories for representative successful and failed
actions. This is the first decision point before a large LLM experiment campaign.

## Agent experiments and observable mechanisms

Replay agents from the saved witness-backed states. Preserve actual proposals,
validator outcomes, candidate counts, and elapsed time; no LLM judge is needed
for primary measurements. Failures may have several contributing mechanisms.

| Hypothesis | Controlled intervention | Evidence to record |
| --- | --- | --- |
| The action specification is ambiguous | Original versus clarified wording with identical action permissions | Change in constraint violations and productive recovery on paired states |
| The agent revisits rejected candidates | Explicit rejected-action memory, then duplicate rejection as a separate condition | Repeated proposals, distinct candidates evaluated, and recovery |
| The agent fails to find a successful action within budget | Equal observation/tool access and rollout budget for LLM, random/grid search, and deterministic policy | Recovery conditional on a witness; candidate coverage and budget curves |
| Decision delay prevents recovery | Immediate execution versus controlled delay replay of the same candidate | Candidate succeeds immediately but fails when delayed; distinguish this from loss of all witnessed options |
| The validator rejects useful actions or admits harmful ones | Replay every candidate against the independent plant outcome contract | Validator false rejection and false acceptance, initially with matched models |
| Available observations conceal needed information | Observation-only versus deliberately augmented information | Effect of information access; do not attribute under-observation to reasoning |

Do not interpret an intervention's improvement as proof of an internal cognitive
cause. The causal claim concerns the changed interface/system component and its
measured effect under the controlled conditions.

Freeze prompts and interventions on development scenarios, then test on new
fault severities, operating points, and disturbance seeds. Include unchanged
PID and deterministic recovery baselines. Treat parameter/model mismatch as a
later extension after the matched-model experiment is trustworthy.

Separate rollout-count comparisons from elapsed-time comparisons. Offline
reference searches may use more computation to establish existence, but their
cost must be recorded and they must not be described as an online baseline.

## Mixer extension

Construct a reference planner over legal transitions, guards, and actuator
assignments. Replay candidate sequences to establish a completion witness.
The existing get_expected_next_state function provides a comparison point,
not independent evidence that every alternative sequence is invalid.

Use stage replacement experiments:

- Reference route and reference actuator mapping: establish executable witness.
- LLM route and reference actuator mapping: isolate route selection.
- Reference route and LLM actuator mapping: isolate action translation.
- LLM route and LLM actuator mapping: measure the complete supervisor.

Keep the information supplied at each stage explicit. Check whether several
fault names share the same routing requirement; these are not automatically
independent recovery mechanisms. Audit the actual local simulation dependency
and physical fault effects before interpreting route correctness as recovery.

## Implementation prerequisites found in the current repository

1. CSTR noise uses NumPy's global random generator. Deep-copying the plant does
   not isolate that global state. Introduce per-instance streams or explicit
   disturbance tapes, so validation attempts cannot alter future plant noise.
2. CSTR fault onset uses Python's separate global random generator. Generate
   explicit scenario configurations from recorded seeds.
3. The current success flag records proposal acceptance before the retry cap,
   rather than the productive physical endpoint defined above.
4. Retry exhaustion marks deterministic fallback as eval_pass without validating
   it in that branch. Preserve separate proposal, validation, fallback, and
   physical outcome records in the study harness.
5. Scheduled shutdown can make end-of-run observations unsuitable for assessing
   continued operation. Use a declared operating horizon before shutdown.
6. Model calls and validation currently occur between plant steps. Implement
   explicit delay replay before drawing conclusions about real-time performance;
   include inference, validation, and orchestration delay as appropriate.
7. The current CSTR grid wording changed between stored experiments. Record an
   immutable action contract, exact prompts, and source hashes for each new run.

Required correctness checks: identical snapshot replay; rollout-order invariance;
validation-call independence of plant noise; reference witness reproducibility;
safe shutdown not counted as productive recovery; known invalid actions rejected;
and elapsed delay represented in plant time. These test the experiment's validity.

## Reporting

Primary measure: productive recovery rate among witness-backed states, with
scenario-paired comparisons and uncertainty intervals. Also report all-scenario
outcomes and witness coverage to expose selection effects.

Secondary measures: time to recovery, later relapse, physical violation duration
and magnitude, production loss, number of distinct proposals, fallback/shutdown
frequency, validator disagreements, and wall-clock decision cost. Treat scenarios
as the units of replication rather than counting every retry as an independent
experiment. Choose final sample sizes after the pilot estimates variability.

The resulting contribution would need an experimentally supported relationship
between recovery opportunity and an identified system bottleneck, or a useful
validated evaluation method. A list of failure labels alone is not the intended
contribution. Novelty remains to be checked against prior work.

## Repository recommendation and migration

Recommended new repository name: recovery-gap. Retain ctrl-alt-recover as the
source framework and use the new repository for the independent research harness.
Reuse the plant models; do not rewrite the physics merely to create separation.

Suggested layout:

```text
recovery-gap/
  README.md
  LICENSE
  provenance/           # upstream URL, commit, source hashes, documented changes
  environments/         # CSTR and mixer adapters, snapshots, disturbance replay
  reference/            # recovery witnesses and search/planning procedures
  agents/               # LLM and non-LLM proposal interfaces
  evaluation/           # independent physical outcome criteria
  experiments/          # immutable scenario and intervention configurations
  analysis/             # deterministic metrics and plotting
  tests/                # replay and experimental-validity checks
  docs/                 # study protocol and migration notes
```

Upstream: https://github.com/AISL-at-Imperial-College-London/ctrl-alt-recover

Inspected HEAD: 4afe06c3eb21e4f61bef4901d4faac6bcf6d7cf5.

The current working tree includes modified case-study/tracing files, untracked
failure-analysis code, and untracked result directories. A clone of HEAD alone
would omit these. At migration, inventory and hash the deliberately selected
working-tree sources and preserve their diff from upstream. Retain the existing
MIT license and attribution for reused code. Record dependencies explicitly.

Keep existing traces/results as immutable exploratory evidence. New experiments
must use a separate manifest and run namespace. Do not copy .env, editor history,
caches, or all historical artifacts into the new source repository. Do not move
or delete the original checkout. Prefer a sibling repository over nested Git
repositories. No new repository or remote has been created by this protocol.
