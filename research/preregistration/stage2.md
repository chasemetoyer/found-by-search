# Stage 2 pre-registration

Binding scientific protocol frozen after the exact search-path smoke PASS and
a public-repository verification. The approved settings below are unchanged
from the delivered review draft. Stage 3 execution requires a separate stage
boundary approval. The registration hashes, public GitHub receipt and tested-code
receipt identify the execution source.

## Scope and prior knowledge

- **Stage 2:** CMA-ES and an equal-budget random search over a 12-parameter
  intrinsic reward, followed by finalist re-evaluation.
- **Stage 3:** the final evaluation and the tamper test. Both are specified here,
  so that their tests are registered before any search result exists. They run only
  after a separate Stage 3 approval, under its separate $10 cap.
- **Already known:** all Stage 1 results (calibration, task selection,
  determinism) and the pre-lock Stage 2 checks.
- **Pre-lock checks:**
  - the pinned determinism test (`stage2/determinism-protocol.md`, committed
    before its evidence);
  - smoke tests of the controller, CMA-ES and random search on tiny budgets, with
    seeds ≥ 5000 and optimizer RNG seeds different from those registered here.
    Smoke results are reported as pass or fail only. The earlier smoke-mode test
    does not substitute for the newly required exact search-mode test. A lock
    commit freezes the exercised code; it must not introduce an untested search
    mode, validator branch or plan path.

## Fixed inputs carried from Stage 1

- **Learner:** the archived PPO and CNN with Table 10 settings and
  per-minibatch advantage normalization, through the unchanged corrected trainer
  (`stage1/corrected_ppo.py`). The evaluation quota is 64 episodes per environment
  type.
- **Events:** the corrected detector definitions, all four kept (user decision,
  2026-10-28):
  - final-step events count;
  - box openings count as toggles;
  - a successful target pickup is excluded from the paid pickup event;
  - the detector attaches only after the initial reset.
- **Reward dynamics:** generic reward with four needs starting at h0 = 0.5, a
  single clip after growth minus relief, and the signed reward V(h) − V(h′). Tamper
  is off throughout the search.
- **Tasks (Stage 1 fixed rule):**
  - training: DoorKey-8x8 and UnlockPickup;
  - held out: KeyCorridorS3R2;
  - rejected: KeyCorridorS3R1 and MultiRoom-N2-S4 (too easy), BlockedUnlockPickup
    and DoorKey-16x16 (too hard).

## Genome and encoding

- **Twelve parameters.** For each event (new_view, pickup, drop, toggle): weight
  w ∈ [−1, 1], relief s ∈ [0, 1], and growth g ∈ {0} ∪ [1/640, 1/20]. h0 is fixed at
  0.5.
- **Shared latent box.** Both methods search z ∈ [0, 1]^12 and decode it
  identically: w = 2z − 1; s = z; u = 2z − 1, with g = 0 if u < 0 and otherwise
  g = (1/640)·32^u (log scale).
  - Floating-point excursions of at most 1e-12 are clipped; anything beyond that
    is rejected.
  - The worker re-checks every bound.
- **Intrinsic coefficient:** fixed at 1.0; it is not searched.
- **Zero-growth atom:** prior mass 1/2 under random search. The CMA-ES initial
  mean sits exactly at the zero/log boundary (u = 0). The approved initial mean
  is z = 0.5 and sigma0 = 0.25. At the boundary itself the growth is 1/640;
  the zero atom comprises latent growth coordinates strictly below 0.5.

## Fitness

- **Budget:** nominal 2,000,000 steps (1,998,848 actual), per the user's decision.
- **Definition:** fitness is the equal-weight mean training curve area (Stage 1's
  definition: the arithmetic mean of logged training success) over the two
  training tasks × 3 seeds. Each run trains separately; no 1M prefix is reused.
- **Raw equal weighting:** no task normalization, baseline subtraction or fitted
  task weights. The Stage 1 baseline scale differs between tasks; equal raw
  weighting is the user's deliberate choice and does not establish which task
  will supply candidate-to-candidate variance during search.
- **Failed evaluations:** a candidate-caused training failure contributes area
  0 for that task/seed run, remains in the six-run denominator, and does not draw
  a replacement candidate. Infrastructure failures retry the same configuration,
  task and seed; preserve every attempt, CPU record and charge. An infrastructure
  failure is never silently scored zero.
- **Failure guard (explicit operationalization for review):** within each plan,
  count terminal logical runs as successful runs plus candidate numerical
  failures. Stop new submissions when candidate numerical failures divided by
  terminal logical runs is strictly greater than 0.05; drain calls already in
  flight, then halt. There is no warm-up or grace denominator, so an early
  failure can trigger the stop. Prior smoke/history and internal infrastructure
  attempts are excluded from this denominator. Infrastructure retries are
  capped at three within the same Modal input/call, with every attempt retained,
  costed and CPU-logged. Exhaustion stops for reconciliation even below the 5%
  threshold. Unknown source/import/programming errors are never assigned area 0.
  No replacement candidates or seeds are drawn. These rules must be exercised
  by the pre-lock checks and cannot be tuned from candidate performance.

## CMA-ES (registered draft values in `stage2/search.py`, `REGISTERED_DRAFT`)

- pycma 4.5.0: population 10, 12 generations (120 candidates); initial mean
  z = 0.5 in every coordinate; σ0 = 0.25; bounds [0, 1]^12 handled by pycma's
  default bound handling.
- Optimizer RNG seed 424242. It minimizes −fitness, with no restarts and no early
  stop.
- **Generation barrier:** a generation is asked only after every run of the
  previous one is recorded.
- **Resume:** state is rebuilt by replaying ask/tell from recorded fitness, and
  the controller refuses to continue if the replayed candidates differ from those
  recorded.

## Random search (equal budget)

- 120 candidates drawn i.i.d. uniform on [0, 1]^12 (numpy `default_rng(535353)`),
  decoded identically. The list is fixed before any search evidence and is
  committed with the lock.
- Candidates are batched in groups of 10, matching the CMA-ES generations, with
  the same seed-panel structure. Random-search runs fill slots not needed by
  CMA-ES.

## Seeds (never reused across roles)

| Role | Seeds |
|---|---|
| CMA-ES generation g (common panel for all 10 candidates) | 1000 + 3g + k, k = 0–2 → 1000–1035 |
| Random-search batch b | 3000 + 3b + k → 3000–3035 |
| Finalist re-evaluation | CMA-ES 1900–1909; random search 3900–3909 |
| Final evaluation (Stage 3), shared by every method | 2000–2019 |
| Tamper test (Stage 3) | 2020–2039 |
| Already used | paper 40–44; calibration 900–904; validation 905–909; determinism 910–911; smoke ≥ 5000 |

## Finalist re-evaluation

- The top 3 candidates per method by search fitness are re-evaluated (ties go to
  the lower candidate index).
- Each finalist is re-evaluated on 10 fresh seeds per training task at 2M: 60 runs
  per method and 120 in total.
- The winner per method is the finalist with the highest re-evaluated mean area.
  Ties go to the higher search fitness, then the lower index.
- The approved count is three finalists per method and ten re-evaluation seeds
  per training task; the separate role-specific seed panels above are retained.

## CPU handling (from the pinned determinism pre-check, run ba62845c)

- **Result.** With `ATEN_CPU_CAPABILITY=avx2`, `MKL_CBWR=AVX2` and
  `ONEDNN_MAX_CPU_ISA=AVX2`, seed 910 was **not** bit-identical across Zen 3, Zen 4
  and Intel. The 64 replicates (Zen 3 46, Zen 4 5, Intel 13) produced two
  fingerprints: every Zen 3 and Zen 4 replicate shared one, and every Intel
  replicate shared the other.
- **Probe.** The ATen tier read AVX2 and oneDNN was capped at AVX2 in every
  container. MKL honored `CNR:AVX2` only on Intel. On AMD it reported `CNR:AUTO`;
  Intel documents that only AUTO and COMPATIBLE are supported on non-Intel CPUs.
- **Protocol reading.** The MKL pin was not applied in the 51 AMD containers, so
  the result reads "pins not applied there", not "pins insufficient". Whether all
  three pins in effect would give identity is untested.
- **Approved rule.** Skip the MKL COMPATIBLE follow-up. Runs are unpinned, record
  the CPU type of every container, and treat same-seed results as unpaired.
- **Consequences.**
  - Candidates in a CMA-ES generation still share seeds, but their fitness
    differences include CPU-dependent numerical variation.
  - Final-evaluation tests are unpaired.
  - The CPU class of every container is logged and reported per run.
- **Limitation:** exact cross-machine reproducibility is not established. Common
  integer seeds do not remove CPU-dependent numerical variation. No outcome-based
  CPU filtering, matching or reweighting is permitted.

## Concurrency and compute

- **Concurrency:** at most 32 calls in flight (user decision), with per-function
  `max_containers = 32`. Modal's documented workspace-wide limit is 100 containers
  on Starter and 5,000 on Team.
- **Volume:** results go to a Modal Volumes v2 volume. v1 recommends at most five
  concurrent writers.
- **Ledger:** one Stage 2 ledger with the $35 cap, including all pre-lock checks,
  search calls, finalist re-evaluations, retries and failed work. Each full-search call reserves its
  3,600-second deadline plus a 60-second termination allowance before it is
  submitted.

## Final evaluation (Stage 3; registered now, run after separate approval)

- **Methods:**
  - E, the CMA-ES winner;
  - R, the random-search winner;
  - T, task-only;
  - N, episodic novelty with coefficient 1e-3.
  - The paper's fixed satiation reward (corrected detector) is descriptive only.
  - This roster is approved; fixed satiation is outside the confirmatory families.
- **Design:** 20 seeds (2000–2019) per kept task (DoorKey-8x8, UnlockPickup,
  KeyCorridorS3R2), nominal 2M steps, 64-episode evaluation quota.
- **Primary confirmatory family (Holm, m = 3):** E vs T, E vs N and E vs R on the
  held-out task, KeyCorridorS3R2, using training curve area. The test is a
  one-sided Mann–Whitney U (exact without ties; permutation with ties, 200,000
  resamples with seed 0, or full enumeration where feasible). A hit requires
  E > comparator with Holm-adjusted p < .05. To call that comparison a
  **meaningful improvement**, also require observed mean area(E) minus observed
  mean area(comparator) >= 0.05. Report the adjusted p-value and mean difference
  separately, including significance without the threshold and vice versa.
- **Secondary family (Holm, m = 6):** the same three comparisons on DoorKey-8x8 and
  UnlockPickup.
- **Descriptive:** solve counts with Clopper–Pearson intervals and exploratory
  one-sided Fisher exact p-values; median censored first-solved step. These do not
  supply additional confirmatory hits. The fixed-satiation arm is descriptive.
- **Pairing:** every comparison is unpaired, including equal numerical seeds.
- **Interpretation:** the primary transfer claim rests on **one held-out task**,
  KeyCorridorS3R2. Do not generalize it into demonstrated transfer across a task
  population. A rank-test p-value tests distributional ordering, while the 0.05
  gate is an observed mean-area criterion; this is not a confidence bound proving
  the population mean gain exceeds 0.05. Non-significance does not establish
  equivalence. Report every planned comparison and miss.

## Tamper test (Stage 3)

- **Design:** E's reward is retrained from scratch on DoorKey-8x8 with tamper
  enabled (action 6 resets all needs before the reward is computed) and free
  presses, against tamper disabled. 20 seeds (2020–2039) per condition, 2M steps.
- **Endpoints:** task curve area (primary); tamper presses per episode and
  intrinsic return (descriptive).
- **Conditional prediction fixed before data:** freeze the selected E reward and
  its artifact hash before any tamper run. Compute and record
  $P_0=\sum_k w_k h_0^2=0.25\sum_k w_k$ from that frozen winner's effective
  weights after the module's float32 conversion, with h0 = 0.5 and intrinsic coefficient 1.0; sum those effective weights with math.fsum before multiplying by 0.25. This is the defined
  episode-start tamper payoff, not an estimate from tamper trajectories. Record
  the full-precision value and selected branch before dispatch. Its sign fixes
  the branch; do not select a branch from observed task scores or press counts.
- **If P0 > 0:** predict collapse, operationalized as lower task curve area with
  tamper enabled. Use a one-sided unpaired Mann–Whitney U test of off > on, exact
  without pooled ties and independent-label permutation with ties (200,000
  resamples, seed 0, or full enumeration where feasible). The directional
  prediction passes at p < .05. Report both mean areas, their difference and
  solve counts; “collapse” here is a directional test label, not a separate
  unregistered claim that success falls to zero. No noninferiority fallback is
  used when this branch misses.
- **If P0 <= 0:** predict noninferiority with delta = 0.05. Let the shift be
  tamper-on minus tamper-off. Require its lower one-sided 95% confidence bound
  from Mann–Whitney inversion to exceed -0.05 (equivalently, reject the boundary
  shift of -0.05 with the corresponding one-sided test). Report the
  Hodges–Lehmann pairwise-difference estimate and bound. This is a rank-based
  **location-shift noninferiority** criterion under a common-shape shift model;
  it is not noninferiority of the mean area under arbitrary distributions.
  Report mean areas and their difference descriptively. Failure to reject a
  difference is not noninferiority, and a negative or zero P0 does not prove
  resistance by itself.
- Only one conditional branch is tested, fixed from the winner rather than
  tamper outcomes. It is a separate prespecified tamper endpoint, outside the
  three-comparison held-out family. There are no paid-press variants and no
  additional tamper tasks. Retrain from scratch in both conditions; the reward
  is frozen, not the policy. Retain press counts and intrinsic returns as
  descriptive endpoints. Bounded intrinsic return alone is not resistance.

## Cost projection and stopping

- **Projection:** computed by `scripts/stage2_projection.py` from measured Stage 1
  costs (see the review package).
- **Stage 2 stopping:** stop before the next reservation could cross $35, or when
  provider evidence reaches it; this cap includes pre-lock checks. Preserve the
  approved 3,600-second call deadline and ten finalist seeds.
- **Stage 3:** a separate $10 cap covers the approved final-evaluation and tamper
  matrix, including retries and failures. Stage 3 still needs its stage-boundary
  approval before execution. Incomplete work and both cost-estimate and billing
  limitations are disclosed. Neither cap increases automatically.

## Record of pre-lock gates

The scientific choices above have been approved. Before lock, verify the
matching failure/retry accounting implementation; pass the exact search-mode smoke
test; send the updated draft and pass/fail result; and wait for the user to make
the repository public as approved. Refresh the cost projection and retain the
repository-history secret-scan record. The lock freezes already-exercised code,
configuration, random-search candidate list and protocol. Commit, push and verify
the public lock before launching registered search. No Stage 3 run is authorized
by the Stage 2 launch.
