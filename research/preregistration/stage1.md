# Stage 1: infrastructure, detector verification, and task calibration

This protocol is fixed before any Stage 1 validation trajectories or training.
Hash this document and the fixed manifest; commit and push the lock before any
evidence-producing execution. Never edit these locked registration files.
Implementation repairs require a separate timestamped amendment describing the
change and replacement source hashes, committed and pushed before rerunning.
Retain all failed runs and costs. Do not change the scientific criteria to fit
observed results. Stage 2 remains unauthorized.

## Prior knowledge and authorization

The paper, supplied Mac outputs, and detector impact audit were inspected before
this lock. Existing paper results are known. The retrospective audit is separate
and was locked under `preregistration/paper-impact-audit.md`. No new Stage 1
policy was trained and no validation trajectory was generated before this lock.
Package resolution and API-signature inspection are engineering preparation.

The user approved the paper correction drafts and resumed Stage 1 with these
specific changes: use the archived detector for the Mac-versus-Modal check; add
a matched corrected-detector descriptive run; use corrected detection in the
new project; exclude target pickups; keep bit-identical reward arithmetic on
recorded paper events as the reduction criterion. These decisions override the
initial request for unchanged event detection and full detector reward identity.

## Reused learner and environment

Use the archived CleanRL-style PPO and CNN: 16 environments, 128-step rollouts,
annealed learning rate 2.5e-4, gamma .99, GAE lambda .95, four minibatches, four
epochs, clip .2, entropy .01, value coefficient .5, gradient norm .5, and
per-minibatch advantage normalization on. Keep Adam epsilon, value loss,
truncation treatment, encoder, action sampling, logging, and the evaluation
episode quota as archived. The corrected adapter changes detector plumbing and
moves detector attachment after reset; it does not change PPO optimization.

Every final evaluation uses 64 episodes per environment type, four episodes from
each of 16 parallel environments, as in round 4. Do not use first-to-finish
episode selection. Keep the archived logging cadence of ten PPO updates plus
the last update. Requested 1M and 2M budgets floor to complete 2,048-step PPO
batches, giving 999,424 and 1,998,848 actual steps. Train each budget separately:
learning-rate annealing makes a 1M prefix of a 2M run a different intervention.

All scientific workers, including validation, execute on Modal. A tracked local
controller performs orchestration and output retrieval only; it does not train
or simulate environments on the Mac. Use one Modal Function input per manifest
entry, with at most four in flight, and retain separate preemption attempts.

Pin Python 3.12.12 and the corresponding archived NumPy 2.5.4 dependency branch.
This choice uses the Mac runtime log and archived dependency markers, before
new results; the earlier paper/handoff's unconditional NumPy 2.4.6 claim was
incomplete. Pin Gymnasium 1.4.0, MiniGrid 3.2.0 and Torch 2.14.1 (Linux CPU wheel
2.14.1+cpu), plus all other packages through uv.lock. Record the complete actual
installed inventory, base-image digest, CPU/cgroup details, and source commit in
every worker. This is a descriptive platform comparison, not a promise of
identical cross-architecture arithmetic or an imported-version inventory of
the historical Mac environment.

## Corrected event definitions fixed for Stage 1

1. `new_view`: exact bytes of the 7x7 symbolic post-action view not yet seen in
   the episode. The reset view is not preinserted. Use true final observations
   on terminating/truncating steps; clear the set after processing that step.
2. `pickup`: an actual carrying transition from nothing to an object, including
   on the final step, **except successful target pickup**. Define this exception
   without object-specific subgoal labels: the pickup transition itself ends
   the task with positive raw task reward. Such target pickups are recorded
   separately as excluded diagnostics and do not trigger pickup relief.
3. `drop`: an executed drop action that changes carrying from an object to
   nothing. Blocked attempts do not count. Final-step drops count.
4. `toggle`: a door changes open/locked state, including closing/reopening, or
   a box is opened/removed/replaced by its contents by a toggle action. Blocked
   attempts do not count. Final-step toggles count.

Capture physical state changes before vector SAME_STEP autoreset. Attach the
detector only after reset. The four definitions requested for later review—
final-step inclusion, box-opening inclusion, target-pickup exclusion, and
after-reset attachment—are **open Stage 2 pre-registration decisions**, even
though the present Stage 1 implementation is fixed. Do not silently change them
after seeing Stage 1 results.

For the corrected descriptive paper configuration only, map the corrected
events new_view/pickup/toggle onto the paper's three needs, preserving its exact
need arithmetic and three-dimensional observed need state. No drop need is
added to that control. Keep the legacy locked-door/key mode available as an
explicit archival reference; use the new generic definitions in the new reward.

## Generic reward and verification

Four needs start at h0=.5. Weights are in [-1,1]; reliefs in [0,1]; growth is
either zero or in [1/640,1/20]. Each step uses equation (1)'s actual operation
order: clip(h + growth - relief * event, 0, 1), a single final clip. Tampering
with action 6 sets every next need to zero before reward is computed. A
nonnegative optional press cost is subtracted from training extrinsic reward,
independent of the intrinsic coefficient; raw task-success metrics are retained.

Compute the signed reward as V(h)-V(h_next), V=sum(w*h^2). Float32 potential
summation preserves archived order new_view,pickup,toggle, then drop, omitting
zero-weight terms, so the three-need reduction has identical arithmetic.

Before training, all validation tests must pass on Modal:

- Five fixed DoorKey seeds 905–909, each 1,000 uniform-random scalar steps, with
  actions unfiltered. Record views, actions, done flags, paper events, both
  detectors' events, needs and rewards. Replay the paper events into the generic
  reward with weights (1,1,0,1), reliefs (.1,1,0,1) and paper growth 1/160;
  compare intrinsic rewards bit for bit. This is the arithmetic reduction test.
- Run both detectors on the same fixed recorded trajectories. Report every
  disagreement by event type and count, with step indices and terminal/initial
  episode context. Do not require detector identity or select trajectories to
  suppress known disagreements. Drop is an added channel, reported separately.
- Targeted tests for each audit defect: box opening/removal; an eligible pickup
  at timeout; initial door detection after reset. Also test successful target
  exclusion, true final-step views, drop success versus blocked attempts,
  reset boundaries, need limits, tamper reset, and extrinsic press-cost handling.
- Numerically check the signed bound on fixed recorded uniform-random rollouts,
  without learning, choosing among prespecified signed weight fixtures, gamma
  values and tamper flags in the locked test code. No candidate is scored for
  fitness or selected. This is not random search over reward parameters.
- Check the reused statistical functions and the scorer against independent
  known cases and incomplete-data cases. Validation reports include failures,
  errors, skips, per-test outcomes, and recorded evidence hashes.

For any episode of length T and 0<gamma<=1, define m=sum(min(w,0)) and
M=sum(max(w,0)). Summation by parts gives
G=V0-(1-gamma)sum_{t=1}^{T-1}gamma^{t-1}Vt-gamma^{T-1}VT.
Future coefficients sum to one, hence V0-M <= G <= V0-m. With h0=.5,
the ceiling is .25*sum(w_positive)+.75*sum(abs(w_negative)) <=3.
For the three positive legacy weights it reduces to .75. Episode resets are
outside paid transitions; do not sum across uncharged resets and claim the
single-episode bound. Float32 numerical checks use the tolerance fixed in the
test code, while the stated theorem is exact over real arithmetic.

## Fixed execution schedule and seeds

The machine-readable schedule is `stage1/manifest.json`.

1. One validation Function input; training is blocked until it passes.
2. Determinism: corrected episodic novelty, coefficient 1e-3, DoorKey-8x8,
   seeds 910 and 911, 131,072 steps, two separate Function calls per seed.
   Compare all numerical learning metrics after removing only SPS/time, the
   summary, and sorted model tensor bytes. Retain raw checkpoint files; serializer
   metadata alone does not define learning identity. Report match or mismatch
   without tuning execution in response to the outcome.
3. Descriptive cross-platform/control runs: satiation 1/160, DoorKey-8x8, seeds
   40–44, nominal 2M, once with the exact archived A1b detector/code and once
   with corrected detection. Compare solve counts and per-seed training curve
   areas with the provided Mac records. The same source A1b configuration and
   three need state are used in both new variants. **No inferential tests,
   hit rules, or comparative conclusions** for these descriptive runs.
4. Calibration: seeds 900–904; task-only and episodic novelty coefficient 1e-3;
   separate nominal 1M and 2M calls for every task/seed/method cell. Corrected
   plumbing is used throughout. Arm order is counterbalanced by seed parity,
   as encoded before any result in the manifest.

Training candidates: DoorKey-8x8, KeyCorridorS3R1, UnlockPickup. Held-out
candidates: KeyCorridorS3R2, BlockedUnlockPickup, DoorKey-16x16, MultiRoom-N2-S4.
Held-out refers to exclusion from future evolution; Stage 1 baseline calibration
on these tasks is authorized and is disclosed. Do not use the old paper seeds
for calibration, evolution or final evaluation. Reserve 1000–1999 for evolution,
3000–3999 for random search and 2000–2999 for final evaluation. No such run is
authorized in Stage 1.

## Outcomes, fixed selection, and baseline separation

Primary per-seed area: arithmetic mean of all logged train/success rows,
missing success fields treated as zero, as in the paper. Solve: mean success
over the final max(1,floor(number_of_rows/10)) log rows >=.9. First-solved step:
the archived exact first vector step with at least 100 completed episodes and
last-100 success >=.9. For the reported median include unsolved seeds censored
at the nominal requested budget, and label the result as censored. Also retain
the raw None values and solve count. No repeat-based behavior label is primary.

For every task/method/budget cell, require all five completed seeds before
reporting a complete-cell mean, solve count or selection outcome. Partial data
are labeled partial; a budget stop does not turn missing runs into failures.
Report per-cell solve count, mean training curve area, median censored
first-solved step, wall time and cost, and retain the per-seed data.

Apply the user's rule mechanically:

- Training task: keep iff the larger of the two baseline mean areas at 1M is
  <=.85 AND at least one baseline seed has positive raw task success by 2M.
  The latter is evidenced by first_goal_step present within actual training
  steps or a positive logged training-success value. A final solve is not
  required for learnability.
- Held-out task: keep iff the larger baseline mean area at 2M is in [.05,.85],
  inclusive. Report every rejected task and the criterion it failed.

To describe whether 1M separates the baselines on training tasks, report the
mean difference and two-sided Mann-Whitney U on the five per-seed areas per arm.
Use exact U when pooled observations are unique; otherwise independent-label
permutation of the same U statistic, up to 200,000 resamples with seed zero
(enumerate all labelings where feasible). Apply Holm over the planned three
training-task comparisons, retaining family size three if a comparison cannot
be computed. Call statistical separation only for adjusted p<.05; failure to
separate does not establish equivalence. Five seeds give limited power. The
user, not an automatic formula, will decide any later fitness-budget change.
Keep scoring and interpretation separate and do not fit new scoring formulas.

## Compute accounting and stop rule

Use fixed 1-CPU, 2-GiB workers and the pinned image. Record child CPU seconds,
requested allocation seconds, peak RSS/cgroup memory, function wall time,
complete call elapsed time, attempts, and dollar estimates from published
resource rates. Preserve complete package versions and all core logs. Retain
Modal outputs on a named volume and sync hashes into the local results folder.

Total Stage 1 spending limit is $15, including validation, both descriptive
variants, determinism, calibration, failures and retries. Before each launch,
reserve its complete call-deadline resource envelope and termination allowance;
include every in-flight reservation and a shared overhead allowance. Stop
before the next reservation could cross $15, or if actual provider evidence
reaches it. Stop dispatch on an unresolved failure and reconcile its state and
charges before more work. Do not discard an entry to restore available budget.
Provider interval billing is retained separately from resource-rate estimates;
do not describe an estimate as an invoice or claim the provider has a verified
hard billing cap if no such cap is configured. Storage/build/egress accounting
limitations are reported. No automatic budget increase is authorized.

## Cost projections and Stage 2 draft

Using measured costs, project 120 CMA-ES candidate evaluations (population 10,
12 generations) and 120 random-search evaluations, each on kept training tasks
with three seeds at the fitness budget: 720 training calls per kept task before
any explicitly disclosed re-evaluation overhead. Do not execute them.

For Stage 3, give costs for 20 fresh seeds per kept task at nominal 2M for the
best evolved reward and the two named baselines, with an additional scenario
including the random-search winner. Project tamper costs as explicit scenarios
because the tested methods, tasks, prices, adaptation budget and pass rules
remain Stage 2 decisions. Do not imply an unspecified tamper matrix is fixed.

Write an **unlocked draft** Stage 2 outline covering the 12-parameter genome,
zero/log growth encoding, fitness, CMA-ES/re-evaluation, equal-budget random
control, seed roles, final tests/hit rules, and tamper design. Clearly list all
open decisions, including the four event-definition choices above. Stop after
the Stage 1 report, or at the budget limit with incomplete cells disclosed.
