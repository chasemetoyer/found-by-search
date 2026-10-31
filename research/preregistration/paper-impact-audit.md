# Retrospective detector impact audit — protocol lock

Scope: the user's requested interruption of Stage 1. Audit every actual round 2–5
run, including the round-5 calibration run, using the unchanged local release
and historical code snapshots. Do not train, evolve, search reward parameters,
or generate substitute policy trajectories. Stop after delivering the audit.

This is a retrospective software/data audit, not a prospective test of the
paper's scientific hypotheses. Before this lock we read the supplied paper,
source code, schemas, run names and selected existing summaries. We identified
the three potential detector defects, the empty initial door-cache mechanism,
the intended exclusion of target-ball pickups in KeyCorridor, and the absence
of complete trajectories in the inspected replay outputs. The original paper's
results were already known. No new training or event-frequency calculation ran.

## Sources and preservation

Use the locally available release.zip and its extracted release directory;
retain original bytes. Record archive and file hashes; verify the released
SHA256SUMS. Use run-list.txt and the scored-round maps to identify rounds. Earlier
exploratory runs reused as comparators may be indexed as supplemental context
but must not be silently relabeled round 2. Exact historical source comes from
the local per-run snapshots, checked against the recorded Git commits where
available. Copy all consulted inputs under handoff-cross-platform/paper-impact-audit.
The release itself is not modified. The identity of a local release copy and a
remote Zenodo deposit is not asserted without remote checksum verification.

## Event audit definitions, fixed before calculations

For every round 2–5 run report task, method, seed count, detector mode and:

1. Missed box opening: a realized Box toggle/removal not recognized as an open
   event by a detector purporting to recognize it. If the actual task generator
   cannot create a Box, report structural zero with that reason. A disabled
   detector is not an observed zero. The KeyCorridor key/locked-door mode does
   not claim to reward box events.
2. Skipped terminal pickup: a realized None-to-object carry transition on
   termination or truncation. Distinguish all-object diagnostic omissions from
   omissions of the event eligible under that run's registered reward mode.
   KeyCorridor target-ball pickup is deliberately ineligible in key-only mode.
   Where summary.final10_exact.solved_episodes permits an exact lower/window
   count of target pickups, report its window and denominator explicitly. Never
   infer whole-training counts by summing overlapping rolling-success means.
3. Door-cache initialization: count structurally exposed initial episodes as
   number of configured environments times number of completed seeds only if
   source proves detector attachment precedes reset. Label these exposures,
   not missed door transitions. Record whether the open need is disabled and
   therefore cannot change reward/state. Exact missed opens remain unknown
   without an event trace, even when an exposure count is exact.

Use four explicit evidence classes: exact recorded count, structurally proved
zero/exposure, not applicable, and not identifiable from retained records.
Do not turn missing data into zero. Report the full training window separately
from final-10%-of-steps and evaluation/replay windows. For final-10% target
pickups, report both numerator and completed-episode denominator, and exclude
task-only controls from claims about intrinsic-detector execution.

## Replay and number checks

Inventory each archived replay family: source run, reset/attach order, policy
sampling, saved fields, and whether historical event counts can be reconstructed.
Inspect both aggregate and per-layout output families. New replaying of a final
checkpoint is not historical training evidence and is out of this audit's scope.

Recompute solve counts and mean logged training-success area from each run's
raw metrics using the paper's final-10%-of-log-rows solve rule (threshold 0.9)
and arithmetic mean of logged success. Label these standardized audit values;
do not substitute this rule for round-2's original farmer labels. Cross-check
round-4/round-5 archived solve counts and per-seed training AUC wherever stored.
Preserve the printed number definitions, even when a newer definition might
be preferable. Check every new report number by a second calculation or
independent structural assertion in an audit-number-check script.

## Interpretation and deliverables

Separate unchanged descriptions of the implementation actually run from the
unidentified counterfactual results of retraining corrected code. Do not claim
that a repaired learner would retain the same solve counts, areas, p-values or
conclusions. Identify any demonstrated numerical inconsistency explicitly;
absence of event traces is not evidence of harmlessness. The potential-difference
ceiling theorem can be checked algebraically independently of event correctness.

Write a per-run machine-readable table, a readable audit report, exact proposed
replacement Setup text, source-provenance/hashes, and a number-check log. Mark
the Setup correction as a draft; do not edit the original manuscript or deposit.
Commit and push this protocol and its hash record before calculations. Later
protocol changes require a new timestamped amendment; this file stays unchanged.
