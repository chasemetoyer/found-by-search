# Stage 2 report

Generated 2026-10-29T06:48:03.730711+00:00. Stage 2 is complete; Stage 3 has not been started.

All 1,560 planned scientific inputs completed: equal-budget CMA-ES and random search, followed by the registered finalist re-evaluation. The CMA-ES-selected reward (candidate 113) had re-evaluated mean training area **0.595504**; the random-search-selected reward (candidate 57) had **0.632047**. The observed CMA-minus-random difference is **-0.036543**. Thus the selected evolved reward did not beat the selected random-search reward on this selection panel. This is descriptive, not a confirmatory test. Held-out transfer and tamper resistance remain untested.

## Registered design and verification

The public registration and timestamped infrastructure/budget amendments govern this run. Locked files were preserved; no scientific formula, seed, task weighting, candidate encoding or winner rule was adjusted after outcomes. The original $35 cap was superseded by the approval 32 hard $50 budget-only amendment. The temporary uncapped Linux migration never entered training.

CMA-ES used population 10 for 12 generations; random search used 120 independent uniform latent candidates. Each candidate had 3 seeds per training task at nominal 2M steps on DoorKey-8x8 and UnlockPickup. The top 3 candidates per method were re-evaluated on 10 fresh seeds per task; 20 task/seed observations determine each finalist mean. Intrinsic coefficient 1.0; initial needs 0.5; equal task weighting. PPO and the corrected detector were fixed; all training ran on Modal. Per-minibatch advantage normalization remained on. CPU kernels were unpinned and CPU metadata retained.

Fitness is the registered arithmetic mean of logged training-success values per run, then the equally weighted mean over task/seed runs; it is not an environment-quota evaluation score or a newly fitted time-weighted integral. Each nominal 2M run completed 1,998,848 steps (whole 2048-step PPO rollouts). Solve counts below use the fixed final-window mean success >= 0.9 definition. Missing first-solved steps are censored at the nominal budget.

Verification passed on all 1,560 recorded specifications and 23,980 saved files, including policy checkpoints. Full Mac ask/tell replay reproduced every recorded candidate. Selection exports match the reconstructed SearchPlan exactly. A separate math.fsum calculation reproduced run areas, finalist ranks and both winners within 1e-15. The provider volume ledger and selection match the final local copies; the controller lock is released. All 1752 current provider calls, including 192 pre-lock checks, are complete. There were no candidate numerical failures. See [number-check.json](number-check.json), [provider-crosscheck.json](provider-crosscheck.json), [result-integrity.json](result-integrity.json), [per-run.csv](per-run.csv) and [build_report.py](build_report.py).

## Finalists and fixed selection outcome

| Method | Candidate | Search area | Re-evaluated area | DoorKey area | DoorKey solved/10 | UnlockPickup area | UnlockPickup solved/10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CMA-ES | 92 | 0.707195 | 0.587261 | 0.838868 | 10 | 0.335653 | 5 |
| CMA-ES (selected) | 113 | 0.691896 | 0.595504 | 0.817667 | 10 | 0.373342 | 3 |
| CMA-ES | 94 | 0.675909 | 0.496658 | 0.766224 | 9 | 0.227092 | 2 |
| Random (selected) | 57 | 0.609717 | 0.632047 | 0.874935 | 10 | 0.389159 | 4 |
| Random | 116 | 0.583983 | 0.625938 | 0.911386 | 10 | 0.340490 | 4 |
| Random | 103 | 0.510494 | 0.618966 | 0.870054 | 10 | 0.367878 | 5 |

The re-evaluation seeds select the winner, so these are selection-panel scores rather than an independent final estimate. Search candidates also used generation/batch-specific seed panels. The top search score is not a held-out performance claim. UnlockPickup now contributes nonzero outcomes, but this does not authorize a change to its weight. Per-task first-solved medians and full-precision values are in [finalists.csv](finalists.csv); all scored observations are in [per-run.csv](per-run.csv). No inferential p-values are introduced at Stage 2.

## Frozen rewards and conditional tamper prediction

The independently checked exports are [frozen-winner-cma.json](frozen-winner-cma.json) and [frozen-winner-random.json](frozen-winner-random.json). Their SHA256 hashes are recorded below and in number-check.json before any tamper data. These exports freeze reward parameters, not policy weights.

| CMA-ES event | Weight (decoded float64) | Relief | Growth |
|---|---:|---:|---:|
| new_view | 0.5537259382632731 | 0.92761239945448304 | 0.00388290510182421 |
| pickup | 0.63219427581183552 | 0.63256379707181387 | 0 |
| drop | 0.96612665166635581 | 0.56427985590730023 | 0 |
| toggle | 0.40649401294738063 | 0.69009645023580712 | 0 |

Using the registered float32-effective weights, P0 = 0.25 × math.fsum(weights) = **0.6396352350711823**. This is positive, so the frozen prediction is **collapse**, operationalized by the prespecified one-sided unpaired test tamper-off area > tamper-on area. A miss cannot switch to noninferiority. “Collapse” is the registered directional label, not a claim that success must reach zero. No tamper runs or paid-press variants were executed.

- CMA winner SHA256: `e3ec0a9f6b76cdc3a7c2c3da58ce8895b21b238c94db705ac4402d348d5e1fdc`
- Random winner SHA256: `f197a64443341b6776433bd34c5a114023152ff7698abd07180e664b374bb000`

## Spending and attempts

These accounting quantities overlap and must not be added together. Guard reservations are conservative estimates; they are not charges or an invoice.

| Quantity | USD |
|---|---:|
| Final hard cap | 50.00 |
| Final guard | 29.407117 |
| Completed function compute estimate, including pre-lock work | 22.373131 |
| Known terminated-attempt compute estimate | 0.421782 |
| Provider reported completed-hour billing | 22.073989 |

The guard decomposes into current-call reservations/charges 25.4972974134047460803680, retained terminated-call guard 3.20982000, completed Linux-diagnostic allowances 0.20, shared overhead allowance 0.50, and setup guard 0.00. Every old charge remains. The 50 terminated calls retain full conservative reservations, regardless of whether their partial compute was cheaper. One terminated attempt lacks usage; its actual compute is unknown. Exact diagnostic Sandbox invoices are also unknown. The billed snapshot covers complete hourly intervals and may omit recent or otherwise untagged diagnostic usage. Local Mac controller electricity/hardware are not priced.

The audit enumerates **1,802 provider calls** (1,560 scientific logical inputs, 192 pre-lock logical inputs, 50 terminated prior calls) and **1,850 physical attempts**. There are **48 additional attempts on existing provider calls** across 47 completed calls, plus 50 explicit same-input manual retries. Additional attempts establish restarts, not a preemption cause. The recorded errors include KeyboardInterrupt and cancellation; an exact provider-preemption count cannot be established. Usage is missing for 4 physical attempts. The completed aggregate estimates include the retained known usage; do not add the physical CSV again. Bounded download/polling/persistence retries were not separately instrumented, so their count and marginal cost cannot be reconstructed; their elapsed cost remains in reservations/billing. This is an accounting limitation, not a fabricated zero.

Every provider call, previous call, replacement handle, guard, known measured cost and CPU class is listed in [provider-calls.csv](provider-calls.csv). Every physical attempt is listed in [physical-attempts.csv](physical-attempts.csv), including its full saved CPU identifiers. [terminated-attempts-and-retries.csv](terminated-attempts-and-retries.csv) isolates the 50 terminated calls and 50 linked replacements. Invoice-level costs per call are unavailable and remain blank.

### Controller interruptions and diagnosis

The full chronology, CPUs and associated worker cost columns are in [interruptions-and-controller-history.csv](interruptions-and-controller-history.csv). Controller costs and worker costs have separate scopes.

| Run | Result | Reconciliation / limitation |
|---|---|---|
| 55d43a42 | Download DNS failure | 7 saved successful outputs recovered; 23 terminated inputs authorized for identical retry. Laptop sleep was not established by the logs. |
| 13b2db55 | FunctionGetOutputs polling deadline failure | 4 saved successful outputs recovered; 27 new terminated inputs joined 9 previous pending retries. Polling protection missing from the first repair was an implementation miss. |
| 4e1c708d | Linux replay verification failed | Stopped at the first input hash; zero training. Intel family 6 model 85; $0.10 allowance, exact billing unknown. |
| fc91e753 | Read-only Linux input diagnosis completed | Confirmed numeric candidate differences; zero training. Intel family 6 model 85; $0.10 allowance, exact billing unknown. |
| 7c1aa413 | Original-Mac restart completed | Current-ledger 528-specification/60-vector replay passed before restart; 492 saved outputs reused. Remaining 1,068 completed inputs came from the final source. |

The first two interruptions were controller transport failures, not candidate numerical failures. Their outstanding provider inputs were canceled by cleanup. Both recoveries preserved successful outputs; completed runs were not resubmitted. Original handles and fees remain in the ledger history. Successful scientific results have source provenance 445 at b94171f, 47 at 739b3ad, and 1,068 at 2b24daa; no new science was collected by the failed cloud verification or diagnosis. The last Mac controller was an authorized single restart and did not stop again.

## CPU and reproducibility limits

Scientific successful-attempt CPU classes: {"intel": 205, "other": 28, "zen3": 1266, "zen4": 61}. All physical attempts, including pre-lock work and terminated work: {"intel": 269, "other": 31, "zen3": 1468, "zen4": 82}. The other class includes recorded AMD family 23 model 49; it is retained, not discarded or retrospectively matched. The final selected-score comparisons remain unpaired. Exact cross-machine reproducibility of PPO is not established.

Exact controller resume across Mac and Linux also failed: four latent coordinates differed by 1 float64 ULP; decoded relief values differed by 1 ULP and one growth value by 4 ULPs. The full-input diagnosis found numeric differences, not just paths/version/platform metadata. No rounding, tolerance hash, candidate substitution or altered scoring was used to force a Linux replay. Resume proceeded on the original arm64 Mac after exact replay passed. **Future stages should run the controller on one platform from the start.** The original-Mac controller still required the laptop powered, lid open, online, sleep disabled and the application running; the failed migration never established lid-closed execution.

## Stage 3 boundary and remaining decisions

Stage 3 is not authorized by this report. Its registered roster is E (the frozen CMA-ES reward), R (the frozen random reward), task-only PPO and episodic novelty 1e-3, with corrected-detector fixed satiation descriptive only. The planned final panel uses 20 fresh seeds per kept task at 2M; the tamper panel retrains E from scratch with free tamper on/off and no paid-press variants. The held-out confirmatory family is the three E comparisons on KeyCorridorS3R2 (Holm m=3), with observed mean-area gain >= 0.05 additionally required for “meaningful.” The primary transfer claim rests on one held-out task. Secondary training-task tests and the conditional tamper test remain as registered. No Stage 2 score supplies a confirmatory transfer or tamper hit.

The prospective matrix is 300 final-evaluation inputs plus 40 tamper inputs, within the separately approved $10 Stage 3 cap. The Stage 2 observed training-task costs price only the known-task component at approximately 3.41USD; the 100 held-out inputs, startup/retry overhead and incomplete billing remain outside that estimate. See [stage3-cost-outline.json](stage3-cost-outline.json). Refresh a complete projection before launch; do not infer that $10 is guaranteed sufficient. User stage-boundary approval is still required.

## Reproduction and saved outputs

No new training is needed to rebuild this report. With the original Mac source and saved final results available, run:

```sh
uv run --locked python build_report.py --repo /path/to/research/worktree --results /path/to/7c1aa413/repo/results --out /path/to/stage2-report
uv run --locked python write_report.py
```

The scripts, checked numeric exports, frozen selection artifacts, ledger and completed-hour billing snapshot are included alongside this report and mirrored in the canonical Mac handoff. The complete recorded training curves, metadata and policy checkpoints remain in the recorded run results and Modal volume; result-integrity.json anchors all of their checked file hashes. The progress dashboard is a separate read-only artifact labeled “in progress, not the stage-2 report”; it is not the inferential report.

