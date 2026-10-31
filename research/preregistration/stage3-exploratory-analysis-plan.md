# Descriptive exploratory analysis of the already observed Stage 2 search

This plan is written **after Stage 2 outcomes and the selected winners were
observed**. Its location in the registration directory does not make it a
pre-registration against unseen Stage 2 data. The analysis is descriptive and
exploratory. It adds no p-values or registered hypothesis tests, and changes
neither fitness, winner identities, Stage 3 tests, seeds nor the final roster.

The lead will commit this plan and its implementation before executing this new
exploratory analysis. Only synthetic fixtures may be executed before that
commit. The original Stage 2 and Stage 3 locked materials remain unchanged.

## Population and outcomes

Include all 240 initial search candidates: 120 CMA-ES and 120 random-search
candidates. Exclude finalist re-evaluation observations from these associations.
For each candidate, use the frozen decoded float64 weights, reliefs and refill
rates from its saved specification, whose identity and file hashes are checked.
Do not regenerate candidates or round their parameter values.

The three descriptive outcomes are the recorded combined search fitness and
the mean recorded area over the three search seeds for each task separately
(DoorKey-8x8 and UnlockPickup). The combined score must agree with all six
recorded areas within a floating-point tolerance of 1e-15; this cross-check does
not change the registered score. Incomplete, duplicate or inconsistent panels
stop the analysis rather than being omitted or imputed.

## Features and average-rank Spearman summaries

For each event in new_view, pickup, drop, toggle, report every association of
each outcome with:

- weight;
- relief;
- the indicator that refill is exactly zero;
- raw refill rate, with its zero values retained;
- refill rate restricted to candidates whose refill for that event is positive.

Also report the parameter-only Boolean indicator that at least one of pickup,
drop or toggle has **both positive weight and positive refill on that same
channel**. The name `any_object_channel_positive_weight_and_refill` does not
assert that an object exists in an environment or that a policy used it.

Use Spearman correlation defined as Pearson correlation of one-based average
ranks, with exact recorded equality defining ties. Report the sample count,
correlation and an explicit undefined status for constant features, constant
outcomes or fewer than two observations. Report all feature/event/outcome rows,
including zeros and undefined values. No p-values, significance thresholds,
feature selection, regression fitting or uncertainty claims are introduced.

## Group summaries

For each event and each outcome, report n, mean, median, Q1, Q3 and IQR for:

- negative, zero and positive weight;
- zero and nonzero refill;
- every cell of weight sign crossed with zero/nonzero refill.

Report the same statistics for each value of the parameter Boolean. Empty groups
are retained with n=0 and undefined summary values. Quartiles use linear
interpolation at index (n-1)p; IQR is Q3-Q1. Exact zero defines zero-weight and
zero-refill groups. Do not combine empty or small cells after inspecting results.

## Stratification and interpretation

The requested pooled view includes all 240 candidates. Also report the same
rows separately for CMA-ES and random search. CMA-ES candidates are adaptive and
not independent uniform draws; generation seed panels differ, parameters are
correlated, and selected populations differ between methods. Associations may
therefore reflect search adaptation, seed panels, task interaction or correlated
features. Do not describe these summaries as causal effects or independent-sample
inference. They do not justify swapping E and R or changing any reward or test.

## Outputs and verification

`stage3/exploratory.py` emits the 240-row candidate table in JSON and CSV, a complete
correlation CSV, complete group-summary CSV and a JSON summary. It reads only
the frozen Stage 2 per-run score table, selection export, markers and saved
specifications. Hash-check specifications, verify panels and score consistency,
and retain the analysis-plan/code hashes in the final artifact manifest. Use
`tests/test_stage3_exploratory.py` for synthetic tied-rank, constant-feature,
conditional-sample, Boolean-channel and empty-group checks.
