# Stage 2 pre-check: pinned determinism test protocol

Committed before any evidence from this test exists. The test was approved by the
user on 2026-10-28 (`approvals-log.jsonl` entries `stage2-cpu-test-add-onednn` and
`stage2-review-before-lock-and-pre-review-tasks`). It is a pre-lock check, not part
of the Stage 2 search, and does not lock the Stage 2 pre-registration.

## Question

Does forcing `ATEN_CPU_CAPABILITY=avx2`, `MKL_CBWR=AVX2` and `ONEDNN_MAX_CPU_ISA=AVX2`
make Stage 1's seed-910 determinism run bit-identical across AMD Zen 3, AMD Zen 4
and Intel containers?

## Fixed specification

- Stage 1 manifest entry `determinism-DoorKey-8x8-episodic_count-131072-seed910-r0`,
  configuration unchanged: DoorKey-8x8, episodic novelty coefficient 1e-3, corrected
  detector, seed 910, nominal 131,072 steps, archived PPO and CNN.
- Only `id`, `purpose` (`pinned_determinism`), `replicate` and `isa_pins` are added.
  The worker rejects any other difference (`stage2/specs.py`).
- The pins are set in the environment of the training subprocess and of a separate
  probe subprocess in the same container. The image recipe is Stage 1's, with the
  Stage 2 package inventory (adds `cma==4.5.0`).

## Replicates and stop rule

Modal does not offer a choice of CPU type, so replicates are submitted in waves.

1. Wave 1: replicates r0–r31, at most 32 in flight.
2. After a wave completes, classify each completed replicate's CPU from its saved
   `/proc/cpuinfo`: Zen 3 = AMD family 19h models 00h–0Fh; Zen 4 = AMD family 19h
   models 10h–1Fh; Intel = any GenuineIntel; other = anything else.
3. Stop when Zen 3, Zen 4 and Intel each have at least 2 completed replicates.
   Otherwise submit the next wave of 32, up to 96 replicates in total.

There is no other stopping or selection rule. Every completed replicate is
reported, including those classified as "other". A recorded replicate is never
resubmitted. A Modal preemption restart of the same call is not a new replicate.

## Identity criterion

Stage 1's three fingerprints are used: metrics without SPS, the summary, and sorted
parameter tensor bytes. **Bit-identical** means every completed replicate has the
same three fingerprints, with all CPU classes pooled. Also reported:

- identity within each CPU class;
- whether the pinned result equals either Stage 1 unpinned replicate (descriptive
  only).

## Pin effectiveness

Each container records:

- `torch.backends.cpu.get_cpu_capability()` and the torch build flags;
- a oneDNN verbose sample from a batched 2×2 convolution;
- an MKL verbose sample from a linear layer.

The pins count as effective in a container when all three hold: the ATen capability
reads AVX2, oneDNN reports no AVX-512 instruction set, and MKL reports `CNR:AVX2`.
If they were not effective somewhere, that is reported, and the result is read as
"pins not applied there", not "pins insufficient".

## Outcome and the rule it implies (specified by the user)

- **Bit-identical across Zen 3, Zen 4 and Intel:** Stage 2 uses these three settings
  in every run.
- **Not bit-identical:** record the CPU type for every run and treat same-seed results
  as unpaired.

## Cost and concurrency

- **Cost:** each call reserves (1200 + 60) s × $0.00001754/s ≈ $0.022 in the shared
  Stage 2 ledger, which has the user-approved $30 cap. At most 96 calls.
- **Concurrency:** the controller records the peak number of running containers, as
  the empirical check of the 32-call setting. Modal's documented workspace limit is
  100 containers on Starter and 5,000 on Team.
- **No adjustment:** no execution setting is tuned in response to the outcome.
