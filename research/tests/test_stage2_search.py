"""Search-logic checks with a fake provider and synthetic curve areas; no Modal, no training."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from generic_reward import MAX_GROWTH, MIN_GROWTH
from stage2.budget import Stage2Ledger
from stage2.common import canonical_sha
from stage2.scheduler import Hooks, run_plan
from stage2.search import (
    REGISTERED_DRAFT, SMOKE, CMAReplay, SearchPlan, canonical_latent, candidate_spec, decode,
    genome_in_bounds, random_candidates, validate_candidate,
)

REGISTERED_SEED_ROLES = [range(40, 45), range(900, 1000), range(1000, 2000), range(2000, 3000), range(3000, 4000)]


def test_decode_bounds_zero_atom_and_log_growth():
    low, mid, high = decode(np.zeros(12)), decode(np.full(12, 0.5)), decode(np.ones(12))
    assert low["weights"] == [-1.0] * 4 and low["relief"] == [0.0] * 4 and low["growth"] == [0.0] * 4
    assert mid["weights"] == [0.0] * 4 and mid["growth"] == [MIN_GROWTH] * 4
    assert high["weights"] == [1.0] * 4 and high["growth"] == [MAX_GROWTH] * 4
    assert all(genome_in_bounds(g) for g in (low, mid, high))
    assert canonical_latent(np.full(12, 1 + 1e-13)).max() == 1.0
    with pytest.raises(ValueError):
        canonical_latent(np.full(12, 1 + 1e-9))
    with pytest.raises(ValueError):
        decode(np.full(12, -0.01))


def synthetic_area(genome, task, seed):
    w, s, g = (np.asarray(genome[k]) for k in ("weights", "relief", "growth"))
    base = 0.4 + 0.25 * w[0] - 0.3 * (s[1] - 0.6) ** 2 + 2.0 * g[3] + (0.05 if task == "DoorKey-8x8" else 0.0)
    return float(np.clip(base + 0.001 * (seed % 7), 0.0, 1.0))


def test_cma_replay_is_deterministic_and_stays_in_bounds():
    def run():
        replay = CMAReplay(SMOKE)
        fit = lambda g, i: synthetic_area(decode(replay.asked[g][i]), "DoorKey-8x8", 5000)
        return [np.array(replay.candidates(g, fit)) for g in range(SMOKE.generations)], replay
    first, replay = run()
    second, _ = run()
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert all(((x >= 0) & (x <= 1)).all() for gen in first for x in gen)
    assert replay.es.countiter == SMOKE.generations - 1  # the final generation is told only after it runs


def test_random_search_prior_is_deterministic_and_matches_the_latent_box():
    z = random_candidates(REGISTERED_DRAFT)
    assert z.shape == (120, 12) and np.array_equal(z, random_candidates(REGISTERED_DRAFT))
    assert not np.array_equal(random_candidates(SMOKE)[:12], z[:12])
    assert ((z >= 0) & (z < 1)).all() and all(genome_in_bounds(decode(row)) for row in z)
    zero = np.mean([v == 0 for row in z for v in decode(row)["growth"]])
    assert 0.35 < zero < 0.65  # zero-growth atom has prior mass 1/2


def test_smoke_seeds_and_rng_are_outside_every_registered_role():
    cfg = SMOKE
    seeds = [s for g in range(cfg.generations) for s in cfg.cma_seeds(g)]
    seeds += [s for b in range(cfg.rs_candidates // cfg.population) for s in cfg.rs_seeds(b)]
    seeds += cfg.finalist_seeds("cma") + cfg.finalist_seeds("random")
    assert min(seeds) >= 5000 and not any(s in role for s in seeds for role in REGISTERED_SEED_ROLES)
    assert {cfg.cma_rng_seed, cfg.rs_rng_seed}.isdisjoint({REGISTERED_DRAFT.cma_rng_seed, REGISTERED_DRAFT.rs_rng_seed})
    assert cfg.budget_steps == 8192 and REGISTERED_DRAFT.budget_steps == 2_000_000


def test_registered_draft_is_an_equal_budget_search():
    cfg = REGISTERED_DRAFT
    assert cfg.population * cfg.generations == cfg.rs_candidates == 120
    runs = len(cfg.tasks) * (2 * 120 * cfg.seeds_per_task + 2 * cfg.finalists_per_method * cfg.finalist_seeds_per_task)
    assert runs == 1560
    panels = [cfg.cma_seeds(g) for g in range(12)] + [cfg.rs_seeds(b) for b in range(12)]
    assert all(1000 <= s < 2000 for p in panels[:12] for s in p) and all(3000 <= s < 4000 for p in panels[12:] for s in p)


def test_candidate_validation_rejects_any_deviation():
    z = random_candidates(SMOKE)[0]
    spec = candidate_spec(SMOKE, "cma", 3, 0, "DoorKey-8x8", SMOKE.cma_seeds(0)[0], z)
    validate_candidate(spec, SMOKE)
    mutations = [lambda s: s.update(seed=1000), lambda s: s.update(task="KeyCorridorS3R1"),
                 lambda s: s["config"]["run"].update(total_timesteps=16384),
                 lambda s: s["genome"]["weights"].__setitem__(0, 0.99),
                 lambda s: s["config"]["intrinsic"].update(coef=0.5), lambda s: s.update(purpose="search_cma")]
    for mutate in mutations:
        bad = copy.deepcopy(spec)
        mutate(bad)
        with pytest.raises((ValueError, KeyError)):
            validate_candidate(bad, SMOKE)


class Call:
    def __init__(self, object_id):
        self.object_id, self.polls = object_id, 0


class FakeBackend:
    app_id = "ap-fake"

    def __init__(self):
        self.spawned, self.reattached, self.calls, self.max_active, self.active = [], [], {}, 0, 0

    def spawn(self, spec, deadline):
        self.spawned.append(spec["id"])
        call = Call("fc-" + spec["id"])
        self.calls[call.object_id] = (call, spec)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        return call

    def reattach(self, call_id):
        self.reattached.append(call_id)
        self.active += 1
        return self.calls[call_id][0]

    def poll(self, call):
        call.polls += 1
        if call.polls < 2:
            return None
        self.active -= 1
        spec = self.calls[call.object_id][1]
        return {"id": spec["id"], "attempt": "a-" + spec["id"], "files": {}, "fingerprints": {},
                "spec_sha256": canonical_sha(spec),
                "usage": {"wall_seconds": 30.0, "compute_cost_estimate_usd": "0.0005"}}

    def cancel(self, call):
        pass


def hooks_for(tmp_path, backend, store):
    def sync(result):
        spec = backend.calls["fc-" + result["id"]][1]
        d = tmp_path / "results" / result["id"] / "attempts" / result["attempt"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "spec.json").write_text(json.dumps(spec))
        area = synthetic_area(spec["genome"], spec["task"], spec["seed"])
        (d / "metrics.jsonl").write_text(json.dumps({"global_step": 8192, "train/success": area}) + "\n")
        (d / "summary.json").write_text(json.dumps({"total_timesteps": 8192, "first_solved_step": None,
                                                    "first_goal_step": None, "seed": spec["seed"]}))
        store[result["id"]] = result
        return d

    return Hooks(persist=lambda: None, sync=sync, sync_existing=lambda key: (store[key], sync(store[key])),
                 sleep=lambda s: None, now=lambda: 0.0)


def expected_runs(cfg):
    return len(cfg.tasks) * (cfg.population * cfg.generations * cfg.seeds_per_task
                             + cfg.rs_candidates * cfg.seeds_per_task
                             + 2 * cfg.finalists_per_method * cfg.finalist_seeds_per_task)


def test_smoke_plan_end_to_end_with_fake_provider(tmp_path):
    backend, store = FakeBackend(), {}
    plan = SearchPlan(SMOKE)
    ledger = Stage2Ledger(tmp_path / "ledger.json")
    run_plan(plan, ledger, backend, hooks_for(tmp_path, backend, store), concurrency=32)
    assert len(backend.spawned) == len(set(backend.spawned)) == expected_runs(SMOKE) == 56
    assert backend.max_active == 32  # the queue is deep enough to fill every slot
    summary = plan.summary()
    assert summary["equal_candidate_budget"] and summary["all_genomes_in_bounds"]
    assert summary["replay_reproduces_cma_candidates"] and summary["winner_selected"] == {"cma": True, "random": True}
    assert summary["runs_recorded"] == summary["runs_expected"] == 56
    assert not any(isinstance(v, float) for v in summary.values())  # no performance values in the summary


def test_interrupted_search_resumes_without_resubmission(tmp_path):
    reference_backend, reference_store = FakeBackend(), {}
    run_plan(SearchPlan(SMOKE), Stage2Ledger(tmp_path / "ref.json"), reference_backend,
             hooks_for(tmp_path / "ref", reference_backend, reference_store), concurrency=32)
    order = reference_backend.spawned
    backend, store = FakeBackend(), {}
    ledger = Stage2Ledger(tmp_path / "ledger.json")
    probe = SearchPlan(SMOKE)  # regenerates specs in submission order to build the interrupted state
    done_keys, running_keys = order[:20], order[20:32]
    for key in done_keys + running_keys:
        spec = reference_backend.calls["fc-" + key][1]
        ledger.reserve(key, canonical_sha(spec), SMOKE.deadline_seconds)
        call = backend.spawn(spec, 0)
        ledger.associate(key, call.object_id, "ap-fake")
        if key in done_keys:
            backend.active -= 1
            ledger.complete(key, 30.0, {"wall_seconds": 30.0})
            store[key] = {"id": key, "attempt": "a-" + key, "files": {}, "fingerprints": {}, "usage": {},
                          "spec_sha256": canonical_sha(spec)}
    del probe
    backend.active = 0
    resumed_spawns_before = len(backend.spawned)
    plan = SearchPlan(SMOKE)
    run_plan(plan, ledger, backend, hooks_for(tmp_path, backend, store), concurrency=32)
    new = backend.spawned[resumed_spawns_before:]
    assert sorted(backend.reattached) == sorted("fc-" + k for k in running_keys)
    assert not set(new) & set(done_keys + running_keys)
    assert set(done_keys + running_keys + new) == set(order)  # same keys as the uninterrupted run
    assert plan.summary()["replay_reproduces_cma_candidates"]
