"""Stage 2 search: genome encoding, CMA-ES with replayable state, random search, finalists.

Both methods search the same normalized latent box z in [0, 1]^12 and decode it
identically, so boundary atoms (zero growth) are handled the same way:
  weight w = 2z - 1 in [-1, 1];  relief s = z in [0, 1];
  growth u = 2z - 1: g = 0 if u < 0, else (1/640) * 32**u in [1/640, 1/20] (log scale).
CMA-ES state is never pickled: it is rebuilt by replaying ask/tell from recorded
fitness, and run_plan's spec-hash check proves the re-asked candidates equal the
recorded ones. REGISTERED_DRAFT is an unlocked proposal pending user approval.
"""
from __future__ import annotations

import copy
import json
import math
import re
import tomllib
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from generic_reward import EVENTS, MAX_GROWTH, MIN_GROWTH
from stage1.score import score_seed
from stage2.common import ISA_PINS, ROOT

DIM = 12
NAMES = [f"weight/{e}" for e in EVENTS] + [f"relief/{e}" for e in EVENTS] + [f"growth/{e}" for e in EVENTS]
TRAINING_TASKS = ("DoorKey-8x8", "UnlockPickup")  # kept by the Stage 1 fixed selection rule


def canonical_latent(z) -> np.ndarray:
    """Clip only floating-point excursions of at most 1e-12; reject anything else outside the box."""
    z = np.asarray(z, dtype=float)
    if z.shape != (DIM,) or not np.isfinite(z).all() or (z < -1e-12).any() or (z > 1 + 1e-12).any():
        raise ValueError("Latent candidate outside [0, 1]^12")
    return np.clip(z, 0.0, 1.0)


def decode(z) -> dict:
    z = np.asarray(z, dtype=float)
    if z.shape != (DIM,) or not np.isfinite(z).all() or (z < 0).any() or (z > 1).any():
        raise ValueError("Latent candidate outside [0, 1]^12")
    u = 2 * z[8:] - 1
    growth = [0.0 if ui < 0 else float(np.clip(MIN_GROWTH * (MAX_GROWTH / MIN_GROWTH) ** ui, MIN_GROWTH, MAX_GROWTH))
              for ui in u]
    return {"weights": [float(2 * v - 1) for v in z[:4]], "relief": [float(v) for v in z[4:8]], "growth": growth}


def genome_in_bounds(genome: dict) -> bool:
    w, s, g = (np.asarray(genome[k], dtype=float) for k in ("weights", "relief", "growth"))
    return (w.shape == s.shape == g.shape == (4,) and ((-1 <= w) & (w <= 1)).all() and ((0 <= s) & (s <= 1)).all()
            and ((g == 0) | ((MIN_GROWTH <= g) & (g <= MAX_GROWTH))).all())


@dataclass(frozen=True)
class SearchConfig:
    name: str                      # also the key prefix
    population: int
    generations: int
    rs_candidates: int
    seeds_per_task: int
    budget_steps: int
    finalists_per_method: int
    finalist_seeds_per_task: int
    cma_seed_base: int             # generation g uses seeds base + g*seeds_per_task + k
    rs_seed_base: int              # batch b (size = population) uses seeds base + b*seeds_per_task + k
    finalist_seed_base_cma: int
    finalist_seed_base_rs: int
    cma_rng_seed: int
    rs_rng_seed: int
    sigma0: float = 0.25
    deadline_seconds: int = 3600
    coef: float = 1.0
    isa_pins: dict = field(default_factory=dict)
    tasks: tuple = TRAINING_TASKS

    def purposes(self) -> dict:
        return {m: f"{self.name}_{m}" for m in ("cma", "random", "finalist_cma", "finalist_random")}

    def cma_seeds(self, g):
        return [self.cma_seed_base + g * self.seeds_per_task + k for k in range(self.seeds_per_task)]

    def rs_seeds(self, b):
        return [self.rs_seed_base + b * self.seeds_per_task + k for k in range(self.seeds_per_task)]

    def finalist_seeds(self, method):
        base = self.finalist_seed_base_cma if method == "cma" else self.finalist_seed_base_rs
        return [base + k for k in range(self.finalist_seeds_per_task)]


# Pre-lock smoke test: tiny budget; every seed >= 5000 is outside all registered
# roles (calibration 900-999, evolution 1000-1999, final 2000-2999, random
# 3000-3999, paper 40-44); optimizer RNG seeds differ from the registered draft.
SMOKE = SearchConfig(name="smoke", population=6, generations=2, rs_candidates=12, seeds_per_task=1,
                     budget_steps=8192, finalists_per_method=1, finalist_seeds_per_task=2,
                     cma_seed_base=5000, rs_seed_base=5100, finalist_seed_base_cma=5200, finalist_seed_base_rs=5300,
                     cma_rng_seed=5001, rs_rng_seed=5002, deadline_seconds=900)

# Approved settings; full evidence collection remains gated by the later lock.
REGISTERED_DRAFT = SearchConfig(name="search", population=10, generations=12, rs_candidates=120, seeds_per_task=3,
                                budget_steps=2_000_000, finalists_per_method=3, finalist_seeds_per_task=10,
                                cma_seed_base=1000, rs_seed_base=3000, finalist_seed_base_cma=1900,
                                finalist_seed_base_rs=3900, cma_rng_seed=424242, rs_rng_seed=535353)

# Exercises the search-mode controller and worker validator, with isolated keys,
# tiny budgets and seeds outside every registered scientific role.
SEARCH_PATH_SMOKE = SearchConfig(name="search_path_smoke", population=6, generations=2, rs_candidates=12,
                                seeds_per_task=1, budget_steps=8192, finalists_per_method=3,
                                finalist_seeds_per_task=2, cma_seed_base=5400, rs_seed_base=5500,
                                finalist_seed_base_cma=5600, finalist_seed_base_rs=5700,
                                cma_rng_seed=5401, rs_rng_seed=5402, deadline_seconds=900)


@lru_cache(maxsize=1)
def base_config() -> dict:
    return tomllib.loads((ROOT / "handoff-cross-platform/code/final-round5/config.toml").read_text())


def candidate_config(task: str, seed: int, budget: int, genome: dict, coef: float) -> dict:
    cfg = copy.deepcopy(base_config())
    cfg["env"]["id"] = f"MiniGrid-{task}-v0"
    cfg["env"]["eval_ids"] = [f"MiniGrid-{task}-v0"]
    cfg["run"].update(seeds=[seed], total_timesteps=budget)
    cfg["intrinsic"] = {"kind": "generic_homeostatic", "coef": coef, "h0": 0.5, "weights": list(genome["weights"]),
                        "relief": list(genome["relief"]), "growth": list(genome["growth"]), "tamper": False}
    return cfg


def candidate_spec(cfg: SearchConfig, purpose_kind: str, candidate: int, group: int, task: str, seed: int,
                   latent) -> dict:
    purpose = cfg.purposes()[purpose_kind]
    genome = decode(latent)
    tag = f"g{group:02d}" if purpose_kind == "cma" else (f"b{group:02d}" if purpose_kind == "random" else "f")
    spec = {"id": f"{purpose}-{tag}-c{candidate:03d}-{task}-{cfg.budget_steps}-seed{seed}", "purpose": purpose,
            "method": "cma" if purpose_kind.endswith("cma") else "random", "candidate": candidate, "group": group,
            "task": task, "seed": seed, "total_timesteps": cfg.budget_steps, "replicate": 0,
            "source": "final-round5", "detector": "corrected", "latent": [float(v) for v in latent],
            "genome": genome, "config": candidate_config(task, seed, cfg.budget_steps, genome, cfg.coef)}
    if cfg.isa_pins:
        spec["isa_pins"] = dict(cfg.isa_pins)
    return spec


def validate_candidate(spec: dict, cfg: SearchConfig) -> None:
    """Worker-side check: the spec is exactly what this config would generate."""
    kinds = {v: k for k, v in cfg.purposes().items()}
    kind = kinds.get(spec.get("purpose"))
    if kind is None or spec.get("task") not in cfg.tasks:
        raise ValueError("Purpose or task not allowed by this search configuration")
    if kind == "cma":
        allowed = cfg.cma_seeds(spec["group"]) if 0 <= spec["group"] < cfg.generations else []
        if not spec["group"] * cfg.population <= spec["candidate"] < (spec["group"] + 1) * cfg.population:
            raise ValueError("Candidate index outside its CMA generation")
    elif kind == "random":
        allowed = cfg.rs_seeds(spec["group"]) if 0 <= spec["group"] * cfg.population < cfg.rs_candidates else []
        if not 0 <= spec["candidate"] < cfg.rs_candidates or spec["candidate"] // cfg.population != spec["group"]:
            raise ValueError("Candidate index outside its random-search batch")
    else:
        allowed = cfg.finalist_seeds("cma" if kind == "finalist_cma" else "random")
        limit = cfg.population * cfg.generations if kind == "finalist_cma" else cfg.rs_candidates
        if spec["group"] != 0 or not 0 <= spec["candidate"] < limit:
            raise ValueError("Invalid finalist index or group")
    if spec.get("seed") not in allowed:
        raise ValueError("Seed outside the configured panel for this purpose")
    if spec != candidate_spec(cfg, kind, spec["candidate"], spec["group"], spec["task"], spec["seed"], spec["latent"]):
        raise ValueError("Candidate spec differs from the configured template")
    if not genome_in_bounds(spec["genome"]):
        raise ValueError("Genome outside the registered bounds")


class CMAReplay:
    """pycma CMA-ES whose state is rebuilt from recorded fitness by replaying ask/tell."""

    def __init__(self, cfg: SearchConfig):
        import cma
        warnings.filterwarnings("ignore", message="Could not import matplotlib")
        self.cfg = cfg
        self.es = cma.CMAEvolutionStrategy([0.5] * DIM, cfg.sigma0, {
            "popsize": cfg.population, "bounds": [[0.0] * DIM, [1.0] * DIM], "seed": cfg.cma_rng_seed,
            "verbose": -9, "verb_log": 0, "verb_disp": 0})
        self.asked: list[list[np.ndarray]] = []
        self._raw: list[list[np.ndarray]] = []
        self.told = 0

    def candidates(self, g: int, fitness) -> list[np.ndarray]:
        """Candidates of generation g; fitness(gen, i) must be known for every earlier generation."""
        if not 0 <= g < self.cfg.generations:
            raise ValueError("Generation outside the configured range")
        while len(self.asked) <= g:
            if len(self.asked) > self.told:
                gen = len(self.asked) - 1
                # CMA-ES minimizes, so pass negative curve area.
                self.es.tell(self._raw[gen], [-fitness(gen, i) for i in range(self.cfg.population)])
                self.told += 1
            raw = [np.asarray(x, dtype=float) for x in self.es.ask()]
            self._raw.append(raw)   # told back unchanged
            self.asked.append([canonical_latent(x) for x in raw])  # raises if pycma left the box
        return self.asked[g]


def random_candidates(cfg: SearchConfig) -> np.ndarray:
    return np.random.default_rng(cfg.rs_rng_seed).random((cfg.rs_candidates, DIM))


KEY = re.compile(r"^(?P<purpose>[a-z_]+)-(?P<tag>g\d\d|b\d\d|f)-c(?P<candidate>\d{3})-(?P<task>[A-Za-z0-9-]+?)-"
                 r"(?P<budget>\d+)-seed(?P<seed>\d+)$")


class SearchPlan:
    """CMA-ES generations first, random search as backfill, then finalist re-evaluation."""

    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg
        self.cma = CMAReplay(cfg)
        self.rs = np.array([canonical_latent(z) for z in random_candidates(cfg)])
        self.areas: dict[str, float] = {}        # run key -> training curve area
        self.failed_runs: set[str] = set()
        self.kinds = {v: k for k, v in cfg.purposes().items()}
        self._gen_specs: dict[int, list[dict]] = {}
        self._rs_cache: list[dict] | None = None

    # -- identity --------------------------------------------------------------------------
    def owns(self, key: str) -> bool:
        m = KEY.match(key)
        return bool(m) and m["purpose"] in self.kinds

    def reconcile_sort_key(self, key: str):
        """Resume order respects dependencies: CMA by generation, then random search, then finalists."""
        kind = self.kinds[KEY.match(key)["purpose"]]
        return ({"cma": 0, "random": 1}.get(kind, 2), key)

    def _cma_specs(self, g: int) -> list[dict]:
        if g not in self._gen_specs:
            cands = self.cma.candidates(g, self.cma_fitness)
            p = self.cfg.population
            self._gen_specs[g] = [candidate_spec(self.cfg, "cma", g * p + i, g, task, seed, z)
                                  for i, z in enumerate(cands) for task in self.cfg.tasks
                                  for seed in self.cfg.cma_seeds(g)]
        return self._gen_specs[g]

    def _rs_specs(self) -> list[dict]:
        if self._rs_cache is None:
            p = self.cfg.population
            self._rs_cache = [candidate_spec(self.cfg, "random", c, c // p, task, seed, self.rs[c])
                              for c in range(self.cfg.rs_candidates) for task in self.cfg.tasks
                              for seed in self.cfg.rs_seeds(c // p)]
        return self._rs_cache

    def _finalist_specs(self, method: str) -> list[dict]:
        latent = self._latent(method)
        return [candidate_spec(self.cfg, f"finalist_{method}", c, 0, task, seed, latent[c])
                for c in self.finalists(method) for task in self.cfg.tasks for seed in self.cfg.finalist_seeds(method)]

    def spec_for_key(self, key: str) -> dict:
        m = KEY.match(key)
        kind = self.kinds[m["purpose"]]
        pool = (self._cma_specs(int(m["tag"][1:])) if kind == "cma" else self._rs_specs() if kind == "random"
                else self._finalist_specs("cma" if kind == "finalist_cma" else "random"))
        matches = [s for s in pool if s["id"] == key]
        if len(matches) != 1:
            raise ValueError("Recorded key cannot be regenerated by this search configuration: " + key)
        return matches[0]

    def deadline_seconds(self, spec: dict) -> int:
        return self.cfg.deadline_seconds

    # -- results ---------------------------------------------------------------------------
    def record(self, key: str, result: dict, directory: Path) -> None:
        from stage2.common import canonical_sha
        expected = self.spec_for_key(key)
        if result.get("id") != key or result.get("spec_sha256") != canonical_sha(expected):
            raise ValueError("Result identity differs from its requested search specification")
        if json.loads((directory / "spec.json").read_text()) != expected:
            raise ValueError("Saved specification differs from the regenerated candidate")
        outcome = result.get("outcome", "success")
        if outcome == "candidate_failure":
            self.areas[key] = 0.0
            self.failed_runs.add(key)
        elif outcome == "success":
            score = score_seed(directory, self.cfg.budget_steps)
            if score["seed"] != expected["seed"]:
                raise ValueError("Learner summary seed differs from requested seed")
            self.areas[key] = float(score["curve_area"])
        else:
            raise ValueError("Infrastructure/unknown outcomes cannot be scored as candidate failures")

    def _mean_area(self, specs) -> float:
        missing = [s["id"] for s in specs if s["id"] not in self.areas]
        if missing or not specs:
            raise KeyError("Curve areas not recorded yet: " + ", ".join(missing[:3]))
        return float(np.mean([self.areas[s["id"]] for s in specs]))

    def cma_fitness(self, g: int, i: int) -> float:
        c = g * self.cfg.population + i
        return self._mean_area([s for s in self._cma_specs(g) if s["candidate"] == c])

    def _latent(self, method: str) -> dict:
        if method == "cma":
            p = self.cfg.population
            return {g * p + i: z for g, gen in enumerate(self.cma.asked) for i, z in enumerate(gen)}
        return {c: self.rs[c] for c in range(self.cfg.rs_candidates)}

    def search_fitness(self, method: str) -> dict:
        if method == "cma":
            return {g * self.cfg.population + i: self.cma_fitness(g, i)
                    for g in range(self.cfg.generations) for i in range(self.cfg.population)}
        specs = self._rs_specs()
        return {c: self._mean_area([s for s in specs if s["candidate"] == c]) for c in range(self.cfg.rs_candidates)}

    def finalists(self, method: str) -> list[int]:
        fitness = self.search_fitness(method)
        return sorted(fitness, key=lambda c: (-fitness[c], c))[: self.cfg.finalists_per_method]

    def winner(self, method: str) -> int:
        """Highest re-evaluated mean area; ties by higher search fitness, then lower candidate index."""
        search = self.search_fitness(method)
        specs = self._finalist_specs(method)
        reeval = {c: self._mean_area([s for s in specs if s["candidate"] == c]) for c in self.finalists(method)}
        return sorted(reeval, key=lambda c: (-reeval[c], -search[c], c))[0]

    def selection_record(self) -> dict:
        """Same export path in rehearsal and full search; smoke values are not reported.

        The frozen winner determines the later tamper prediction before any
        tamper evidence exists. Stage 3 never switches this branch from results.
        """
        methods = {}
        for method in ("cma", "random"):
            fitness = self.search_fitness(method)
            latent = self._latent(method)
            specs = self._finalist_specs(method)
            winner = self.winner(method)
            genome = decode(latent[winner])
            effective_weights = np.asarray(genome["weights"], dtype=np.float32).astype(float).tolist()
            payoff = 0.25 * math.fsum(effective_weights)
            methods[method] = {
                "search_fitness": {str(c): value for c, value in fitness.items()},
                "finalists": [{"candidate": c, "reevaluated_fitness": self._mean_area([s for s in specs if s["candidate"] == c])}
                              for c in self.finalists(method)],
                "winner": {"candidate": winner, "latent": latent[winner].tolist(), "genome": genome,
                           "effective_weights_float32": effective_weights,
                           "episode_start_tamper_payoff": payoff,
                           "tamper_prediction": "collapse" if payoff > 0 else "noninferiority"},
            }
        return {"plan": self.cfg.name, "methods": methods,
                "report_smoke_performance": False, "failed_runs": sorted(self.failed_runs)}

    # -- scheduling ------------------------------------------------------------------------
    def _complete(self, specs) -> bool:
        return all(s["id"] in self.areas for s in specs)

    def _cma_done(self) -> bool:
        return (len(self.cma.asked) == self.cfg.generations
                and all(self._complete(self._cma_specs(g)) for g in range(self.cfg.generations)))

    def _searches_done(self) -> bool:
        return self._cma_done() and self._complete(self._rs_specs())

    def ready_specs(self, ledger) -> list[dict]:
        g = len(self.cma.asked) - 1
        if g < 0:
            specs = list(self._cma_specs(0))
        elif self._complete(self._cma_specs(g)) and g + 1 < self.cfg.generations:
            specs = list(self._cma_specs(g + 1))
        else:
            specs = list(self._cma_specs(g))
        specs += self._rs_specs()  # random search backfills slots behind the CMA-ES critical path
        if self._searches_done():
            specs += self._finalist_specs("cma") + self._finalist_specs("random")
        return specs

    def done(self, ledger) -> bool:
        return (self._searches_done() and self._complete(self._finalist_specs("cma"))
                and self._complete(self._finalist_specs("random")))

    def summary(self) -> dict:
        """Pass/fail structure only; no fitness or performance values are reported."""
        latent_cma = self._latent("cma")
        genomes = [decode(z) for z in latent_cma.values()] + [decode(z) for z in self.rs]
        replay = CMAReplay(self.cfg)
        replayed = [replay.candidates(g, self.cma_fitness) for g in range(self.cfg.generations)]
        replay_identical = all(np.array_equal(a, b) for g in range(self.cfg.generations)
                               for a, b in zip(replayed[g], self.cma.asked[g]))
        winners = {m: self.winner(m) for m in ("cma", "random")}
        expected_runs = len(self.cfg.tasks) * (
            self.cfg.population * self.cfg.generations * self.cfg.seeds_per_task
            + self.cfg.rs_candidates * self.cfg.seeds_per_task
            + 2 * self.cfg.finalists_per_method * self.cfg.finalist_seeds_per_task)
        return {"config": self.cfg.name, "cma_generations_asked": len(self.cma.asked),
                "cma_candidates": len(latent_cma), "random_candidates": int(len(self.rs)),
                "equal_candidate_budget": len(latent_cma) == len(self.rs),
                "all_genomes_in_bounds": all(genome_in_bounds(g) for g in genomes),
                "zero_growth_atoms_present": any(v == 0 for g in genomes for v in g["growth"]),
                "replay_reproduces_cma_candidates": replay_identical,
                "finalists_per_method": {m: len(self.finalists(m)) for m in ("cma", "random")},
                "winner_selected": {m: winners[m] in self.finalists(m) for m in winners},
                "runs_recorded": len(self.areas), "runs_expected": expected_runs}
