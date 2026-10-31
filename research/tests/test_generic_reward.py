"""Stage-1 verification; execute only after the pre-registration is locked.

Run with unittest or pytest. STAGE1_UNIT_OUTPUT selects the recording directory
(default results/unit-tests). Reduction records fixed uniform-random DoorKey
actions, including terminal steps, without filtering trajectories. It checks
the numerical kernel on archived detector events; complete generic detector
semantics, including the intentional terminal/box fixes, are checked separately.
An end-to-end comparison on the same trajectories records every mismatch,
without interpreting a mismatch as a pass of the user's reduction criterion.
"""

import hashlib
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Box, Door, Key

from generic_reward import (
    EVENTS, GenericEventCapture, GenericEventDetector, GenericHomeostatic,
    CorrectedLegacyEventDetector, apply_tamper_cost, signed_return_bounds,
)
from stage1.corrected_ppo import corrected_tree, load_ppo as load_corrected_ppo, SOURCES

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "handoff-cross-platform/code/final-round5"
OUT = Path(os.environ.get("STAGE1_UNIT_OUTPUT", ROOT / "results/unit-tests"))
SEEDS = (905, 906, 907, 908, 909)


def legacy_modules():
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    intrinsic = load("stage1_legacy_intrinsic", LEGACY / "intrinsic.py")
    saved = sys.modules.get("intrinsic")
    try:
        sys.modules["intrinsic"] = intrinsic
        ppo = load("stage1_legacy_ppo", LEGACY / "ppo.py")
    finally:
        if saved is None:
            sys.modules.pop("intrinsic", None)
        else:
            sys.modules["intrinsic"] = saved
    return intrinsic, ppo


def generic_config(weights=(1, 1, 0, 1), relief=(0.1, 1, 0, 1), growth=(1 / 160,) * 4,
                   tamper=False):
    return {"intrinsic": dict(weights=weights, relief=relief, growth=growth, h0=0.5, tamper=tamper)}


def final_observation(obs, done, info):
    nxt = obs.copy()
    if done.any():
        nxt[done] = np.stack(info["final_obs"][done])
    return nxt


def captured_vec_env(ppo, seed):
    original_factory = ppo.make_env

    def make_captured(*args, **kwargs):
        original_thunk = original_factory(*args, **kwargs)
        return lambda: GenericEventCapture(original_thunk())

    try:
        ppo.make_env = make_captured
        return ppo.make_vec_env("MiniGrid-DoorKey-8x8-v0", 1, seed=seed)
    finally:
        ppo.make_env = original_factory


def random_generic_rollout(seed, steps=1000):
    """Record true generic events for one fixed uniform random policy."""
    _, ppo = legacy_modules()
    envs = gym.vector.SyncVectorEnv(
        [lambda: GenericEventCapture(ppo.make_env("MiniGrid-DoorKey-8x8-v0")())],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs.reset(seed=seed * 1000)
    detector = GenericEventDetector(envs)
    rng = np.random.default_rng(seed)
    events, dones, actions = [], [], []
    try:
        for _ in range(steps):
            act = rng.integers(0, 7, size=1)
            obs, _, term, trunc, info = envs.step(act)
            done = term | trunc
            ev = detector.step(torch.tensor(final_observation(obs, done, info)), done)
            events.append(ev[0]); dones.append(bool(done[0])); actions.append(int(act[0]))
    finally:
        envs.close()
    return np.asarray(events), np.asarray(dones), np.asarray(actions)


class RewardReductionTests(unittest.TestCase):
    def test_recorded_legacy_event_reduction_five_seeds(self):
        old, ppo = legacy_modules()
        OUT.mkdir(parents=True, exist_ok=True)
        summaries = []
        full_comparisons = []
        for seed in SEEDS:
            with self.subTest(seed=seed):
                envs = captured_vec_env(ppo, seed)
                legacy_cfg = {"intrinsic": dict(growth=[1 / 160] * 3, satiety=[0.1, 1, 1],
                                               h0=0.5, tamper=False)}
                legacy = old.Homeostatic(legacy_cfg, (7, 7, 3), "cpu")
                # Preserve the archive's attach-before-seeded-reset ordering.
                legacy.attach(envs)
                generic = GenericHomeostatic(generic_config())
                generic.reset(1)
                full_generic = GenericHomeostatic(generic_config())
                envs.reset(seed=seed * 1000)
                full_generic.attach(envs)
                corrected_legacy = load_corrected_ppo().archived_intrinsic.Homeostatic(legacy_cfg, (7, 7, 3), "cpu")
                corrected_legacy.attach(envs)
                rng = np.random.default_rng(seed)
                observations, actions, dones, events, expected, actual = [], [], [], [], [], []
                full_events, full_rewards, diagnostics, corrected_rewards = [], [], [], []
                try:
                    for _ in range(1000):
                        act = rng.integers(0, 7, size=1)
                        obs, _, term, trunc, info = envs.step(act)
                        done = term | trunc
                        nxt = final_observation(obs, done, info)
                        ref = legacy.reward(torch.tensor(nxt), done, act).numpy()
                        ev = np.zeros((1, 4), dtype=np.float32)
                        ev[:, (0, 1, 3)] = legacy.events
                        got = generic.transition(ev, done, act)
                        before = full_generic.h.copy()
                        full_reward = full_generic.reward(torch.tensor(nxt), done, act).numpy()
                        corrected_reward = corrected_legacy.reward(torch.tensor(nxt), done, act).numpy()
                        np.testing.assert_array_equal(full_reward.view(np.uint32), corrected_reward.view(np.uint32))
                        corrected_rewards.append(corrected_reward[0])
                        diagnostics.append(envs.envs[0].last_diagnostics.copy())
                        full_events.append(full_generic.events[0].copy())
                        full_rewards.append(full_reward[0])
                        # Assert the specified generic dynamics/reward on the
                        # events it actually detected, rather than asserting
                        # identity to known incomplete archived detectors.
                        after = np.clip(before + full_generic.growth
                                        - full_generic.relief * full_generic.events, 0, 1)
                        formula = ((before[:, (0, 1, 3)] ** 2).sum(-1)
                                   - (after[:, (0, 1, 3)] ** 2).sum(-1))
                        np.testing.assert_array_equal(full_reward, formula)
                        np.testing.assert_array_equal(full_generic.pre_reset_h_next, after)
                        observations.append(nxt[0]); actions.append(act[0]); dones.append(done[0])
                        events.append(ev[0]); expected.append(ref[0]); actual.append(got[0])
                finally:
                    envs.close()
                expected, actual = np.asarray(expected), np.asarray(actual)
                full_events, full_rewards = np.asarray(full_events), np.asarray(full_rewards)
                target = OUT / f"reduction-seed{seed}.npz"
                np.savez_compressed(target, observations=np.asarray(observations), actions=np.asarray(actions),
                                    done=np.asarray(dones), archived_events=np.asarray(events),
                                    archived_rewards=expected, generic_rewards=actual,
                                    full_generic_events=full_events, full_generic_rewards=full_rewards,
                                    corrected_three_need_rewards=np.asarray(corrected_rewards))
                same = expected.tobytes() == actual.tobytes()
                summaries.append(dict(seed=seed, scalar_steps=len(actions), bit_identical=same,
                                      terminal_steps=int(np.count_nonzero(dones)),
                                      event_counts=np.asarray(events).sum(0).tolist(),
                                      file=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
                mismatches = np.flatnonzero(expected.view(np.uint32) != full_rewards.view(np.uint32)) + 1
                shared_event_difference = np.any(np.asarray(events)[:, (0, 1, 3)]
                                                 != full_events[:, (0, 1, 3)], axis=1)
                event_steps = np.flatnonzero(shared_event_difference) + 1
                event_differences = np.asarray(events) != full_events
                first_done = next((i + 1 for i, done in enumerate(dones) if done), len(dones))
                detail = []
                for step in event_steps:
                    i = int(step) - 1
                    detail.append(dict(step_1based=int(step), first_episode=int(step) <= first_done,
                                       terminal=bool(dones[i]), action=int(actions[i]),
                                       archived_events=np.asarray(events)[i].tolist(),
                                       corrected_events=full_events[i].tolist(), physical=diagnostics[i]))
                full_comparisons.append(dict(
                    seed=seed, scalar_steps=len(actions), bit_identical=len(mismatches) == 0,
                    reward_mismatch_count=len(mismatches), reward_mismatch_steps_1based=mismatches.tolist(),
                    shared_event_mismatch_count=len(event_steps),
                    shared_event_mismatch_steps_1based=event_steps.tolist(),
                    full_generic_event_counts=full_events.sum(0).tolist(),
                    event_disagreement_counts={name: int(event_differences[:, j].sum())
                                               for j, name in enumerate(EVENTS)},
                    disagreement_details=detail,
                    corrected_three_need_and_generic_rewards_bit_identical=True,
                    terminal_steps_1based=(np.flatnonzero(dones) + 1).tolist(),
                    file=target.name,
                ))
                self.assertEqual(expected.dtype, np.float32)
                self.assertEqual(expected.tobytes(), actual.tobytes())
        (OUT / "reduction-summary.json").write_text(json.dumps({
            "scope": "numerical kernel on recorded archived events; detector differences tested separately",
            "events": EVENTS, "seeds": summaries,
        }, indent=2) + "\n")
        (OUT / "detector-trajectory-comparison.json").write_text(json.dumps({
            "scope": "full generic detector plus reward versus untouched archived module; same trajectories",
            "scope_amendment": "Corrected detector differences are reported, not hidden by changing actions; arithmetic reduction uses recorded paper events.",
            "successful_target_pickup_excluded": True,
            "events": EVENTS, "seeds": full_comparisons,
        }, indent=2) + "\n")

    def test_growth_is_clipped_after_relief(self):
        module = GenericHomeostatic(generic_config(weights=(1, 0, 0, 0),
                                                   relief=(0.1, 0, 0, 0), growth=(0.05, 0, 0, 0)))
        module.reset(1)
        module.h[0, 0] = 1
        module.transition(np.array([[1, 0, 0, 0]], dtype=np.float32))
        expected = np.clip(np.float32(1) + np.float32(0.05) - np.float32(0.1), 0, 1)
        self.assertEqual(module.h[0, 0], expected)
        self.assertNotEqual(module.h[0, 0], np.float32(0.9))

    def test_zero_growth_never_refills_and_zero_weight_still_has_need(self):
        module = GenericHomeostatic(generic_config(weights=(1, 0, 0, 0), growth=(0, 0, 0, 0),
                                                   relief=(1, 1, 1, 1)))
        module.reset(1)
        module.transition(np.ones((1, 4), dtype=np.float32))
        np.testing.assert_array_equal(module.h, np.zeros((1, 4), dtype=np.float32))
        for _ in range(10):
            np.testing.assert_array_equal(module.transition(np.zeros((1, 4))), [0])
        np.testing.assert_array_equal(module.h, np.zeros((1, 4), dtype=np.float32))

    def test_tamper_reset_cost_and_terminal_reset_order(self):
        module = GenericHomeostatic(generic_config(weights=(1, -1, 0, 0.5), tamper=True))
        module.reset(2)
        module.h[:] = np.asarray([[0.3, 0.4, 0.8, 0.9], [0.6, 0.2, 0.7, 0.1]], dtype=np.float32)
        old_potential = module.potential(module.h)
        raw = np.asarray([0.0, 1.0])
        act = np.asarray([6, 6])
        r = module.transition(np.zeros((2, 4)), done=[False, True], actions=act)
        np.testing.assert_array_equal(r, old_potential)
        np.testing.assert_array_equal(module.pre_reset_h_next, np.zeros((2, 4)))
        np.testing.assert_array_equal(module.h[0], [0, 0, 0, 0])
        np.testing.assert_array_equal(module.h[1], [0.5] * 4)
        for cost in (0, 0.01, 0.1, 1):
            paid = apply_tamper_cost(raw, act, cost, enabled=True)
            np.testing.assert_array_equal(paid, raw - cost)
            for coef in (0, 0.001, 1):
                np.testing.assert_array_equal(paid + coef * r, raw - cost + coef * r)
        np.testing.assert_array_equal(raw, [0, 1])
        np.testing.assert_array_equal(apply_tamper_cost(raw, act, 0.1, enabled=False), raw)
        np.testing.assert_array_equal(apply_tamper_cost(raw, [6, 2], 0.1, enabled=True), [-0.1, 1])

    def test_invalid_parameters_rejected(self):
        for key, bad in (("weights", [1.1, 0, 0, 0]), ("relief", [-0.1, 0, 0, 0]),
                         ("growth", [1 / 1000, 0, 0, 0]), ("growth", [0.1, 0, 0, 0]),
                         ("weights", [float("nan"), 0, 0, 0]), ("h0", 0.4)):
            cfg = generic_config()
            cfg["intrinsic"][key] = bad
            with self.subTest(key=key, value=bad), self.assertRaises(ValueError):
                GenericHomeostatic(cfg)

    def test_signed_ceiling_on_fixed_random_rollouts(self):
        # Fixed reward vectors exercise signs without searching or comparing
        # reward fitness. The same random action trajectories are replayed.
        cases = ((1, 1, 1, 1), (-1, -1, -1, -1), (1, -1, 0.5, -0.5), (0, 0, 0, 0))
        rows = []
        OUT.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            events, dones, actions = random_generic_rollout(seed)
            for weights in cases:
                for gamma in (0.5, 0.99, 1.0):
                    for tamper in (False, True):
                        module = GenericHomeostatic(generic_config(weights=weights, relief=(0.1, 1, 1, 1),
                                                                   tamper=tamper))
                        module.reset(1)
                        lower, upper = signed_return_bounds(weights)
                        total, discount, episode, n = 0.0, 1.0, 0, 0
                        for i, (ev, done, action) in enumerate(zip(events, dones, actions)):
                            reward = float(module.transition(ev[None], [done], [action])[0])
                            total += discount * reward
                            discount *= gamma
                            n += 1
                            # Every prefix is an episode segment from h0; the
                            # final open episode is checked as a finite prefix.
                            self.assertLessEqual(total, upper + 2e-5)
                            self.assertGreaterEqual(total, lower - 2e-5)
                            if done or i == len(events) - 1:
                                rows.append(dict(seed=seed, weights=weights, gamma=gamma, tamper=tamper,
                                                 episode=episode, steps=n, return_value=total,
                                                 lower=lower, upper=upper, terminal=bool(done)))
                                total, discount, n = 0.0, 1.0, 0
                                episode += 1
        (OUT / "bound-check.json").write_text(json.dumps({
            "tolerance": 2e-5, "parameter_search": False, "episode_checks": rows,
        }, indent=2) + "\n")
        self.assertEqual(signed_return_bounds((-1, -1, -1, -1)), (-1, 3))
        self.assertEqual(signed_return_bounds((1, 1, 0, 1)), (-2.25, 0.75))


class ScalarFixture(gym.Env):
    """Tiny deterministic transition fixture for events discarded by autoreset."""
    class Actions:
        pickup, drop, toggle = 3, 4, 5
    actions = Actions()

    def __init__(self, mode):
        self.mode = mode
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(0, 255, shape=(7, 7, 3), dtype=np.uint8)
        self.reset()

    @property
    def front_pos(self):
        return np.array((2, 1))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = Grid(4, 4)
        self.carrying = None
        if self.mode == "box":
            self.grid.set(2, 1, Box("yellow", contains=Key("blue")))
        elif self.mode in ("door", "timeout_door"):
            self.grid.set(2, 1, Door("yellow", is_locked=True))
        elif self.mode == "drop":
            self.carrying = Ball("blue")
        return np.zeros((7, 7, 3), dtype=np.uint8), {}

    def step(self, action):
        terminated = False
        truncated = self.mode in ("timeout_pickup", "timeout_door")
        if self.mode in ("pickup", "terminal_pickup", "timeout_pickup") and action == self.actions.pickup:
            self.carrying = Ball("blue")
            terminated = self.mode == "terminal_pickup"
        elif self.mode == "drop" and action == self.actions.drop:
            self.grid.set(2, 1, self.carrying)
            self.carrying = None
        elif self.mode == "box" and action == self.actions.toggle:
            self.grid.set(2, 1, self.grid.get(2, 1).contains)
        elif self.mode in ("door", "timeout_door") and action == self.actions.toggle:
            door = self.grid.get(2, 1)
            door.is_locked, door.is_open = False, True
        return np.ones((7, 7, 3), dtype=np.uint8), float(terminated), terminated, truncated, {}


class GenericDetectorTests(unittest.TestCase):
    def test_any_object_pickup_drop_toggle_and_terminal_capture(self):
        for mode, action, physical in (("pickup", 3, [1, 0, 0]),
                                       ("terminal_pickup", 3, [0, 0, 0]),
                                       ("timeout_pickup", 3, [1, 0, 0]),
                                       ("timeout_door", 5, [0, 0, 1]),
                                       ("drop", 4, [0, 1, 0]),
                                       ("box", 5, [0, 0, 1]),
                                       ("door", 5, [0, 0, 1])):
            with self.subTest(mode=mode):
                envs = gym.vector.SyncVectorEnv([lambda: GenericEventCapture(ScalarFixture(mode))],
                                               autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
                envs.reset(seed=910)
                detector = GenericEventDetector(envs)
                obs, _, term, trunc, info = envs.step(np.asarray([action]))
                done = term | trunc
                got = detector.step(torch.tensor(final_observation(obs, done, info)), done)
                np.testing.assert_array_equal(got[0], [1] + physical)
                capture = envs.envs[0]
                self.assertEqual(capture.last_diagnostics["target_pickup_excluded"], mode == "terminal_pickup")
                if mode == "terminal_pickup":
                    self.assertTrue(capture.last_diagnostics["raw_pickup"])
                    self.assertEqual(capture.last_episode_counts["raw_pickups"], 1)
                    self.assertEqual(capture.last_episode_counts["eligible_pickups"], 0)
                if done[0]:
                    self.assertIsNone(envs.envs[0].unwrapped.carrying)
                    self.assertEqual(len(detector.seen[0]), 0)
                with self.assertRaises(RuntimeError):
                    detector.step(torch.tensor(obs), done)
                envs.close()

    def test_repeated_view_is_unpaid_until_episode_reset(self):
        envs = gym.vector.SyncVectorEnv([lambda: GenericEventCapture(ScalarFixture("pickup"))],
                                       autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        envs.reset(seed=910)
        detector = GenericEventDetector(envs)
        for expected in (1, 0, 0):
            obs, _, term, trunc, _ = envs.step(np.asarray([6]))
            got = detector.step(torch.tensor(obs), term | trunc)
            self.assertEqual(got[0, 0], expected)
        envs.close()

    def test_missing_capture_rejected(self):
        envs = gym.vector.SyncVectorEnv([lambda: ScalarFixture("pickup")])
        with self.assertRaises(ValueError):
            GenericEventDetector(envs)
        envs.close()

    def test_initial_door_cache_is_populated_after_reset(self):
        old, _ = legacy_modules()

        def make_empty_constructor():
            fixture = ScalarFixture("door")
            fixture.grid = Grid(4, 4)  # MiniGrid's actual constructor state.
            return GenericEventCapture(fixture)

        envs = gym.vector.SyncVectorEnv([make_empty_constructor],
                                       autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        legacy = old.EventDetector(envs)
        with self.assertRaises(ValueError):
            GenericEventDetector(envs)
        envs.reset(seed=910)
        corrected = GenericEventDetector(envs)
        obs, _, term, trunc, _ = envs.step([5])
        prior = legacy.step(torch.tensor(obs), term | trunc)
        current = corrected.step(torch.tensor(obs), term | trunc)
        self.assertEqual(prior[0, 2], 0)
        self.assertEqual(current[0, 3], 1)
        envs.close()

    def test_box_and_timeout_events_compare_both_detectors_on_same_transition(self):
        old, _ = legacy_modules()
        for mode, action, event_column in (("box", 5, 2), ("timeout_pickup", 3, 1),
                                          ("timeout_door", 5, 2)):
            with self.subTest(mode=mode):
                envs = gym.vector.SyncVectorEnv([lambda: GenericEventCapture(ScalarFixture(mode))],
                                               autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
                envs.reset(seed=910)
                legacy, corrected = old.EventDetector(envs), CorrectedLegacyEventDetector(envs)
                obs, _, term, trunc, info = envs.step([action])
                done = term | trunc
                final = torch.tensor(final_observation(obs, done, info))
                prior = legacy.step(final, done)
                current = corrected.step(final, done)
                self.assertEqual(prior[0, event_column], 0)
                self.assertEqual(current[0, event_column], 1)
                envs.close()


class CorrectedLearnerContractTests(unittest.TestCase):
    def test_only_attach_statement_moves_in_both_archived_learners(self):
        for source in SOURCES:
            with self.subTest(source=source):
                text = (ROOT / "handoff-cross-platform/code" / source / "ppo.py").read_text()
                original, corrected = ast.parse(text), corrected_tree(text)
                old_main = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "main")
                new_main = next(n for n in corrected.body if isinstance(n, ast.FunctionDef) and n.name == "main")
                is_attach = lambda n: isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "intrinsic.attach"
                old_main.body = [n for n in old_main.body if not is_attach(n)]
                new_indices = [i for i, n in enumerate(new_main.body) if is_attach(n)]
                self.assertEqual(len(new_indices), 1)
                reset = new_main.body[new_indices[0] - 1]
                self.assertEqual(ast.unparse(reset.value.func), "envs.reset")
                new_main.body = [n for n in new_main.body if not is_attach(n)]
                self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(corrected, include_attributes=False))

    def test_corrected_three_need_factory_preserves_dimension_and_tamper_arithmetic(self):
        for source in SOURCES:
            ppo = load_corrected_ppo(source=source)
            envs = ppo.make_vec_env("MiniGrid-DoorKey-8x8-v0", 1)
            envs.reset(seed=910)
            cfg = {"intrinsic": dict(kind="homeostatic", growth=[1 / 160] * 3,
                                     satiety=[0.1, 1, 1], h0=0.5, tamper=True)}
            module = ppo.make_intrinsic(cfg, (7,7,3), "cpu")
            module.attach(envs)
            self.assertEqual(module.state_dim, 3)
            self.assertEqual(tuple(module.state().shape), (1, 3))
            obs, _, term, trunc, info = envs.step([6])
            done = term | trunc
            reward = module.reward(torch.tensor(final_observation(obs, done, info)), done, [6])
            self.assertEqual(float(reward[0]), 0.75)
            np.testing.assert_array_equal(module.h, [[0,0,0]])
            envs.close()


if __name__ == "__main__":
    unittest.main()
