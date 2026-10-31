"""Four-event bounded intrinsic reward, with legacy-compatible float32 arithmetic.

The event order exposed to callers is new_view, pickup, drop, toggle. The paper's
equation (1) and archived code use ONE clip after growth minus relief, not a
separate clip of growth. Original files in handoff-cross-platform are untouched.

Install GenericEventCapture around each scalar env before constructing a vector
env. It observes the actual transition before SAME_STEP autoreset can discard
terminal pickups or box changes. The archived detector omits those events and
can retain references across an external reset; numerical legacy reduction is
therefore tested by replaying its recorded events, separately from detector tests.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from minigrid.core.world_object import Box, Door, Key

EVENTS = ("new_view", "pickup", "drop", "toggle")
TAMPER_ACTION = 6
MIN_GROWTH = 1 / 640
MAX_GROWTH = 1 / 20
H0 = 0.5


def _array4(value, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain four finite numbers in {EVENTS} order")
    return a


def signed_return_bounds(weights, h0=H0):
    """Episode-local lower/upper bounds for sum gamma**t * (V_t - V_t+1).

    Valid for every T >= 1 and 0 < gamma <= 1, including tamper transitions.
    With Vmin = sum(min(w, 0)), Vmax = sum(max(w, 0)), summation by parts gives
    V(h0)-Vmax <= G <= V(h0)-Vmin. Episode resets are not paid transitions.
    """
    w = _array4(weights, "weights")
    v0 = float(h0) ** 2 * float(w.sum())
    return v0 - float(np.maximum(w, 0).sum()), v0 - float(np.minimum(w, 0).sum())


def apply_tamper_cost(extrinsic, actions, cost=0.0, enabled=False):
    """Charge cost to extrinsic reward only, exactly as archived PPO does.

    The caller must retain raw environment rewards for task-success metrics.
    Intrinsic coefficient never multiplies this charge.
    """
    if not np.isfinite(cost) or cost < 0:
        raise ValueError("tamper cost must be finite and nonnegative")
    return extrinsic - cost * (np.asarray(actions) == TAMPER_ACTION) if enabled and cost else extrinsic


class GenericEventCapture(gym.Wrapper):
    """Capture physical events before an outer vector env performs autoreset.

    Reset deliberately leaves last_transition intact: SAME_STEP calls reset
    before the reward module reads the terminal transition. The next step
    replaces it. Captures reference scalar env state directly, so an explicit
    reset cannot leave stale door references in the generic detector.
    """

    def __init__(self, env):
        super().__init__(env)
        self.last_transition = None
        self.transition_serial = 0
        self.episode_serial = 0
        self.has_reset = False
        self.initial_locked_ids = set()
        self.last_diagnostics = None
        self.last_episode_counts = None

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.episode_serial += 1
        self.has_reset = True
        self.initial_locked_ids = {id(o) for o in self.unwrapped.grid.grid
                                   if isinstance(o, Door) and o.is_locked}
        self.episode_counts = {name: 0 for name in (
            "raw_pickups", "eligible_pickups", "target_pickups_excluded", "drops",
            "toggles", "door_changes", "box_changes", "key_pickups", "locked_door_opens")}
        return result

    def step(self, action):
        if not self.has_reset:
            raise RuntimeError("Reset each environment before stepping or attaching its detector")
        u = self.unwrapped
        episode_serial = self.episode_serial
        carry_before = u.carrying
        doors = [(o, bool(o.is_open), bool(o.is_locked))
                 for o in u.grid.grid if isinstance(o, Door)]
        front = tuple(u.front_pos)
        front_obj = u.grid.get(*front)
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_pickup = carry_before is None and u.carrying is not None
        # Explicit Stage-1 correction: success-triggering target pickups are
        # diagnostic events but receive no pickup bonus. A timeout alone does
        # not make a pickup ineligible. Reward is the raw environment reward.
        target_pickup = bool(raw_pickup and terminated and float(reward) > 0)
        pickup = bool(raw_pickup and not target_pickup)
        key_pickup = bool(pickup and isinstance(u.carrying, Key))
        drop = (int(action) == int(u.actions.drop)
                and carry_before is not None and u.carrying is None)
        door_change = any((old_open, old_locked) != (bool(o.is_open), bool(o.is_locked))
                          for o, old_open, old_locked in doors)
        locked_door_open = any(id(o) in self.initial_locked_ids and not old_open and o.is_open
                               for o, old_open, _ in doors)
        # Toggling a box removes it or replaces it with its contents. Carrying
        # or dropping a box is not a toggle event.
        box_change = (int(action) == int(u.actions.toggle) and isinstance(front_obj, Box)
                      and u.grid.get(*front) is not front_obj)
        self.transition_serial += 1
        self.last_episode_serial = episode_serial
        self.last_transition = np.asarray((pickup, drop, door_change or box_change), dtype=np.float32)
        self.last_diagnostics = dict(raw_pickup=bool(raw_pickup), target_pickup_excluded=target_pickup,
                                     eligible_pickup=pickup, key_pickup=key_pickup, drop=bool(drop),
                                     door_change=bool(door_change), box_change=bool(box_change),
                                     locked_door_open=bool(locked_door_open), terminated=bool(terminated),
                                     truncated=bool(truncated), raw_extrinsic_reward=float(reward))
        increments = dict(raw_pickups=raw_pickup, eligible_pickups=pickup,
                          target_pickups_excluded=target_pickup, drops=drop,
                          toggles=door_change or box_change, door_changes=door_change,
                          box_changes=box_change, key_pickups=key_pickup,
                          locked_door_opens=locked_door_open)
        for name, value in increments.items():
            self.episode_counts[name] += int(value)
        self.last_episode_counts = self.episode_counts.copy()
        return obs, reward, terminated, truncated, info


def _capture(env):
    while isinstance(env, gym.Wrapper):
        if isinstance(env, GenericEventCapture):
            return env
        env = env.env
    raise ValueError("GenericEventCapture must wrap every scalar env before vectorization")


class GenericEventDetector:
    """Exact-view episodic novelty plus captured physical events.

    As in the archive, the reset observation is not preinserted into seen: the
    first post-action view is eligible for new_view. On terminal steps the caller
    supplies final_obs, not SAME_STEP's next-episode reset observation.
    """

    def __init__(self, envs):
        self.captures = [_capture(e) for e in envs.envs]
        if not all(c.has_reset for c in self.captures):
            raise ValueError("Attach the corrected detector only after the initial seeded reset")
        self.n = len(self.captures)
        self.seen = [set() for _ in self.captures]
        self.consumed_serial = [c.transition_serial for c in self.captures]
        self.observed_episodes = [c.episode_serial for c in self.captures]

    def step(self, next_obs, done=None):
        obs = next_obs.detach().cpu().numpy() if isinstance(next_obs, torch.Tensor) else np.asarray(next_obs)
        done = np.zeros(self.n, dtype=bool) if done is None else np.asarray(done, dtype=bool)
        ev = np.zeros((self.n, 4), dtype=np.float32)
        for i, c in enumerate(self.captures):
            if c.last_transition is None or self.consumed_serial[i] == c.transition_serial:
                raise RuntimeError("one fresh env transition is required for each detector step")
            self.consumed_serial[i] = c.transition_serial
            if c.last_episode_serial != self.observed_episodes[i]:
                self.seen[i].clear()
                self.observed_episodes[i] = c.last_episode_serial
            view = obs[i].tobytes()
            ev[i, 0] = view not in self.seen[i]
            self.seen[i].add(view)
            ev[i, 1:] = c.last_transition
            if done[i]:
                self.seen[i].clear()
        return ev


class CorrectedLegacyEventDetector(GenericEventDetector):
    """Three-channel detector with the archive's selectable event semantics.

    The default maps corrected generic events to explore, grasp, open. The
    locked_door_key mode keeps key-only pickups and originally locked door
    openings. Both modes retain the successful-target-pickup exclusion.
    """

    def __init__(self, envs, mode="doorkey"):
        if mode not in ("doorkey", "locked_door_key"):
            raise ValueError(mode)
        super().__init__(envs)
        self.mode = mode

    def step(self, next_obs, done=None):
        generic = super().step(next_obs, done)
        events = generic[:, (0, 1, 3)].copy()
        if self.mode == "locked_door_key":
            for i, capture in enumerate(self.captures):
                events[i, 1] = capture.last_diagnostics["key_pickup"]
                events[i, 2] = capture.last_diagnostics["locked_door_open"]
        return events


class GenericHomeostatic:
    """PPO-compatible bounded signed reward with four observed needs.

    Config keys: intrinsic.weights, relief, growth, h0 (fixed 0.5), tamper.
    All arrays use EVENTS order. Multiplication and accumulation are float32.
    V is summed in legacy order (new_view, pickup, toggle), then drop, with zero
    weights omitted. Computing V(h)-V(h_next), rather than summing per-coordinate
    differences, preserves the archived three-need rounding in reduction tests.
    Both expressions implement the same specified mathematical reward.
    """

    state_dim = 4

    def __init__(self, cfg, obs_shape=None, device="cpu"):
        ic = cfg["intrinsic"]
        weights = _array4(ic["weights"], "weights")
        relief = _array4(ic["relief"], "relief")
        growth = _array4(ic["growth"], "growth")
        if not ((-1 <= weights) & (weights <= 1)).all():
            raise ValueError("weights must be in [-1, 1]")
        if not ((0 <= relief) & (relief <= 1)).all():
            raise ValueError("relief must be in [0, 1]")
        if not ((growth == 0) | ((MIN_GROWTH <= growth) & (growth <= MAX_GROWTH))).all():
            raise ValueError("growth must be 0 or in [1/640, 1/20]")
        if ic.get("h0", H0) != H0:
            raise ValueError("h0 is fixed at 0.5")
        self.device = device
        self.weights = weights.astype(np.float32)
        self.relief = relief.astype(np.float32)
        self.growth = growth.astype(np.float32)
        self.tamper = bool(ic.get("tamper", False))
        self._active = [i for i in (0, 1, 3, 2) if self.weights[i] != 0]

    def reset(self, n):
        self.h = np.full((n, 4), H0, dtype=np.float32)
        self.events = np.zeros((n, 4), dtype=np.float32)

    def attach(self, envs):
        self.det = GenericEventDetector(envs)
        self.reset(envs.num_envs)

    def potential(self, h):
        if not self._active:
            return np.zeros(h.shape[0], dtype=np.float32)
        # Copy into a contiguous row before np.sum, matching the archived
        # three-column reduction exactly when drop has weight zero.
        active_h = np.ascontiguousarray(h[:, self._active])
        return ((active_h ** 2) * self.weights[self._active]).sum(axis=-1)

    def transition(self, events, done=None, actions=None):
        """Replay a recorded event batch; returns float32 rewards before reset.

        Exposed to test reduction independently of intentional detector fixes.
        All four needs evolve, including zero-weight needs.
        """
        ev = np.asarray(events, dtype=np.float32)
        if ev.shape != self.h.shape or not np.isin(ev, [0, 1]).all():
            raise ValueError("events must be a binary [num_envs, 4] array")
        h_next = np.clip(self.h + self.growth - self.relief * ev, 0.0, 1.0)
        if self.tamper and actions is not None:
            h_next[np.asarray(actions) == TAMPER_ACTION] = 0.0
        reward = self.potential(self.h) - self.potential(h_next)
        self.events = ev.copy()
        self.pre_reset_h_next = h_next.copy()
        self.h = h_next
        if done is not None:
            self.h[np.asarray(done, dtype=bool)] = H0
        return reward

    def reward(self, next_obs, done=None, actions=None):
        return torch.tensor(self.transition(self.det.step(next_obs, done), done, actions), device=self.device)

    def state(self):
        return torch.tensor(self.h, device=self.device)

    def update(self, obs):
        return {f"need/{name}": float(self.h[:, j].mean()) for j, name in enumerate(EVENTS)}

    def parameters(self):
        return []
