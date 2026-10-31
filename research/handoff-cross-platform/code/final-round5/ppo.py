"""Single-file PPO for MiniGrid, CleanRL style, with exploration diagnostics.

Usage: python ppo.py --seed 0 --out results/seed0 [--config config.toml] [--total-timesteps N]

Writes <out>/metrics.jsonl (one row per log interval) and <out>/summary.json.
Read top to bottom: env setup -> network -> rollout -> GAE -> PPO update -> eval.
"""

import argparse
import json
import os
import random
import time
import tomllib
from collections import deque
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  (registers MiniGrid-* envs)
import numpy as np
import torch
import torch.nn as nn
from minigrid.core.world_object import Door, Key, Wall
from minigrid.wrappers import ImgObsWrapper
from torch.distributions.categorical import Categorical

from intrinsic import TAMPER_ACTION, make_intrinsic


# ---------------------------------------------------------------- environment

class EarlyGoalStart(gym.Wrapper):
    """Early-goal intervention: while `active`, each reset starts, with probability `frac`, inside the goal
    room with the door unlocked and open (agent on a random empty cell right of the wall, random direction,
    key left where the layout put it). Uses its own RNG so layouts match runs without the intervention."""

    def __init__(self, env, frac, seed):
        super().__init__(env)
        self.frac, self.active, self.rng = frac, frac > 0, np.random.default_rng(seed)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.active and self.rng.random() < self.frac:
            u = self.env.unwrapped
            door = next(c for c in u.grid.grid if isinstance(c, Door))
            door.is_locked, door.is_open = False, True
            wx = door.cur_pos[0]
            cells = [(x, y) for x in range(wx + 1, u.width - 1) for y in range(1, u.height - 1)
                     if u.grid.get(x, y) is None]
            u.agent_pos = np.array(cells[self.rng.integers(len(cells))])
            u.agent_dir = int(self.rng.integers(4))
            obs = u.gen_obs()
        return obs, info


def make_env(env_id, early_frac=0.0, seed=0):
    def thunk():
        # 7x7x3 egocentric partial view
        return ImgObsWrapper(EarlyGoalStart(gym.make(env_id), early_frac, seed))
    return thunk


def make_vec_env(env_id, n, early_frac=0.0, seed=0):
    # SyncVectorEnv so we can read each env's internal state (agent position, key, door).
    # SAME_STEP autoreset: the obs returned at a terminal step is already the next episode's.
    return gym.vector.SyncVectorEnv(
        [make_env(env_id, early_frac, seed * 1000 + i) for i in range(n)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


class EpisodeTracker:
    """Per-env episode statistics read from the unwrapped MiniGrid state.

    coverage = fraction of non-wall cells the agent stood on during the episode.
    got_key / opened_door / success = how far the agent got through the task chain.
    """

    def __init__(self, envs):
        self.envs = [e.unwrapped for e in envs.envs]
        n = len(self.envs)
        self.ret = np.zeros(n)
        self.int_ret = np.zeros(n)
        self.len = np.zeros(n, dtype=int)
        self.visited = [set() for _ in range(n)]
        self.floor = np.ones(n)
        self.doors = [[] for _ in range(n)]
        self.got_key = np.zeros(n, dtype=bool)
        self.opened = np.zeros(n, dtype=bool)
        self.pickups = np.zeros(n, dtype=int)
        self.prev_carry = [None] * n
        self.door_flips = np.zeros(n, dtype=int)
        self.prev_open = [()] * n
        self.tampers = np.zeros(n, dtype=int)
        self.key_pickups = np.zeros(n, dtype=int)       # pick-ups of a Key (None -> Key)
        self.locked_door_opens = np.zeros(n, dtype=int) # doors locked at episode start going closed -> open
        self.locked_doors = [[] for _ in range(n)]
        self.prev_locked_open = [()] * n
        for i in range(n):
            self._start(i)

    def _start(self, i):
        g = self.envs[i].grid
        cells = [g.get(x, y) for x in range(g.width) for y in range(g.height)]
        self.floor[i] = sum(not isinstance(c, Wall) for c in cells)
        self.doors[i] = [c for c in cells if isinstance(c, Door)]
        self.visited[i] = set()
        self.ret[i], self.len[i], self.int_ret[i] = 0.0, 0, 0.0
        self.got_key[i] = self.opened[i] = False
        self.pickups[i], self.prev_carry[i] = 0, self.envs[i].carrying
        self.door_flips[i], self.prev_open[i] = 0, tuple(d.is_open for d in self.doors[i])
        self.tampers[i] = 0
        self.key_pickups[i] = self.locked_door_opens[i] = 0
        self.locked_doors[i] = [d for d in self.doors[i] if d.is_locked]
        self.prev_locked_open[i] = tuple(d.is_open for d in self.locked_doors[i])

    def before_step(self):
        for i, e in enumerate(self.envs):
            self.visited[i].add(tuple(e.agent_pos))

    def after_step(self, reward, done, r_int=None, actions=None):
        """Returns a list of finished-episode dicts; resets trackers for envs that finished."""
        finished = []
        for i, e in enumerate(self.envs):
            self.ret[i] += reward[i]
            if r_int is not None:
                self.int_ret[i] += r_int[i]
            self.len[i] += 1
            if actions is not None:
                self.tampers[i] += int(actions[i]) == TAMPER_ACTION  # "done" presses (a tamper only if enabled)
            if not done[i]:
                self.got_key[i] |= isinstance(e.carrying, Key)
                self.opened[i] |= any(d.is_open for d in self.doors[i])
                self.pickups[i] += self.prev_carry[i] is None and e.carrying is not None
                self.key_pickups[i] += self.prev_carry[i] is None and isinstance(e.carrying, Key)
                self.prev_carry[i] = e.carrying
                now_lo = tuple(d.is_open for d in self.locked_doors[i])
                self.locked_door_opens[i] += sum((not a) and b for a, b in zip(self.prev_locked_open[i], now_lo))
                self.prev_locked_open[i] = now_lo
                now_open = tuple(d.is_open for d in self.doors[i])
                self.door_flips[i] += sum(a != b for a, b in zip(now_open, self.prev_open[i]))
                self.prev_open[i] = now_open
                continue
            success = reward[i] > 0
            finished.append(dict(
                ret=self.ret[i], int_ret=self.int_ret[i], len=self.len[i], success=float(success),
                coverage=len(self.visited[i]) / self.floor[i],
                got_key=float(self.got_key[i] or success),
                opened_door=float(self.opened[i] or success),
                pickups=float(self.pickups[i]),  # >1 means drop/pick-up repeats
                door_flips=float(self.door_flips[i]),  # >1 means close/open repeats
                tampers=float(self.tampers[i]),
                key_pickups=float(self.key_pickups[i]),
                locked_door_opens=float(self.locked_door_opens[i]),
                env=i,  # which env finished (bookkeeping only, not a metric)
            ))
            self._start(i)  # env i has already been reset (SAME_STEP autoreset)
        return finished


# -------------------------------------------------------------------- network

def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Encoder(nn.Module):
    """Small CNN over the 7x7x3 symbolic view (object, color, state ids)."""

    def __init__(self, obs_shape, out=64):
        super().__init__()
        h, w, c = obs_shape
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(c, 16, 2)), nn.ReLU(), nn.MaxPool2d(2),
            layer_init(nn.Conv2d(16, 32, 2)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 2)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.net(torch.zeros(1, c, h, w)).shape[1]
        self.fc = nn.Sequential(layer_init(nn.Linear(n, out)), nn.ReLU())

    def forward(self, obs):
        x = obs.float().permute(0, 3, 1, 2) / 10.0  # NHWC ints -> NCHW floats
        return self.fc(self.net(x))


class Agent(nn.Module):
    """Actor-critic. `state_dim` > 0 lets the policy also see the intrinsic module's internal
    state (Agent 3's needs), concatenated to the image features."""

    def __init__(self, obs_shape, n_actions, state_dim=0):
        super().__init__()
        self.encoder = Encoder(obs_shape)
        self.actor = layer_init(nn.Linear(64 + state_dim, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(64 + state_dim, 1), std=1.0)

    def features(self, obs, state):
        return torch.cat([self.encoder(obs), state], -1)

    def get_value(self, obs, state):
        return self.critic(self.features(obs, state)).squeeze(-1)

    def get_action_and_value(self, obs, state, action=None):
        z = self.features(obs, state)
        dist = Categorical(logits=self.actor(z))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(z).squeeze(-1)


# ----------------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(agent, cfg, env_id, episodes, seed, device, greedy, gen=None):
    """Each of the n parallel envs contributes exactly episodes/n consecutive episodes. (Before round 4 this
    kept the first `episodes` episodes to finish across envs, which over-weights short, successful episodes.)
    `gen`: optional torch.Generator for sampled actions, so evaluating mid-training leaves the training RNG alone."""
    n = min(16, episodes)
    quota = -(-episodes // n)
    envs = make_vec_env(env_id, n)
    obs, _ = envs.reset(seed=seed)
    tracker = EpisodeTracker(envs)
    intrinsic = make_intrinsic(cfg, envs.single_observation_space.shape, device)
    intrinsic.attach(envs)  # keeps the policy's internal state (needs) evolving during eval
    done_eps, per_env = [], np.zeros(n, dtype=int)
    while per_env.min() < quota:
        o = torch.tensor(obs, device=device)
        st = intrinsic.state()
        logits = agent.actor(agent.features(o, st))
        if greedy:
            action = logits.argmax(-1)
        elif gen is not None:
            action = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(-1)
        else:
            action = Categorical(logits=logits).sample()
        tracker.before_step()
        acts = action.cpu().numpy()
        obs, r, term, trunc, info = envs.step(acts)
        done = term | trunc
        nxt = obs.copy()
        if done.any():
            nxt[done] = np.stack(info["final_obs"][done])
        intrinsic.reward(torch.tensor(nxt), done, acts)
        for e in tracker.after_step(r, done, None, acts):
            if per_env[e["env"]] < quota:
                per_env[e["env"]] += 1
                done_eps.append(e)
    envs.close()
    return {k: float(np.mean([e[k] for e in done_eps])) for k in done_eps[0] if k != "env"}


# ----------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.toml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/seed0")
    p.add_argument("--total-timesteps", type=int, default=None, help="override, for smoke tests")
    args = p.parse_args()

    cfg = tomllib.loads(Path(args.config).read_text())
    run, env_cfg, pc, ic = cfg["run"], cfg["env"], cfg["ppo"], cfg["intrinsic"]
    total = args.total_timesteps or run["total_timesteps"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)  # one core per seed; seeds run in parallel
    device = torch.device("cpu")

    n_envs, n_steps = env_cfg["num_envs"], pc["num_steps"]
    batch = n_envs * n_steps
    mb = batch // pc["num_minibatches"]
    num_updates = total // batch

    early_frac, early_steps = env_cfg.get("early_goal_frac", 0.0), env_cfg.get("early_goal_steps", 0)
    envs = make_vec_env(env_cfg["id"], n_envs, early_frac, args.seed)
    obs_shape = envs.single_observation_space.shape
    max_steps_env = int(envs.envs[0].unwrapped.max_steps)
    intrinsic = make_intrinsic(cfg, obs_shape, device)
    intrinsic.attach(envs)
    agent = Agent(obs_shape, envs.single_action_space.n, intrinsic.state_dim).to(device)
    optimizer = torch.optim.Adam(
        list(agent.parameters()) + list(intrinsic.parameters()), lr=pc["learning_rate"], eps=1e-5
    )

    obs_buf = torch.zeros((n_steps, n_envs) + obs_shape, dtype=torch.uint8)
    act_buf = torch.zeros((n_steps, n_envs), dtype=torch.long)
    logp_buf = torch.zeros((n_steps, n_envs))
    rext_buf = torch.zeros((n_steps, n_envs))
    rint_buf = torch.zeros((n_steps, n_envs))
    done_buf = torch.zeros((n_steps, n_envs))
    val_buf = torch.zeros((n_steps, n_envs))
    state_buf = torch.zeros((n_steps, n_envs, intrinsic.state_dim))

    next_obs, _ = envs.reset(seed=args.seed * 1000)
    next_obs = torch.tensor(next_obs)
    next_done = torch.zeros(n_envs)
    tracker = EpisodeTracker(envs)
    recent = deque(maxlen=100)
    global_step, t0 = 0, time.time()
    first_solved_step = None  # exact env step at which rolling success over the last 100 episodes first >= 0.9
    first_goal_step = None    # exact env step of the first episode that reached the goal
    metrics_f = open(out / "metrics.jsonl", "w")
    # exact accumulators over episodes that finish in the last 10% of environment steps
    f10_start = 0.9 * num_updates * batch
    tamper_cost = float(ic.get("tamper_cost", 0.0)) if ic.get("tamper", False) else 0.0
    f10 = dict(episodes=0, tampers=0.0, solved=0, solved_len=0.0)
    # optional evaluation curve on every eval env (fixed layouts; own RNG so training randomness is unchanged)
    eval_every = env_cfg.get("eval_every_updates", 0)
    curve_eps = env_cfg.get("eval_curve_episodes", 16)
    eval_gen = torch.Generator().manual_seed(7_000_000 + args.seed)
    curve_f = open(out / "eval_curve.jsonl", "w") if eval_every else None

    for update in range(1, num_updates + 1):
        if pc["anneal_lr"]:
            optimizer.param_groups[0]["lr"] = (1.0 - (update - 1) / num_updates) * pc["learning_rate"]

        # ---- rollout
        for t in range(n_steps):
            global_step += n_envs
            if early_frac > 0 and global_step > early_steps and envs.envs[0].env.active:
                for e in envs.envs:
                    e.env.active = False  # intervention over: standard starts from here on
            obs_buf[t], done_buf[t] = next_obs, next_done
            state_buf[t] = st = intrinsic.state()
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs, st)
            act_buf[t], logp_buf[t], val_buf[t] = action, logp, value

            tracker.before_step()
            acts = action.numpy()
            o, r, term, trunc, info = envs.step(acts)
            done = term | trunc
            # round 5: optional per-press tamper cost, charged to the extrinsic reward PPO trains on (not scaled by
            # coef). Task metrics (success, ret) keep the raw environment reward.
            r_train = r - tamper_cost * (acts == TAMPER_ACTION) if tamper_cost else r
            rext_buf[t] = torch.tensor(r_train, dtype=torch.float32)
            # Intrinsic reward is computed on the obs that resulted from the action. On a terminal
            # step `o` is already the next episode's start, so use the true final obs from info.
            nxt = o.copy()
            if done.any():
                nxt[done] = np.stack(info["final_obs"][done])
            rint_buf[t] = intrinsic.reward(torch.tensor(nxt), done, acts)
            # int_ret = coef-weighted intrinsic return per episode, comparable to the task return
            finished = tracker.after_step(r, done, ic["coef"] * rint_buf[t].numpy(), acts)
            for e in finished:
                e.pop("env")
                if global_step > f10_start:
                    f10["episodes"] += 1; f10["tampers"] += e["tampers"]
                    if e["success"]:
                        f10["solved"] += 1; f10["solved_len"] += e["len"]
            if first_goal_step is None and any(e["success"] for e in finished):
                first_goal_step = global_step
            recent.extend(finished)
            if (finished and first_solved_step is None and len(recent) == recent.maxlen
                    and np.mean([e["success"] for e in recent]) >= 0.9):
                first_solved_step = global_step
            next_obs, next_done = torch.tensor(o), torch.tensor(done, dtype=torch.float32)

        # ---- GAE on combined reward (truncation treated as termination, as in CleanRL)
        rewards = rext_buf + ic["coef"] * rint_buf
        with torch.no_grad():
            next_value = agent.get_value(next_obs, intrinsic.state())
        adv = torch.zeros_like(rewards)
        lastgae = 0
        for t in reversed(range(n_steps)):
            nonterm = 1.0 - (next_done if t == n_steps - 1 else done_buf[t + 1])
            nv = next_value if t == n_steps - 1 else val_buf[t + 1]
            delta = rewards[t] + pc["gamma"] * nv * nonterm - val_buf[t]
            adv[t] = lastgae = delta + pc["gamma"] * pc["gae_lambda"] * nonterm * lastgae
        returns = adv + val_buf

        b_obs = obs_buf.reshape((-1,) + obs_shape)
        b_state = state_buf.reshape(batch, intrinsic.state_dim)
        b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
        b_adv, b_ret, b_val = adv.reshape(-1), returns.reshape(-1), val_buf.reshape(-1)

        # ---- PPO update
        idx = np.arange(batch)
        clipfracs = []
        for _ in range(pc["update_epochs"]):
            np.random.shuffle(idx)
            for s in range(0, batch, mb):
                m = idx[s:s + mb]
                _, newlogp, ent, newv = agent.get_action_and_value(b_obs[m], b_state[m], b_act[m])
                ratio = (newlogp - b_logp[m]).exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (newlogp - b_logp[m])).mean()
                    clipfracs.append(((ratio - 1).abs() > pc["clip_coef"]).float().mean().item())
                a = b_adv[m]
                if pc.get("norm_adv", True):
                    a = (a - a.mean()) / (a.std() + 1e-8)
                pg_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - pc["clip_coef"], 1 + pc["clip_coef"])).mean()
                v_loss = 0.5 * ((newv - b_ret[m]) ** 2).mean()
                loss = pg_loss - pc["ent_coef"] * ent.mean() + pc["vf_coef"] * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), pc["max_grad_norm"])
                optimizer.step()
        int_stats = intrinsic.update(b_obs)

        # ---- logging
        if eval_every and (update % eval_every == 0 or update == num_updates):
            point = dict(global_step=global_step)
            for env_id in env_cfg["eval_ids"]:
                name = env_id.removeprefix("MiniGrid-").removesuffix("-v0")
                point[name] = evaluate(agent, cfg, env_id, curve_eps, 3_000_000 + args.seed, device, False,
                                       gen=eval_gen)["success"]
            curve_f.write(json.dumps(point) + "\n"); curve_f.flush()
        if update % run["log_every_updates"] == 0 or update == num_updates:
            row = dict(
                global_step=global_step,
                sps=int(global_step / (time.time() - t0)),
                r_int_mean=float(rint_buf.mean()), r_int_std=float(rint_buf.std()),
                pg_loss=pg_loss.item(), v_loss=v_loss.item(), entropy=ent.mean().item(),
                approx_kl=approx_kl.item(), clipfrac=float(np.mean(clipfracs)),
                **int_stats,
            )
            if recent:
                row.update({f"train/{k}": float(np.mean([e[k] for e in recent])) for k in recent[0]})
                row["train/len_median"] = float(np.median([e["len"] for e in recent]))
            metrics_f.write(json.dumps(row) + "\n"); metrics_f.flush()
            print(f"[seed {args.seed}] step {global_step:>9} sps {row['sps']:>5} "
                  f"success {row.get('train/success', 0):.2f} ret {row.get('train/ret', 0):.3f} "
                  f"cov {row.get('train/coverage', 0):.2f} key {row.get('train/got_key', 0):.2f} "
                  f"door {row.get('train/opened_door', 0):.2f} tamp {row.get('train/tampers', 0):.2f} "
                  f"r_int {row['r_int_mean']:.4f}", flush=True)
    envs.close()
    metrics_f.close()
    if curve_f:
        curve_f.close()

    # ---- generalization eval on fresh layouts (eval seeds disjoint from training seeds)
    evals = {}
    for env_id in env_cfg["eval_ids"]:
        for greedy in (False, True):
            key = f"{env_id.removeprefix('MiniGrid-').removesuffix('-v0')}/{'greedy' if greedy else 'sample'}"
            evals[key] = evaluate(agent, cfg, env_id, env_cfg["eval_episodes"], 10**6 + args.seed, device, greedy)
            print(f"[seed {args.seed}] eval {key}: success {evals[key]['success']:.2f} "
                  f"ret {evals[key]['ret']:.3f} cov {evals[key]['coverage']:.2f}", flush=True)

    rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    last = rows[-max(1, len(rows) // 10):]  # final 10% of training
    summary = dict(
        seed=args.seed, total_timesteps=global_step, intrinsic=ic, env_id=env_cfg["id"],
        max_steps=max_steps_env,
        first_solved_step=first_solved_step, first_goal_step=first_goal_step,
        # "keeps exploring?": episodic coverage once the task is solved (None if never solved)
        coverage_after_solved=(float(np.mean([r["train/coverage"] for r in rows
                                              if r["global_step"] >= first_solved_step]))
                               if first_solved_step else None),
        final_train={k: float(np.mean([r[k] for r in last if k in r]))
                     for k in last[-1] if k.startswith("train/") or k.startswith("r_int")},
        final10_exact=dict(start_step=f10_start, episodes=f10["episodes"],
                           tampers_per_episode=f10["tampers"] / max(1, f10["episodes"]),
                           solved_episodes=f10["solved"],
                           solved_len_mean=(f10["solved_len"] / f10["solved"]) if f10["solved"] else None),
        eval=evals,
    )
    if eval_every:
        pts = [json.loads(l) for l in open(out / "eval_curve.jsonl")]
        summary["eval_curve_area"] = {k: float(np.mean([p[k] for p in pts])) for k in pts[0] if k != "global_step"}
        summary["eval_curve_points"] = len(pts)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(agent.state_dict(), out / "agent.pt")


if __name__ == "__main__":
    main()
