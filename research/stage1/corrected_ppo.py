"""Corrected event plumbing around the unchanged archived PPO implementation.

The only AST edit moves main's intrinsic.attach(envs) statement to immediately
after the initial seeded reset. All remaining statements, including every PPO
update and CNN definition, are preserved. Runtime factory replacements capture
true pre-autoreset events, select the corrected detector, and attach descriptive
episode counters. No archive file is rewritten and importing starts no run.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

from generic_reward import (
    CorrectedLegacyEventDetector, GenericEventCapture, GenericHomeostatic, _capture,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("final-round5", "a1b-run-commit-1533734")


def corrected_tree(source_text):
    """Move precisely one attach statement; refuse unfamiliar source layouts."""
    tree = ast.parse(source_text)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")

    def is_attach(node):
        return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "intrinsic.attach")

    attach_indices = [i for i, n in enumerate(main.body) if is_attach(n)]
    if len(attach_indices) != 1:
        raise ValueError("Expected one top-level intrinsic.attach in archived main")
    attach = main.body.pop(attach_indices[0])
    reset_indices = [i for i, node in enumerate(main.body)
                     if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                     and ast.unparse(node.value.func) == "envs.reset"]
    if len(reset_indices) != 1:
        raise ValueError("Expected one top-level initial envs.reset in archived main")
    main.body.insert(reset_indices[0] + 1, attach)
    return ast.fix_missing_locations(tree)


def _load_intrinsic(path):
    spec = importlib.util.spec_from_file_location("stage1_corrected_archived_intrinsic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Original classes resolve this global when attach is called. Their state
    # dimensions and every reward-arithmetic operation remain unchanged.
    module.EventDetector = CorrectedLegacyEventDetector
    return module


def load_ppo(*, source="final-round5"):
    """Return the corrected learner for either preserved archive snapshot."""
    if source not in SOURCES:
        raise ValueError(f"Unknown archived source {source!r}")
    directory = ROOT / "handoff-cross-platform/code" / source
    intrinsic = _load_intrinsic(directory / "intrinsic.py")
    text = (directory / "ppo.py").read_text()
    module = types.ModuleType("stage1_corrected_archived_ppo")
    module.__file__ = str(directory / "ppo.py")
    saved = sys.modules.get("intrinsic")
    try:
        sys.modules["intrinsic"] = intrinsic
        exec(compile(corrected_tree(text), str(directory / "ppo.py"), "exec"), module.__dict__)
    finally:
        if saved is None:
            sys.modules.pop("intrinsic", None)
        else:
            sys.modules["intrinsic"] = saved

    original_intrinsic_factory = module.make_intrinsic
    original_env_factory = module.make_env
    original_tracker = module.EpisodeTracker

    def make_intrinsic(cfg, obs_shape, device):
        if cfg["intrinsic"]["kind"] == "generic_homeostatic":
            return GenericHomeostatic(cfg, obs_shape, device)
        return original_intrinsic_factory(cfg, obs_shape, device)

    def make_env(*args, **kwargs):
        original_thunk = original_env_factory(*args, **kwargs)
        return lambda: GenericEventCapture(original_thunk())

    class CorrectedEpisodeTracker(original_tracker):
        """Descriptive event counts, captured before reset; raw success unchanged."""

        def __init__(self, envs):
            super().__init__(envs)
            self.captures = [_capture(e) for e in envs.envs]

        def after_step(self, reward, done, r_int=None, actions=None):
            episodes = super().after_step(reward, done, r_int, actions)
            for episode in episodes:
                c = self.captures[episode["env"]].last_episode_counts
                episode.update(
                    pickups=float(c["eligible_pickups"]), raw_pickups=float(c["raw_pickups"]),
                    target_pickups_excluded=float(c["target_pickups_excluded"]),
                    drops=float(c["drops"]), door_flips=float(c["door_changes"]),
                    box_toggles=float(c["box_changes"]), generic_toggles=float(c["toggles"]),
                    key_pickups=float(c["key_pickups"]), locked_door_opens=float(c["locked_door_opens"]),
                )
            return episodes

    module.make_intrinsic = make_intrinsic
    module.make_env = make_env
    module.EpisodeTracker = CorrectedEpisodeTracker
    module.archived_intrinsic = intrinsic
    module.stage1_provenance = {
        "source": source,
        "archived_ppo_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "ast_change": "move intrinsic.attach(envs) immediately after initial seeded envs.reset",
        "factory_changes": ["capture physical events before SAME_STEP reset",
                            "corrected three-event or generic four-event detector",
                            "exclude successful target pickup from paid pickup event",
                            "record true episode physical-event diagnostics"],
    }
    return module


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", choices=SOURCES, default="final-round5")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    load_ppo(source=args.source).main()


if __name__ == "__main__":
    main()
