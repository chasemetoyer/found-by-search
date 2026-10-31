"""Stage 2 run specifications and the worker-side validator.

Stage 1 workers checked each spec against a static manifest. Stage 2 search
candidates are generated during the search, so the worker validates structure
instead: every field must match its purpose's fixed template exactly, apart
from the explicitly variable fields.
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache

from stage2.common import ISA_PINS, ROOT

STAGE1_DETERMINISM_KEY = "determinism-DoorKey-8x8-episodic_count-131072-seed910-r0"
PINNED_PURPOSE = "pinned_determinism"
MAX_PINNED_REPLICATES = 96


@lru_cache(maxsize=1)
def stage1_manifest() -> dict:
    return json.loads((ROOT / "stage1/manifest.json").read_text())


def stage1_determinism_spec() -> dict:
    matches = [s for s in stage1_manifest()["runs"] if s["id"] == STAGE1_DETERMINISM_KEY]
    if len(matches) != 1:
        raise ValueError("Stage 1 seed-910 determinism spec not found exactly once")
    return copy.deepcopy(matches[0])


def pinned_determinism_spec(replicate: int) -> dict:
    """Stage 1's exact seed-910 spec; only identity fields and the ISA pins are added."""
    if not 0 <= replicate < MAX_PINNED_REPLICATES:
        raise ValueError("Pinned replicate outside the registered range")
    spec = stage1_determinism_spec()
    spec.update(id=f"{PINNED_PURPOSE}-DoorKey-8x8-episodic_count-131072-seed910-r{replicate}",
                purpose=PINNED_PURPOSE, replicate=replicate, isa_pins=dict(ISA_PINS))
    return spec


def committed_mode() -> str:
    return json.loads((ROOT / "stage2/mode.json").read_text())["mode"]


def validate(spec: dict) -> None:
    """Raise unless the spec is exactly an allowed entry for the committed mode."""
    purpose = spec.get("purpose")
    mode = committed_mode()
    if purpose == PINNED_PURPOSE and mode == "pinned_determinism":
        replicate = spec.get("replicate")
        if not isinstance(replicate, int):
            raise ValueError("Pinned replicate must be an integer")
        if spec != pinned_determinism_spec(replicate):
            raise ValueError("Pinned determinism spec differs from Stage 1 seed 910 plus the approved pins")
        return
    if mode == "smoke":
        from stage2.search import SMOKE, validate_candidate
        if purpose in SMOKE.purposes().values():
            validate_candidate(spec, SMOKE)
            return
    if mode == "search":
        from stage2.registration import configured_search
        from stage2.search import validate_candidate
        cfg = configured_search()
        if purpose in cfg.purposes().values():
            validate_candidate(spec, cfg)
            return
    raise ValueError(f"Purpose {purpose!r} not allowed in committed mode {mode!r}")
