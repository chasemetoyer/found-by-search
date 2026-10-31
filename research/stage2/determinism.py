"""User-approved pinned determinism test (protocol: stage2/determinism-protocol.md).

Replicates of Stage 1's exact seed-910 spec run with all three kernel libraries
capped at AVX2. Modal cannot be asked for a CPU type, so replicates are sent in
waves until each required CPU class has enough completed replicates.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from stage2.common import ISA_PINS, cpu_class, parse_cpuinfo
from stage2.specs import MAX_PINNED_REPLICATES, PINNED_PURPOSE, pinned_determinism_spec

WAVE_SIZE = 32
REQUIRED_CLASSES = ("zen3", "zen4", "intel")
MIN_PER_CLASS = 2
DEADLINE_SECONDS = 1200
FINGERPRINT_KEYS = ("metrics_without_sps_sha256", "summary_sha256", "checkpoint_tensors_sha256")
# Stage 1 unpinned replicates of the same spec, for descriptive comparison only.
STAGE1_UNPINNED = {
    "r0 (Zen 3, unpinned)": {"metrics_without_sps_sha256": "ab0113705d98792b6e8eb16d67029f445db99483e6498b010c2ea82cbc26ca2c",
                             "summary_sha256": "b5f9ab50407cf887dc90a216e3a9936bd918cb2d1d12555f3341a615e8105a38",
                             "checkpoint_tensors_sha256": "b694f76365b8b0923e54e84175226e8530d6f5a3d1db95be6d678015654abad8"},
    "r1 (Zen 4, unpinned)": {"metrics_without_sps_sha256": "d3bb9b81486539f16a9408df043ef320ad9b6ccda9cafb561f982c8920950a5c",
                             "summary_sha256": "8d8644ad6893ffbf640f845d6fd2a5cebfffb6b65b071b8319f2394f5167788b",
                             "checkpoint_tensors_sha256": "409321ab8559c41689d1bb2ced3c1ef57ceb3c4a03319c4d53dc8cc38d33b2a3"},
}


def key_replicate(key: str) -> int:
    return int(key.rsplit("-r", 1)[1])


class PinnedDeterminismPlan:
    name = "pinned_determinism"

    def __init__(self):
        self.records: dict[str, dict] = {}

    def owns(self, key: str) -> bool:
        return key.startswith(PINNED_PURPOSE + "-")

    def spec_for_key(self, key: str) -> dict:
        return pinned_determinism_spec(key_replicate(key))

    def deadline_seconds(self, spec: dict) -> int:
        return DEADLINE_SECONDS

    def record(self, key: str, result: dict, directory: Path) -> None:
        environment = json.loads((Path(directory) / "environment.json").read_text())
        info = parse_cpuinfo(environment.get("cpuinfo"))
        probe_path = Path(directory) / "isa_probe.json"
        self.records[key] = {"replicate": key_replicate(key), "attempt": result["attempt"], "cpu": info,
                             "cpu_class": cpu_class(info), "fingerprints": result["fingerprints"],
                             "isa_probe": json.loads(probe_path.read_text()) if probe_path.exists() else None}

    def class_counts(self) -> Counter:
        return Counter(r["cpu_class"] for r in self.records.values())

    def rule_satisfied(self) -> bool:
        counts = self.class_counts()
        return all(counts[c] >= MIN_PER_CLASS for c in REQUIRED_CLASSES)

    def _submitted(self, ledger) -> int:
        replicates = sorted(key_replicate(k) for k in ledger.data["calls"] if self.owns(k))
        if replicates != list(range(len(replicates))):
            raise ValueError("Pinned replicates must be submitted contiguously from r0")
        return len(replicates)

    @staticmethod
    def _wave_end(submitted: int) -> int:
        """End of the wave containing the last submitted replicate (waves are fixed blocks of 32)."""
        return max(WAVE_SIZE, -(-submitted // WAVE_SIZE) * WAVE_SIZE)

    def ready_specs(self, ledger) -> list[dict]:
        submitted = self._submitted(ledger)
        wave_end = self._wave_end(submitted)
        if submitted < wave_end:  # finish the current wave, including after a controller restart
            return [pinned_determinism_spec(r) for r in range(submitted, wave_end)]
        if len(self.records) < submitted or self.rule_satisfied() or submitted >= MAX_PINNED_REPLICATES:
            return []  # wave barrier, stop rule met, or replicate cap reached
        return [pinned_determinism_spec(r) for r in range(submitted, min(submitted + WAVE_SIZE, MAX_PINNED_REPLICATES))]

    def done(self, ledger) -> bool:
        submitted = self._submitted(ledger)
        return (submitted > 0 and submitted == self._wave_end(submitted) and len(self.records) == submitted
                and (self.rule_satisfied() or submitted >= MAX_PINNED_REPLICATES))

    def summary(self) -> dict:
        rows = sorted(self.records.values(), key=lambda r: r["replicate"])
        triple = lambda r: tuple(r["fingerprints"].get(k) for k in FINGERPRINT_KEYS)
        distinct = Counter(triple(r) for r in rows)
        by_class = {}
        for c in sorted({r["cpu_class"] for r in rows}):
            members = [r for r in rows if r["cpu_class"] == c]
            by_class[c] = {"replicates": len(members), "distinct_fingerprints": len({triple(r) for r in members})}

        def pin_effects(r):
            probe = r["isa_probe"] or {}
            onednn = " ".join(probe.get("onednn_verbose", {}).get("lines", []))
            mkl = " ".join(probe.get("mkl_verbose", {}).get("lines", []))
            return {"environment_set": probe.get("environment", {}).get("ATEN_CPU_CAPABILITY") == ISA_PINS["ATEN_CPU_CAPABILITY"]
                                       and probe.get("environment", {}).get("MKL_CBWR") == ISA_PINS["MKL_CBWR"]
                                       and probe.get("environment", {}).get("ONEDNN_MAX_CPU_ISA") == ISA_PINS["ONEDNN_MAX_CPU_ISA"],
                    "aten_capability": probe.get("aten_cpu_capability"),
                    "onednn_reports_avx512": "AVX-512" in onednn or "avx512" in onednn.lower(),
                    "onednn_lines_seen": bool(onednn), "mkl_cnr_avx2": "CNR:AVX2" in mkl, "mkl_lines_seen": bool(mkl)}

        effects = {r["replicate"]: pin_effects(r) for r in rows}
        stage1 = {label: any(triple(r) == tuple(fp[k] for k in FINGERPRINT_KEYS) for r in rows)
                  for label, fp in STAGE1_UNPINNED.items()}
        return {
            "protocol": "stage2/determinism-protocol.md", "pins": ISA_PINS, "completed_replicates": len(rows),
            "class_counts": dict(self.class_counts()), "stop_rule_satisfied": self.rule_satisfied(),
            "replicate_cap": MAX_PINNED_REPLICATES, "distinct_fingerprint_triples": len(distinct),
            "bit_identical_all_replicates": len(rows) > 0 and len(distinct) == 1,
            "by_class": by_class,
            "pins_effective_all": all(e["environment_set"] and e["aten_capability"] == "AVX2" and e["onednn_lines_seen"]
                                      and not e["onednn_reports_avx512"] and e["mkl_cnr_avx2"] for e in effects.values()),
            "pin_effects_by_replicate": effects,
            "matches_stage1_unpinned": stage1,
            "replicates": [{"replicate": r["replicate"], "attempt": r["attempt"], "cpu_class": r["cpu_class"],
                            "vendor_id": r["cpu"]["vendor_id"], "cpu_family": r["cpu"]["cpu_family"],
                            "model": r["cpu"]["model"], "avx512f": r["cpu"]["avx512f"], "fingerprints": r["fingerprints"]}
                           for r in rows],
        }
