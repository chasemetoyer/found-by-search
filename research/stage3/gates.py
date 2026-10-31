"""Fail-closed Stage3 scientific launch gates; no launch or tracking writes.

`require_science_gate(root=ROOT, current_ledger=None)` returns a compact verified
receipt. With no injected ledger it reads the Stage3 Modal volume budget ledger
only. All other evidence is SHA256-bound files beneath the committed source.
See tests/test_stage3_gates.py's complete fixture for the explicit JSON schema.

Composite evidence is deliberate: the original worker smoke has34 real tiny
inputs. A later Linux controller validation may replay those34 saved results
with0submissions, provided all worker implementation hashes remain identical.
The user's lid witness applies to the ORIGINAL worker-smoke completion, not to
the later controller-validation receipt. An explicitly recorded lid-only waiver
is accepted solely as WAIVED_UNVERIFIED, never as a verified closed-lid test. These are never described as a second
successful training smoke or silently conflated.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

from stage3.common import ROOT, PROJECT_ID, VOLUME_NAME

RECEIPT = "stage3/prelaunch-receipt.json"
APPROVAL = "stage3/launch-approval.json"
RECEIPT_SCHEMA = "stage3-prelaunch-receipt-v1"
APPROVAL_SCHEMA = "stage3-explicit-launch-approval-v1"
PROOF_NAMES = frozenset({"source_inventory", "worker_inventory", "worker_smoke",
                         "controller_validation", "mac_decode", "linux_decode",
                         "visualize", "lid_closed", "cost_projection", "summary", "amendment_commit"})
WORKER_FILES = (
    "stage3/modal_app.py", "stage3/run_training.py", "stage3/instrumentation.py",
    "stage3/specs.py", "stage3/common.py",
)
IMPLEMENTATION_FILES = frozenset(WORKER_FILES + (
    "stage3/controller.py", "stage3/budget.py", "stage3/observer.py",
    "stage3/tracking.py", "stage3/tracking_server.py", "stage3/registration.py",
    "stage3/gates.py", "scripts/run_stage3.sh",
))



class LaunchGateError(ValueError):
    """A required pre-launch condition has not been demonstrated."""


def _require(condition, message):
    if not condition:
        raise LaunchGateError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _file(root, name):
    path = Path(name)
    _require(not path.is_absolute() and ".." not in path.parts, "Proof path escapes source root")
    target = (root / path).resolve()
    _require(target.is_relative_to(root.resolve()), "Proof symlink escapes source root")
    _require(target.is_file(), "Required pre-launch proof file is absent")
    return target


def _ref(root, value, *, parse=True):
    _require(isinstance(value, dict) and set(value) == {"path", "sha256"}, "Invalid proof reference")
    _require(isinstance(value["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
             "Proof reference has no valid SHA256")
    raw = _file(root, value["path"]).read_bytes()
    _require(_sha(raw) == value["sha256"], "Pre-launch proof hash mismatch")
    return json.loads(raw) if parse else raw


def _timestamp(value):
    _require(isinstance(value, str), "Proof timestamp must be explicit")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LaunchGateError("Invalid proof timestamp") from error
    _require(timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
             "Proof timestamp requires an explicit UTC offset/timezone")
    return timestamp


def _decimal(value, label):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise LaunchGateError("Invalid " + label) from error
    _require(result.is_finite(), "Nonfinite " + label)
    return result


def _approval_rows(root, ref):
    raw = _ref(root, ref, parse=False).decode()
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _decode(proof):
    return proof.get("record", proof)


def _read_ledger():
    import modal
    from stage2.transport import retry_io
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False, version=2)
    return json.loads(retry_io(lambda: b"".join(volume.read_file("/budget-ledger.json"))))

POST_SUMMARY_INFRA_FILES = frozenset({
    "stage3/tracking_server.py", "stage3/gates.py",
    "tests/test_stage3_tracking.py", "tests/test_stage3_gates.py",
})


def _public_amendment_time(proof, inventory_sha):
    commit = proof.get("commit", "")
    _require(proof.get("status") == "PASS" and isinstance(commit, str)
             and re.fullmatch(r"[0-9a-f]{40}", commit)
             and proof.get("github_ref_commit") == commit
             and proof.get("repository_visibility") == "PUBLIC",
             "Budget/descriptive amendments have no verified public commit")
    _require(proof.get("implementation_inventory_sha256") == inventory_sha,
             "Public amendment receipt does not bind this tested implementation")
    return _timestamp(proof.get("verified_utc"))



def require_science_gate(*, root=ROOT, current_ledger=None):
    """Verify every user gate before a scientific reservation or worker spawn.

    Required approval log row: seq, quote, recorded_utc, approval_scope exactly
    'stage3-scientific-launch', scientific_launch_authorized true. The row must
    be recorded AFTER the hashed six-item pre-launch summary was delivered.
    This cannot infer approval from a conditionally approved setup, an empty
    field, the passage of time, or a successful smoke.
    """
    root = Path(root)
    receipt_raw = _file(root, RECEIPT).read_bytes()
    approval = json.loads(_file(root, APPROVAL).read_text())
    receipt = json.loads(receipt_raw)
    _require(approval.get("schema") == APPROVAL_SCHEMA and receipt.get("schema") == RECEIPT_SCHEMA,
             "Unknown launch receipt schema")
    _require(approval.get("scientific_launch_authorized") is True,
             "No explicit full scientific-launch approval")
    _require(approval.get("prelaunch_receipt_sha256") == _sha(receipt_raw),
             "Go-ahead does not bind this pre-launch receipt")
    refs = receipt.get("proofs", {})
    _require(set(refs) == PROOF_NAMES, "Required pre-launch proof set is incomplete")
    proofs = {name: _ref(root, ref, parse=name != "summary") for name, ref in refs.items()}
    summary_bytes = proofs["summary"]
    _require(bool(summary_bytes.strip()), "Pre-launch summary is empty")
    delivered = _timestamp(receipt.get("summary_delivered_utc"))
    _require(receipt.get("summary_items") == ["hash_verification", "cost_and_cap", "mac_linux_decode",
                                             "lid_closed_smoke", "amendment_commit", "native_visualize"],
             "Delivered summary does not cover the six required checks")
    log_rows = _approval_rows(root, approval.get("approval_log"))
    matches = [row for row in log_rows if row.get("seq") == approval.get("approval_seq")]
    _require(len(matches) == 1, "Full launch go-ahead has no unique recorded user instruction")
    goahead = matches[0]
    _require(goahead.get("scientific_launch_authorized") is True
             and goahead.get("approval_scope") == "stage3-scientific-launch"
             and isinstance(goahead.get("quote"), str) and bool(goahead["quote"].strip()),
             "Setup approval is not a full launch go-ahead")
    _require(_timestamp(goahead.get("recorded_utc")) > delivered,
             "Full launch go-ahead predates the delivered six-item summary")

    # Source proof binds the already-exercised implementation inventory. Frozen
    # inherited files remain guarded by the original registration as well.
    from stage2.registration import authorize, mode_record
    authorize(mode_record(root), root=root)
    inventory = proofs["source_inventory"]
    actual_inventory = _file(root, "preregistration/stage3-runtime-manifest.json").read_bytes()
    _require(refs["source_inventory"]["sha256"] == _sha(actual_inventory),
             "Tested source inventory is not the current inventory")
    for name, expected in inventory.get("sha256", {}).items():
        _require(_sha(_file(root, name).read_bytes()) == expected,
                 "Current implementation differs from its tested source inventory")
    _require(IMPLEMENTATION_FILES <= set(inventory.get("sha256", {})),
             "Source inventory omits required worker/controller/tracking implementation")
    amendment = proofs["amendment_commit"]
    verified_time = _public_amendment_time(amendment, refs["source_inventory"]["sha256"])
    amendment_files = (("preregistration/stage3-budget15-amendment-20261029.json", "budget_amendment_sha256"),
                       ("preregistration/stage3-descriptive-amendment-20261029.md", "descriptive_amendment_sha256"))
    for name, key in amendment_files:
        _require(amendment.get(key) == _sha(_file(root, name).read_bytes()),
                 "Published amendment hash differs from its approved file")
    post_summary_changes = []
    if verified_time > delivered:
        # A necessary tracking shutdown repair may be verified after the
        # already-delivered summary and explicit go-ahead. Preserve that earlier
        # public receipt and source inventory; never pretend the new commit was
        # already reviewed, and never broaden this path to science or dispatch.
        _require(amendment.get("scope") == "post-summary-infrastructure-only",
                 "Post-summary amendment has no restricted infrastructure scope")
        prior_ref = amendment.get("prior_amendment_commit")
        prior_inventory_ref = amendment.get("prior_source_inventory")
        prior = _ref(root, prior_ref)
        prior_inventory = _ref(root, prior_inventory_ref)
        prior_time = _public_amendment_time(prior, prior_inventory_ref["sha256"])
        _require(prior_time <= delivered,
                 "Prior public amendment was not verified before the delivered summary")
        _require(IMPLEMENTATION_FILES <= set(prior_inventory.get("sha256", {})),
                 "Prior tested source inventory is incomplete")
        for name, key in amendment_files:
            expected = _sha(_file(root, name).read_bytes())
            _require(prior.get(key) == amendment.get(key) == expected
                     and prior_inventory.get("sha256", {}).get(name) == inventory["sha256"].get(name) == expected,
                     "Post-summary repair changed the approved budget/descriptive amendment")
        old_hashes, new_hashes = prior_inventory["sha256"], inventory["sha256"]
        changed = {name for name in set(old_hashes) | set(new_hashes)
                   if old_hashes.get(name) != new_hashes.get(name)}
        _require(changed <= POST_SUMMARY_INFRA_FILES,
                 "Post-summary repair changed scientific, worker or dispatch implementation outside the infrastructure allowlist")
        post_summary_changes = sorted(changed)
    original_inventory = proofs["worker_inventory"]
    for name in WORKER_FILES:
        _require(original_inventory.get("sha256", {}).get(name) == inventory["sha256"][name],
                 "Worker implementation changed after the real tiny-budget smoke")

    # Verify the literal frozen exports in this source root, then generate every
    # registered smoke/science input with the unchanged configuration functions.
    from stage3.specs import winners, specs, decode_record, EXPECTED_HASHES
    _require(root.resolve() == ROOT.resolve() or current_ledger is not None,
             "Alternate-root fixture requires an injected ledger")
    # Production uses ROOT. Unit fixtures monkeypatch these pure specification
    # functions; no worker, optimizer or tracking client is imported here.
    chosen = winners()
    _require({name: w["candidate"] for name, w in chosen.items()} == {"E": 113, "R": 57},
             "Frozen winners were substituted")
    for name, expected in EXPECTED_HASHES.items():
        _require(_sha(_file(root, f"stage3/frozen/frozen-winner-{name}.json").read_bytes()) == expected,
                 "Frozen winner SHA256 mismatch")
    expected_smoke = {spec["id"]: _canonical(spec) for spec in specs("smoke")}
    _require(len(expected_smoke) == 34, "Tiny smoke plan does not have exactly34inputs")
    _require(all(spec["seed"] >= 5000 and spec["total_timesteps"] == 8192 for spec in specs("smoke")),
             "Smoke used a scientific seed or full training budget")
    worker = proofs["worker_smoke"]
    controller = proofs["controller_validation"]
    for proof in (worker, controller):
        _require(proof.get("status") == "PASS" and proof.get("profile") == "smoke"
                 and proof.get("completed") == proof.get("expected") == 34,
                 "Actual smoke/controller validation is incomplete")
        _require(isinstance(proof.get("run_id"), str) and bool(proof["run_id"]),
                 "Smoke proof has no tracked run identity")
        _require(str(proof.get("platform", "")).startswith("Linux"),
                 "Smoke/controller validation did not execute on Linux")
    _require(controller.get("implementation_inventory_sha256") == refs["source_inventory"]["sha256"],
             "Controller validation used a different implementation inventory")
    _require(controller.get("worker_smoke_run_id") == worker["run_id"],
             "Controller validation does not identify the original worker smoke")
    _require(controller.get("submitted") == 0 and controller.get("replayed") == 34,
             "Post-fix validation must replay34saved inputs without resubmission")
    _require(worker.get("implementation_inventory_sha256") == refs["worker_inventory"]["sha256"],
             "Original worker smoke is not bound to its source inventory")
    input_rows = controller.get("completed_inputs", [])
    _require(len(input_rows) == 34 and len({row.get("id") for row in input_rows}) == 34,
             "Smoke input receipts are incomplete or duplicated")
    for row in input_rows:
        key = row.get("id")
        _require(key in expected_smoke and row.get("spec_sha256") == expected_smoke[key],
                 "Smoke input identity/hash differs from the exact tiny plan")
        marker = _ref(root, row.get("marker"))
        _require(marker.get("id") == key and marker.get("spec_sha256") == expected_smoke[key]
                 and marker.get("outcome") == "success"
                 and marker.get("source_commit") == worker.get("source_commit")
                 and bool(marker.get("source_commit")),
                 "Smoke marker is not a validated successful original-worker result")
        prefix = f"runs/{key}/attempts/{marker.get('attempt')}/"
        needed = {prefix + name for name in ("spec.json", "metrics.jsonl", "summary.json", "environment.json",
                                             "descriptive-episodes.jsonl", "descriptive-windows.jsonl",
                                             "descriptive-status.json")}
        _require(needed <= set(marker.get("files", {})), "Smoke marker omits required result/diagnostic evidence")
    _require({row["id"] for row in input_rows} == set(expected_smoke),
             "Smoke receipts do not cover the exact plan")

    expected_decode = decode_record()
    mac, linux = proofs["mac_decode"], proofs["linux_decode"]
    _require(str(mac.get("platform", "")).startswith(("Darwin", "macOS")),
             "Mac configuration proof has no Mac platform provenance")
    _require(str(linux.get("platform", "")).startswith("Linux"),
             "Linux configuration proof has no Linux platform provenance")
    for proof in (mac, linux):
        record = _decode(proof)
        _require(record.get("winner_sha256") == EXPECTED_HASHES,
                 "Decode proof used different frozen winners")
        _require(record.get("configuration_sha256") == expected_decode["configuration_sha256"],
                 "Mac/Linux scientific configurations differ from the frozen plan")
        _require(record.get("combined_sha256") == _canonical(record["configuration_sha256"]),
                 "Decode proof canonical hash is inconsistent")

    # UI proof is native OpenResearch visibility plus actual browser verification,
    # not only successful SDK queuing or a standalone HTML substitute.
    from stage3.tracking import LABEL, panel_html, snapshot_payload
    required_panels = {f"{LABEL}/{name}" for name in panel_html(snapshot_payload([], {}))}
    visual = proofs["visualize"]
    _require(len(required_panels) == 18 and set(visual.get("panel_names", [])) == required_panels,
             "Native Visualize proof omits one or more required panels")
    _require(visual.get("project_id") == PROJECT_ID and visual.get("run_id") == controller["run_id"]
             and visual.get("completed_runs") == 34,
             "Native Visualize does not show the current34-input smoke validation")
    _require(all(visual.get(name) is True for name in ("native_run_listed", "native_visualize_button_verified",
                                                     "browser_verified", "remote_delivery_verified")),
             "Native Visualize/browser delivery is not verified")
    visual_time = _timestamp(visual.get("browser_verified_utc"))
    _require(visual_time <= delivered, "Pre-launch summary preceded native Visualize verification")

    lid = proofs["lid_closed"]
    _require(isinstance(lid.get("user_quote"), str) and bool(lid["user_quote"].strip())
             and lid.get("worker_smoke_run_id") == worker["run_id"],
             "Lid-closed proof has no explicit user record")
    witness_rows = [row for row in log_rows if row.get("seq") == lid.get("approval_seq")]
    _require(len(witness_rows) == 1 and witness_rows[0].get("quote") == lid["user_quote"],
             "Lid witness/waiver does not match the recorded user instruction")
    _require(_timestamp(witness_rows[0].get("recorded_utc")) <= delivered,
             "Pre-launch summary preceded the lid witness/waiver")
    _require(approval.get("approval_seq") != lid.get("approval_seq"),
             "A lid-only waiver/witness is not the separate post-summary launch go-ahead")
    if lid.get("status") == "WAIVED_UNVERIFIED":
        _require(lid.get("waiver_scope") == "lid_closed_smoke_witness_only"
                 and lid.get("actual_lid_closed_verified") is False
                 and lid.get("explicit_user_confirmation") is False,
                 "A waived lid test must remain explicitly unverified")
        lid_status = "WAIVED_UNVERIFIED"
    else:
        _require(lid.get("explicit_user_confirmation") is True,
                 "Lid-closed completion has no explicit user witness")
        closed, opened = _timestamp(lid.get("closed_at")), _timestamp(lid.get("opened_at"))
        finished = _timestamp(worker.get("finished_utc"))
        _require(closed < opened and closed <= finished <= opened,
                 "Original worker-smoke completion is outside the witnessed lid-closed interval")
        _require(opened <= delivered, "Pre-launch summary preceded the lid-open witness")
        lid_status = "VERIFIED"

    cost = proofs["cost_projection"]
    _require(_decimal(cost.get("hard_cap_usd"), "cap") == Decimal("15"), "Stage3 hard cap is not15USD")
    peak = _decimal(cost.get("projected_guard_peak_usd"), "guard peak")
    _require(Decimal("0") <= peak <= Decimal("13"), "Projected guard peak exceeds the13USD review threshold")
    _require((cost.get("final_evaluation_runs"), cost.get("held_out_runs"), cost.get("tamper_runs"),
              cost.get("max_in_flight")) == (300, 100, 40, 32), "Cost projection omits the full registered matrix")
    _require(all(cost.get(name) is True for name in ("includes_retries", "includes_controller_container",
                                                   "includes_tracking_container", "includes_prelaunch_spend",
                                                   "includes_in_flight_reservations")),
             "Cost projection omits retries, containers or prior spend")
    from stage1.budget import estimate_cost, TERMINATION_ALLOWANCE_SECONDS
    expected_in_flight = 32 * estimate_cost(3600 + TERMINATION_ALLOWANCE_SECONDS)
    _require(_decimal(cost.get("in_flight_reservation_usd"), "in-flight reserve") >= expected_in_flight,
             "Cost projection omits the full32in-flight reservations")
    _require(cost.get("scientific_input_plan_sha256") == _canonical(specs("science")),
             "Cost projection does not bind the complete registered scientific plan")
    ledger = _read_ledger() if current_ledger is None else current_ledger
    _require(ledger.get("stage") == 3 and str(ledger.get("limit_usd")) == "15",
             "Live Stage3 ledger has no hard15USD cap")
    _require(cost.get("provider_ledger_canonical_sha256") == _canonical(ledger),
             "Provider ledger changed after the full cost projection")
    _require(all(row.get("status") == "complete" for row in ledger.get("calls", {}).values()),
             "Unresolved prior input blocks scientific launch")
    _require(_decimal(ledger.get("guard_total_usd"), "current guard") <= peak,
             "Projected guard peak is below current charged/reserved spending")
    return {"status": "PASS", "prelaunch_receipt_sha256": _sha(receipt_raw),
            "approval_seq": approval["approval_seq"], "worker_smoke_run_id": worker["run_id"],
            "controller_validation_run_id": controller["run_id"], "smoke_inputs_verified": 34,
            "native_panels_verified": len(required_panels), "lid_closed_status": lid_status, "hard_cap_usd": "15",
            "projected_guard_peak_usd": str(peak), "post_summary_infrastructure_changes": post_summary_changes, "provider_mutations": 0}
