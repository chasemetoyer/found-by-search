"""Read-only Stage3 billing attribution and a conservative billing floor.

Only rows tagged with this project AND stage3 enter the total, including the
separate tracking role. Native Sandbox costs without those tags remain
unattributed: their controller forecast reservation is not an invoice claim.
Callers persist the returned evidence to their own control volume.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Iterable

PROJECT_ID = '64555583-12e3-4641-82a0-b8841b663e70'
STAGE = '3'
TRACKING_SECOND = Decimal('.25') * Decimal('.0000131') + Decimal('.5') * Decimal('.00000222')
TRACKING_LEASE_SECONDS = 12 * 3600


def money(value) -> Decimal:
    if isinstance(value, bool):
        raise ValueError('Boolean is not a money amount')
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError('Invalid money amount') from error
    if not amount.is_finite() or amount < 0:
        raise ValueError('Money amount must be finite and nonnegative')
    return amount


def _plain(value):
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def attribution_report(rows: Iterable, *, project_id: str = PROJECT_ID,
                       previous_total_usd='0', queried_utc: datetime | None = None,
                       native_controller_proxy_usd=None) -> dict:
    """Filter before reading costs; never charge other projects/stages to Stage3.

    The retained floor cannot fall after a later report omits or revises rows.
    The native-controller proxy is contextual evidence only; the ledger already
    reserves it, so it is never added to the provider total here.
    """
    selected, excluded = [], {'different_project': 0, 'different_or_missing_stage': 0}
    roles = {}
    total = Decimal(0)
    for row in rows:
        row = _plain(row)
        if not isinstance(row, dict):
            raise ValueError('Billing row must be a mapping or dataclass')
        tags = row.get('tags') or {}
        if tags.get('project_id') != project_id:
            excluded['different_project'] += 1
            continue
        if tags.get('stage') != STAGE:
            excluded['different_or_missing_stage'] += 1
            continue
        amount = money(row.get('cost'))
        role = str(tags.get('role') or 'unlabeled_stage3')
        roles[role] = roles.get(role, Decimal(0)) + amount
        # Retain attributable evidence, not unrelated workspace rows or secrets.
        selected.append({key: row[key] for key in ('cost', 'cost_by_resource', 'description',
                         'environment_name', 'interval_start', 'object_id', 'tags') if key in row})
        total += amount
    previous = money(previous_total_usd)
    proxy = None if native_controller_proxy_usd is None else str(money(native_controller_proxy_usd))
    when = queried_utc or datetime.now(timezone.utc)
    return {
        'status': 'ok', 'queried_utc': when.isoformat(), 'resolution': 'h',
        'project_id': project_id, 'stage': STAGE,
        'provider_reported_complete_intervals_cost_usd': str(total),
        'retained_billed_floor_usd': str(max(total, previous)),
        'cost_by_role_usd': {key: str(value) for key, value in sorted(roles.items())},
        'rows': selected, 'excluded_row_counts': excluded,
        'native_controller_proxy_usd': proxy,
        'native_controller_invoice_usd': None,
        'native_controller_attribution': 'Unknown unless the provider row carries both project_id and stage=3. The native Sandbox resource proxy stays in the ledger; missing tagged billing is not zero cost.',
        'coverage': 'Complete hourly report intervals only; recent costs, native Sandbox attribution, volume/egress or other untagged costs may be absent. Not a final invoice or a hard invoice cap.',
        'read_only_provider_query': True,
    }


def query_billing(start: datetime, *, workspace=None, project_id: str = PROJECT_ID,
                  previous_total_usd='0', native_controller_proxy_usd=None,
                  queried_utc: datetime | None = None) -> dict:
    """Call the same Workspace billing report API as Stage2, with role tags too.

    Query/validation failures retain the previous valid billing floor. The
    caller decides whether to pause dispatch and persists this returned record.
    No provider mutations, settings changes or training calls occur here.
    """
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError('Billing start must include its timezone')
    previous = money(previous_total_usd)
    if native_controller_proxy_usd is not None:
        money(native_controller_proxy_usd)
    when = queried_utc or datetime.now(timezone.utc)
    try:
        if workspace is None:
            import modal
            workspace = modal.Workspace.from_context()
        rows = workspace.billing.report(start=start, resolution='h',
                                        tag_names=['project_id', 'stage', 'role'])
        result = attribution_report(rows, project_id=project_id,
            previous_total_usd=previous, native_controller_proxy_usd=native_controller_proxy_usd,
            queried_utc=when)
        result['period_start_utc'] = start.astimezone(timezone.utc).isoformat()
        return result
    except Exception as error:
        # Exception messages may contain credential-bearing URLs. Save type only.
        return {
            'status': 'error', 'queried_utc': when.isoformat(),
            'period_start_utc': start.astimezone(timezone.utc).isoformat(),
            'project_id': project_id, 'stage': STAGE,
            'error_type': type(error).__name__,
            'provider_reported_complete_intervals_cost_usd': None,
            'retained_billed_floor_usd': str(previous),
            'native_controller_proxy_usd': None if native_controller_proxy_usd is None else str(money(native_controller_proxy_usd)),
            'native_controller_invoice_usd': None,
            'coverage': 'Billing query failed; previous valid floor retained. Missing provider evidence is not zero cost.',
            'read_only_provider_query': True,
        }


def effective_guard(estimated_guard_usd, evidence: dict, *, overhead_usd='.50') -> Decimal:
    """Compare an estimate already including overhead with billed floor+overhead.

    This max prevents double-counting tagged provider costs, while late or higher
    billing cannot evade the cap. Controller/tracker reservations remain in the
    estimate; never add their proxy again to the provider total.
    """
    estimated = money(estimated_guard_usd)
    billed = money(evidence['retained_billed_floor_usd'])
    return max(estimated, billed + money(overhead_usd))


def tracking_lease_reservation(seconds: int = TRACKING_LEASE_SECONDS) -> Decimal:
    """Forecast .25physicalcore/.5GiB service with strict caps, max1container.

    The caller must enforce those limits and finite lease, and refresh the full
    reservation at scientific launch. A late request/503 can still cold-start
    compute; TTL alone cannot promise zero cost after expiry. Disable the
    deployment through its authenticated control path to end that exposure.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
        raise ValueError('Tracking lease must be a positive integer duration')
    return TRACKING_SECOND * seconds
