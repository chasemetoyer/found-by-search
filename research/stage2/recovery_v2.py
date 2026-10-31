"""Approved infrastructure recovery; scientific code and input specifications unchanged."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

AMENDMENT = 'preregistration/stage2-recovery2-amendment-20261028.json'
REPLACEMENTS = {'stage2/controller.py', 'stage2/registration.py'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


from stage2.transport import retry_io


def authorized_hashes(lock, root):
    from stage2.recovery import authorized_hashes as prior_hashes, AMENDMENT as prior_manifest
    effective = prior_hashes(lock, root)
    path = root/AMENDMENT
    if not path.exists():
        return effective
    a = json.loads(path.read_text())
    if a.get('approval_seq') != 27 or a.get('scientific_changes') is not False:
        raise ValueError('Unapproved second recovery amendment')
    if a['prior_amendment_sha256'] != sha((root/prior_manifest).read_bytes()):
        raise ValueError('Prior recovery amendment changed')
    if set(a['implementation_sha256']) != REPLACEMENTS:
        raise ValueError('Unexpected second recovery replacement scope')
    for name in REPLACEMENTS:
        if sha((root/'preregistration/frozen-recovery1-stage2'/name).read_bytes()) != effective[name]:
            raise ValueError('First recovery runtime not preserved: '+name)
    for name, expected in a['added_file_sha256'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or sha((root/p).read_bytes()) != expected:
            raise ValueError('Second recovery helper changed: '+name)
    effective.update(a['implementation_sha256'])
    return effective


def reuse_previous_output(remote, local, expected):
    from stage2.common import ROOT
    a = json.loads((ROOT/AMENDMENT).read_text())
    parts = Path(remote).parts
    if parts[0] != 'runs' or '..' in parts:
        raise ValueError('Invalid recovery output path')
    previous = Path(a['previous_results_directory']).joinpath(*parts[1:])
    if not previous.is_file() or sha(previous.read_bytes()) != expected:
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(previous, local)
    except FileExistsError:
        local.unlink()
        os.link(previous, local)
    except OSError:
        shutil.copyfile(previous, local)
    return True


def reconcile_interrupted(ledger, volume, results, root=None, query=None):
    from stage2.common import ROOT, utc_now
    import modal
    root = ROOT if root is None else root
    path = root/AMENDMENT
    if not path.exists():
        return
    a = json.loads(path.read_text())
    identity = sha(path.read_bytes())
    if identity in ledger.data.get('recovery_amendments', []):
        return  # never turn a future unrelated failure into an authorized retry
    if canonical(ledger.data) != a['interrupted_ledger_canonical_sha256']:
        raise ValueError('Ledger changed since audited interruption; reconcile before retry')
    query = query or (lambda call_id: modal.FunctionCall.from_id(call_id).get_call_graph())
    decisions = []
    for key, saved in a['interrupted_calls'].items():
        c = ledger.data['calls'][key]
        if c['status'] != 'uncertain' or c['function_call_id'] != saved['function_call_id']:
            raise ValueError('Interrupted input differs: '+key)
        graph = retry_io(lambda: query(c['function_call_id']))
        statuses = [g.status.name for g in graph]
        if not statuses or any(s != saved['provider_status'] for s in statuses):
            raise ValueError('Provider state changed or is not terminal: '+key)
        if saved['provider_status'] == 'SUCCESS':
            marker = json.loads(retry_io(lambda: b''.join(volume.read_file('/runs/'+key+'/complete.json'))))
            if (marker['id'] != key or marker['spec_sha256'] != c['spec_sha256'] or
                canonical(marker) != saved['completion_marker_canonical_sha256']):
                raise ValueError('Recovered output identity changed: '+key)
            if marker['outcome'] not in ('success', 'candidate_failure'):
                raise ValueError('Invalid recovered outcome')
            decisions.append((key, marker))
        elif saved['provider_status'] == 'TERMINATED':
            # Prove absence independently; never overwrite an existing result.
            names = retry_io(lambda: [e.path for e in volume.listdir('/runs/'+key, recursive=True)])
            if any(n.endswith('/complete.json') for n in names):
                raise ValueError('Terminated input now has saved output: '+key)
            decisions.append((key, None))
        else:
            raise ValueError('Recovery is limited to audited terminal states')
    # All queries validated before any ledger mutation; retain every old charge.
    for key, marker in decisions:
        c = ledger.data['calls'][key]
        c['reconciled_utc'] = utc_now()
        c['recovery_amendment'] = identity
        if marker:
            c.update(status='complete', outcome=marker['outcome'], failure=marker.get('failure'),
                     measured=marker['usage'], completed_utc=utc_now(),
                     recovery_note='Saved successful output; original full reservation retained conservatively')
        else:
            c.update(status='retry_pending', provider_terminal_status='TERMINATED',
                     recovery_note='Identical registered input authorized for a new provider call; old full reservation retained')
    ledger.data.setdefault('recovery_amendments', []).append(identity)
    ledger.save()
    (results/'recovery-reconciliation.json').write_text(json.dumps({
        'amendment_sha256':identity, 'recovered_saved_outputs':sum(m is not None for _,m in decisions),
        'terminated_inputs_for_identical_retry':sum(m is None for _,m in decisions),
        'previous_charges_retained':True},indent=2))
