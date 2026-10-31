"""Model-independent shared handoff; no training, provider calls or credentials."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = '64555583-12e3-4641-82a0-b8841b663e70'
SHARED = Path('/Users/chasemetoyer/Desktop/evolved-intrinsic-rewards/handoff-cross-platform')
LOCAL = ROOT / 'handoff-cross-platform'


def now():
    return datetime.now(timezone.utc).isoformat()


def command(*args):
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return dict(exit_code=proc.returncode, stdout=proc.stdout.strip(), stderr=proc.stderr.strip())


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def snapshot_sources():
    result = command('git', 'ls-files', '--cached', '--others', '--exclude-standard')
    keep = {'generic_reward.py', 'README.md', 'AGENTS.md', 'CLAUDE.md',
            'pyproject.toml', 'uv.lock', '.python-version', '.gitignore'}
    prefixes = ('stage1/', 'stage2/', 'scripts/', 'tests/', 'preregistration/')
    paths = [p for p in result['stdout'].splitlines()
             if p in keep or p.startswith(prefixes)]
    paths = [p for p in paths if (ROOT / p).is_file()
             and not any(s in p.lower() for s in ('.env', 'credentials', 'token-secret'))]
    archive = SHARED / 'source-checkpoint.tar.gz'
    temporary = archive.with_name(archive.name + '.tmp')
    hashes = {}
    with tarfile.open(temporary, 'w:gz') as tar:
        for name in sorted(paths):
            p = ROOT / name
            raw = p.read_bytes()
            hashes[name] = hashlib.sha256(raw).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mode = p.stat().st_mode & 0o777
            tar.addfile(info, io.BytesIO(raw))
    temporary.replace(archive)
    return dict(path=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=hashes)


def checkpoint(args, *, automatic=False):
    SHARED.mkdir(parents=True, exist_ok=True)
    owner_path = SHARED / 'agent-owner.json'
    owner = read(owner_path)
    if args.claim:
        owner = dict(owner=args.owner, worktree=str(ROOT), claimed_utc=now(), previous_owner=owner.get('owner'))
        atomic(owner_path, owner)
    if args.release:
        if owner.get('owner') != args.owner:
            raise SystemExit('Only the recorded owner may release the lease')
        owner.update(released_utc=now(), owner=None)
        atomic(owner_path, owner)
    if args.owner and not args.release and owner.get('owner') != args.owner:
        raise SystemExit('Another agent owns editing; read the canonical handoff before work')
    previous = read(SHARED / 'session-state.json')
    status = command('git', 'status', '--short')
    runs = command('orx', 'runs', PROJECT)
    ledger_paths = [ROOT / 'results/budget-ledger.json']
    run_root = Path('/Users/chasemetoyer/.local/share/openresearch/local-runs')
    run_ids = re.findall(r'(?m)^([0-9a-f-]{36})\s', runs['stdout'])
    ledger_paths += [run_root / rid / 'repo/results/budget-ledger.json' for rid in run_ids]
    relevant = [p for p in ledger_paths if p.exists()]
    ledger = max(relevant, key=lambda p: p.stat().st_mtime) if relevant else None
    state = dict(updated_utc=now(), project_id=PROJECT, project_directory=str(SHARED.parent),
                 worktree=str(ROOT), branch=command('git', 'branch', '--show-current')['stdout'],
                 head_commit=command('git', 'rev-parse', 'HEAD')['stdout'], owner=owner,
                 note=args.note or previous.get('note'),
                 next_action=args.next_action or previous.get('next_action'),
                 stage1_lock_present=(ROOT / 'preregistration/stage1-lock.json').exists(),
                 git_status=status, run_registry=runs,
                 budget_ledger_path=str(ledger) if ledger else None,
                 budget_ledger=read(ledger) if ledger else None,
                 runtime_state=read(SHARED / 'runtime-state.json'),
                 stage2_runtime_state=read(SHARED / 'stage2-runtime-state.json'),
                 stage2_mode=read(ROOT / 'stage2/mode.json'),
                 stage2_lock_present=(ROOT / 'preregistration/stage2-lock.json').exists(),
                 source_checkpoint=snapshot_sources(),
                 usage_snapshot=read(SHARED / 'usage-snapshot.json'),
                 automatic=automatic, watcher_pid=os.getpid() if automatic else previous.get('watcher_pid'),
                 no_evolution_or_random_search=read(ROOT / 'stage2/mode.json').get('profile') != 'registered',
                 artifact_directory='/Users/chasemetoyer/.local/share/openresearch/files/evolved-intrinsic-rewards')
    atomic(SHARED / 'session-state.json', state)
    if not automatic:
        with (SHARED / 'session-log.jsonl').open('a') as stream:
            stream.write(json.dumps({k: state[k] for k in ('updated_utc', 'owner', 'note', 'next_action', 'head_commit', 'stage1_lock_present')}) + '\n')
    for name in ('session-state.json', 'agent-owner.json', 'usage-snapshot.json', 'session-log.jsonl'):
        p = SHARED / name
        if p.exists() and p.resolve() != (LOCAL / name).resolve():
            (LOCAL / name).write_bytes(p.read_bytes())
    print(json.dumps(dict(checkpoint=str(SHARED / 'session-state.json'), next_action=state['next_action'], head=state['head_commit'])), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--owner')
    p.add_argument('--claim', action='store_true')
    p.add_argument('--release', action='store_true')
    p.add_argument('--note')
    p.add_argument('--next-action')
    p.add_argument('--watch', action='store_true')
    p.add_argument('--interval', type=int, default=30)
    args = p.parse_args()
    if (args.claim or args.release) and not args.owner:
        p.error('An owner name is required')
    guard = (SHARED / '.checkpoint-write.lock').open('a')
    with guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        checkpoint(args)
        fcntl.flock(guard, fcntl.LOCK_UN)
    if args.watch:
        args.claim = args.release = False
        args.note = args.next_action = None
        while True:
            time.sleep(max(10, args.interval))
            with (SHARED / '.checkpoint-write.lock').open('a') as guard:
                fcntl.flock(guard, fcntl.LOCK_EX)
                checkpoint(args, automatic=True)


if __name__ == '__main__':
    main()
