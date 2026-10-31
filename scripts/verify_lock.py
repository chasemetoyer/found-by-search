"""Verify the pre-registration locks: every file a lock hashed must still match its SHA256.

The Stage 1 and Stage 2 locks (research/preregistration/stage1-lock.json, stage2-lock.json) record the SHA256 of
each registered file, by path relative to the research folder. Those files are kept byte-identical here, including
a few that contain local paths from the machine that ran them, so the hashes still verify.

Usage (repo root): make verify-lock
"""
import hashlib
import json
import re
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1] / "research"


def hashed_files(obj):
    """Every {relative path: sha256} entry anywhere in a lock file."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) and ("/" in k or "." in k):
                yield k, v
            else:
                yield from hashed_files(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from hashed_files(v)


# Later amendments replaced some registered files and recorded the replacement hashes; a file passes if it matches
# its lock hash or a hash recorded for that path by any later lock or amendment.
recorded = {}
for f in sorted((RESEARCH / "preregistration").glob("*.json")):
    for rel, h in hashed_files(json.loads(f.read_text())):
        recorded.setdefault(rel, set()).add(h)
bad = checked = amended = 0
for lock in ("stage1-lock.json", "stage2-lock.json"):
    for rel, want in hashed_files(json.loads((RESEARCH / "preregistration" / lock).read_text())):
        path = RESEARCH / rel
        if not path.is_file():
            continue  # paths outside this release are not copied
        checked += 1
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if h == want:
            continue
        if h in recorded.get(rel, ()):
            amended += 1
            continue
        bad += 1
        print(f"MISMATCH {lock}: {rel}")
print(f"{'PASS' if not bad else 'FAIL'}: {checked} locked file entries checked; "
      f"{checked - amended - bad} match the lock, {amended} match a hash recorded by a later amendment, {bad} mismatches")
sys.exit(1 if bad else 0)
