"""Verify immutable inherited protocol and committed Stage3 implementation hashes."""
import json,hashlib
from stage3.common import ROOT

def verify_sources():
 from stage2.registration import authorize,mode_record
 authorize(mode_record())
 path=ROOT/'preregistration/stage3-runtime-manifest.json'
 manifest=json.loads(path.read_text())
 for name,expected in manifest['sha256'].items():
  if name.startswith('/') or '..' in name.split('/'):raise ValueError('Invalid runtime manifest path')
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Stage3 source changed:'+name)
 return manifest
