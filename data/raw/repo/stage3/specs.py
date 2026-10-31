"""Deterministic Stage3 configurations from exact frozen decoded rewards.

No optimizer or transcendental re-decoding is required to consume a frozen
reward: the JSON-decoded genome is the archived effective input specification.
"""
from __future__ import annotations
import copy,json,hashlib,tomllib
from pathlib import Path
from stage3.common import ROOT,canonical_sha
TASKS=('DoorKey-8x8','UnlockPickup','KeyCorridorS3R2')
METHODS=('E','R','T','N','S')
TAMPER_METHODS=('tamper_off','tamper_on')
EXPECTED_HASHES={'cma':'e3ec0a9f6b76cdc3a7c2c3da58ce8895b21b238c94db705ac4402d348d5e1fdc','random':'f197a64443341b6776433bd34c5a114023152ff7698abd07180e664b374bb000'}

def winners():
 answer={}
 for method,index in [('cma',113),('random',57)]:
  path=ROOT/'stage3/frozen'/f'frozen-winner-{method}.json';raw=path.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=EXPECTED_HASHES[method]:raise ValueError('Frozen winner hash changed:'+method)
  w=json.loads(raw)['winner']
  if w['candidate']!=index:raise ValueError('Frozen winner identity changed')
  answer['E' if method=='cma' else 'R']=w
 return answer

def configuration(method,task,seed,budget):
 if method not in METHODS+TAMPER_METHODS or task not in TASKS:raise ValueError('Unknown Stage3 arm')
 cfg=tomllib.loads((ROOT/'handoff-cross-platform/code/final-round5/config.toml').read_text())
 cfg['env']['id']='MiniGrid-'+task+'-v0';cfg['env']['eval_ids']=[cfg['env']['id']]
 cfg['env']['eval_episodes']=64
 cfg['run'].update(seeds=[seed],total_timesteps=budget)
 if method in ('E','R')+TAMPER_METHODS:
  w=winners()['E' if method in TAMPER_METHODS else method]
  cfg['intrinsic']={'kind':'generic_homeostatic','coef':1.,'h0':.5,**copy.deepcopy(w['genome']),'tamper':method=='tamper_on'}
 elif method=='T':cfg['intrinsic']={'kind':'none','coef':0.}
 elif method=='N':cfg['intrinsic']={'kind':'episodic_count','coef':.001}
 else:
  cfg['intrinsic'].update(kind='homeostatic',coef=1.,growth=[1/160]*3,tamper=False,
      event_mode='locked_door_key' if 'KeyCorridor' in task else 'doorkey')
 return cfg

def specs(profile):
 if profile not in ('smoke','science'):raise ValueError('Unknown Stage3 profile')
 budget=8192 if profile=='smoke' else 2_000_000
 rows=[]
 for mi,method in enumerate(METHODS):
  for ti,task in enumerate(TASKS):
   seeds=range(6000+100*mi+10*ti,6002+100*mi+10*ti) if profile=='smoke' else range(2000,2020)
   for seed in seeds:rows.append(spec(profile,method,task,seed,budget))
 for mi,method in enumerate(TAMPER_METHODS):
  seeds=range(7000+100*mi,7002+100*mi) if profile=='smoke' else range(2020,2040)
  for seed in seeds:rows.append(spec(profile,method,TASKS[0],seed,budget))
 return rows

def spec(profile,method,task,seed,budget):
 purpose='stage3_'+profile
 return {'id':f'{purpose}-{method}-{task}-{budget}-seed{seed}','purpose':purpose,'profile':profile,
     'method':method,'task':task,'seed':seed,'total_timesteps':budget,'replicate':0,
     'source':'final-round5','detector':'corrected','config':configuration(method,task,seed,budget)}

def validate(candidate):
 profile=candidate.get('profile')
 matching=[row for row in specs(profile) if row['id']==candidate.get('id')]
 if len(matching)!=1 or matching[0]!=candidate:raise ValueError('Stage3 specification differs from frozen plan')

def decode_record():
 # Compare the full scientific configuration for every registered E/R input,
 # preserving literal frozen float values rather than re-running search math.
 ws=winners();rows={r['id']:canonical_sha(r['config']) for r in specs('science') if r['method'] in ('E','R')}
 return {'winner_sha256':EXPECTED_HASHES,'candidate_ids':{m:w['candidate'] for m,w in ws.items()},
         'configuration_sha256':rows,'combined_sha256':canonical_sha(rows),
         'encoding':'Use exact frozen decoded genome floats; no optimizer regeneration or rounding.'}
