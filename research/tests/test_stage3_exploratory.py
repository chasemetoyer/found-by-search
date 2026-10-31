"""Fixtures only: no observed Stage2 evidence is inspected by these tests."""
import math
import pytest
from stage3.exploratory import average_ranks,spearman,quantile,candidate_features,analyze_candidates


def test_average_ranks_ties():
    assert average_ranks([3,1,1,4]) == [3,1.5,1.5,4]
    assert math.isclose(spearman([1,1,3],[1,2,3])["rho"],math.sqrt(3)/2)
    assert spearman([1,2,3],[3,2,1])["rho"] == -1


def test_undefined_correlations_and_empty_groups_are_explicit():
    assert spearman([0,0],[1,2])["status"] == "undefined: constant feature"
    assert spearman([1,2],[0,0])["status"] == "undefined: constant area"
    assert spearman([],[])["rho"] is None
    assert quantile([], .25) is None
    with pytest.raises(ValueError):
        average_ranks([float('nan')])


def test_boolean_requires_positive_weight_and_refill_on_same_object_event():
    genome={"weights":[1,1,0,-1],"relief":[.5]*4,"growth":[1/20,0,1/20,1/20]}
    assert candidate_features(genome)["any_object_channel_positive_weight_and_refill"]==0
    genome['growth'][1]=1/640
    assert candidate_features(genome)["any_object_channel_positive_weight_and_refill"]==1


def test_all_groups_and_conditional_samples_are_retained():
    rows=[]
    for method in ('cma','random'):
        for i,(w,g) in enumerate([(-1,0),(0,0),(.5,1/640),(1,1/20)]):
            rows.append({'method':method,'features':candidate_features({'weights':[w]*4,'relief':[.5]*4,'growth':[g]*4}),
                         'areas':{'combined':i/3,'DoorKey-8x8':i/3,'UnlockPickup':i/3}})
    result=analyze_candidates(rows)
    assert result['candidate_counts']=={'pooled':8,'cma':4,'random':4}
    assert result['p_values_computed'] is False
    assert len(result['correlations'])==3*3*(17+4)
    conditional=[r for r in result['correlations'] if r['stratum']=='pooled' and r['subset']=='positive refill only']
    assert len(conditional)==12 and all(r['n']==4 for r in conditional)
    assert all(r['rho'] is None for r in result['correlations'] if r['feature'].startswith('relief/'))
    groups=[g for g in result['groups'] if g['stratum']=='pooled' and g['target']=='combined' and g['event']=='pickup']
    assert len(groups)==11
    empty=next(g for g in groups if g.get('weight_sign')=='negative' and g.get('refill_state')=='nonzero')
    assert empty['n']==0 and empty['mean'] is empty['median'] is empty['iqr'] is None
    sign=next(g for g in groups if g['grouping']=='weight sign' and g['weight_sign']=='positive')
    assert sign['n']==4 and sign['mean']==pytest.approx(5/6)
    assert sign['iqr']==pytest.approx(1/3)


def test_candidate_builder_rejects_an_incomplete_population(tmp_path):
    import json
    from stage3.exploratory import build_candidates
    scores=tmp_path/'scores.csv';scores.write_text('purpose,method,candidate,group,id,task,seed,area\n')
    selection=tmp_path/'selection.json';selection.write_text(json.dumps({'methods':{}}))
    with pytest.raises(ValueError,match='all240'):
        build_candidates(tmp_path,scores,selection)


def test_one_nonfinite_observation_cannot_hide_as_an_undefined_sample():
    with pytest.raises(ValueError,match='finite'):
        spearman([float('nan')],[1])


def test_frozen_population_loads_all240_and_detects_corrupt_spec(tmp_path):
    import csv,hashlib,json,statistics
    from stage3.exploratory import build_candidates
    scored=[];selection={'methods':{m:{'search_fitness':{}} for m in ('cma','random')}};last=None
    for method in ('cma','random'):
        for candidate in range(120):
            group=candidate//10;base=(1000 if method=='cma' else 3000)+3*group;areas=[]
            for task in ('DoorKey-8x8','UnlockPickup'):
                for i in range(3):
                    key=f'search_{method}-fixture-c{candidate:03d}-{task}-seed{base+i}'
                    spec={'id':key,'method':method,'candidate':candidate,'group':group,'task':task,'seed':base+i,
                          'genome':{'weights':[0.]*4,'relief':[.5]*4,'growth':[0.]*4},'latent':[.5]*8+[.25]*4}
                    raw=json.dumps(spec,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                    directory=tmp_path/key/'attempts'/'fixture';directory.mkdir(parents=True);last=directory/'spec.json';last.write_bytes(raw)
                    (tmp_path/key/'complete.json').write_text(json.dumps({'id':key,'attempt':'fixture','spec_sha256':hashlib.sha256(raw).hexdigest(),
                          'files':{f'runs/{key}/attempts/fixture/spec.json':hashlib.sha256(raw).hexdigest()}}))
                    area=candidate/240+i/100;areas.append(area)
                    scored.append({'purpose':'search_'+method,'method':method,'candidate':candidate,'group':group,'id':key,'task':task,'seed':base+i,'area':area})
            selection['methods'][method]['search_fitness'][str(candidate)]=statistics.fmean(areas)
    scores=tmp_path/'scores.csv'
    with scores.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(scored[0]));writer.writeheader();writer.writerows(scored)
    frozen=tmp_path/'selection.json';frozen.write_text(json.dumps(selection))
    loaded=build_candidates(tmp_path,scores,frozen)
    assert len(loaded)==240 and {r['candidate'] for r in loaded}==set(range(120))
    assert loaded[0]['areas']['combined']==pytest.approx(.01)
    last.write_bytes(last.read_bytes()+b' ')
    with pytest.raises(ValueError,match='file hash mismatch'):
        build_candidates(tmp_path,scores,frozen)
