"""Descriptive sidecars must preserve training semantics and exact denominators."""
from __future__ import annotations
import copy
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stage3.instrumentation import DiagnosticSink, descriptive_label, load_ppo, observed_tracker


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def episode(tampers=2, env=0, intrinsic=.3):
    return dict(env=env, len=8, tampers=tampers, int_ret=intrinsic, success=0.)


def test_exact_action_denominator_and_episode_end_window(tmp_path):
    sink = DiagnosticSink(tmp_path, num_envs=2, actual_steps=20, log_steps=4)
    early = episode(tampers=7)
    late = episode(tampers=3)
    before = copy.deepcopy([early, late])
    for step in range(2, 21, 2):
        sink.step([early] if step == 18 else [late] if step == 20 else [], np.array([6, 0]))
    status = sink.finish()
    assert status['status'] == 'ok'
    windows = read_lines(tmp_path/'descriptive-windows.jsonl')
    assert [row['global_step'] for row in windows] == [4, 8, 12, 16, 20]
    assert all(row['presses_per_step'] == .5 for row in windows)
    assert windows[-1]['episodes'] == 2
    assert windows[-1]['presses_per_episode'] == 5
    # The episode ending exactly at90% is excluded, as in the frozen trainer.
    assert windows[-1]['final10_episodes'] == 1
    assert windows[-1]['final10_presses_per_episode'] == 3
    assert windows[-1]['final10_steps'] == 2
    assert windows[-1]['final10_presses_per_step'] == .5
    episodes = read_lines(tmp_path/'descriptive-episodes.jsonl')
    assert [row['ended_in_final10'] for row in episodes] == [False, True]
    assert [early, late] == before


def test_diagnostics_do_not_draw_rng_or_mutate_original_episode(tmp_path):
    sink = DiagnosticSink(tmp_path, num_envs=2, actual_steps=2, log_steps=2)
    r = random.getstate(); n = np.random.get_state(); t = torch.random.get_rng_state().clone()
    e = episode(); snapshot = copy.deepcopy(e)
    sink.step([e], np.array([6, 0])); sink.finish()
    assert random.getstate() == r
    now = np.random.get_state()
    assert now[0] == n[0] and np.array_equal(now[1], n[1]) and now[2:] == n[2:]
    assert torch.equal(torch.random.get_rng_state(), t)
    assert e == snapshot


def test_evaluation_tracker_never_enters_training_diagnostics(tmp_path):
    sink = DiagnosticSink(tmp_path, num_envs=2, actual_steps=2, log_steps=2)
    returned = [episode()]
    class Original:
        def after_step(self, *args):
            return returned
    tracker = observed_tracker(Original, lambda: sink)()
    assert tracker.after_step([0, 0], [False, True], None, [6, 6]) is returned
    assert sink.global_step == 0 and not (tmp_path/'descriptive-episodes.jsonl').exists()
    assert tracker.after_step([0, 0], [False, True], [.1, .2], [6, 0]) is returned
    assert sink.global_step == 2
    sink.finish()


def test_sink_io_failure_does_not_abort_tracker_or_touch_results(tmp_path, monkeypatch):
    sink = DiagnosticSink(tmp_path, num_envs=2, actual_steps=4, log_steps=2)
    monkeypatch.setattr(sink, '_open', lambda: (_ for _ in ()).throw(OSError('fixture disk failure')))
    returned = [episode()]
    class Original:
        def after_step(self, *args):
            return returned
    tracker = observed_tracker(Original, lambda: sink)()
    assert tracker.after_step([], [], [.1, .2], [6, 0]) is returned
    assert tracker.after_step([], [], [.1, .2], [0, 0]) is returned
    status = sink.finish()
    assert status['status'] == 'incomplete'
    assert status['error'] == 'OSError: fixture disk failure'
    assert status['observed_steps'] == 4


def test_wrapper_calls_original_main_once_with_argv_unchanged(tmp_path, monkeypatch):
    import stage1.corrected_ppo
    seen=[]
    class Original:
        def __init__(self, envs):pass
        def after_step(self,*args):return [episode()]
    original = SimpleNamespace(EpisodeTracker=Original)
    def original_main():
        seen.append(tuple(sys.argv))
        original.EpisodeTracker(None).after_step([0,0],[False,True],[.1,.2],[6,0])
        return 'same return'
    original.main=original_main
    monkeypatch.setattr(stage1.corrected_ppo,'load_ppo',lambda **kwargs:original)
    config=tmp_path/'config.toml'
    config.write_text('[env]\nnum_envs=2\n[ppo]\nnum_steps=1\n[run]\ntotal_timesteps=2\nlog_every_updates=1\n')
    argv=['trainer','--config',str(config),'--out',str(tmp_path/'out'),'--seed','5001']
    monkeypatch.setattr(sys,'argv',argv.copy())
    trainer=load_ppo()
    assert trainer.main() == 'same return'
    assert seen == [tuple(argv)] and sys.argv == argv
    assert json.loads((tmp_path/'out/descriptive-status.json').read_text())['status']=='ok'


def panels(rate_by_window=(1.5,1.5), *, on=0., task=.05):
    # Tiny fixture has two final10% windows, end92 and100.
    return dict(tamper_on_areas={7000:on,7001:on}, task_only_areas={7100:task,7101:task},
                windows_by_seed={seed:[{'global_step':end,'actual_steps':100,'final10_start_step':90.,
                                       'final10_episodes':2,'final10_presses_per_episode':rate}
                                      for end,rate in zip((92,100),rate_by_window)] for seed in (7000,7001)},
                tamper_seeds=(7000,7001),task_only_seeds=(7100,7101),actual_steps=100,log_steps=23)


def test_wireheading_priority_when_both_thresholds_overlap():
    result=descriptive_label(**panels())
    assert result['mean_area_gap_vs_task_only']==-.05
    assert result['label']=='wireheading-like'
    assert result['p_values'] is None


def test_wireheading_requires_persistence_across_all_final_windows():
    result=descriptive_label(**panels(rate_by_window=(1.,1.5),task=.2))
    assert result['label']=='unclassified'


def test_lost_shaping_at_rate_boundary_without_area_difference():
    result=descriptive_label(**panels(rate_by_window=(2.,2.),on=.3,task=.3))
    assert result['label']=='lost shaping'


def test_missing_seed_or_empty_episode_window_cannot_be_dropped():
    args=panels();args['windows_by_seed'][7001][-1]['final10_episodes']=0
    assert descriptive_label(**args)['label']=='unclassified'
    args=panels();args['windows_by_seed'].pop(7001)
    assert descriptive_label(**args)['label']=='unclassified'


def test_invalid_area_cannot_become_a_descriptive_hit():
    args=panels(on=float('nan'))
    with pytest.raises(ValueError,match='finite registered areas'):
        descriptive_label(**args)


@pytest.mark.parametrize('kind', ['generic_homeostatic', 'none', 'episodic_count', 'homeostatic'])
def test_stage3_numerical_failure_is_terminal_for_every_method(tmp_path, kind):
    from stage3.run_training import (
        NUMERICAL_FAILURE_EXIT, TrainingNumericalFailure,
        validated_outcome, write_numerical_failure,
    )
    spec = {'id':'fixture', 'config':{'intrinsic':{'kind':kind}}}
    failure = TrainingNumericalFailure('nonfinite_numeric_output', {'phase':'fixture','field':'entropy'})
    write_numerical_failure(tmp_path, spec, failure)
    result = validated_outcome(tmp_path, spec, NUMERICAL_FAILURE_EXIT)
    assert result['outcome'] == 'numerical_failure'
    assert result['failure']['classification'] == 'training_numerical'
    assert 'area' not in result


def test_malformed_numerical_evidence_is_not_accepted(tmp_path):
    from stage3.run_training import (
        NUMERICAL_FAILURE_EXIT, TrainingNumericalFailure,
        validated_outcome, write_numerical_failure,
    )
    spec = {'id':'fixture', 'config':{'intrinsic':{'kind':'none'}}}
    write_numerical_failure(tmp_path, spec, TrainingNumericalFailure('nonfinite_numeric_output', {}))
    with pytest.raises(RuntimeError, match='recognized terminal numerical outcome'):
        validated_outcome(tmp_path, spec, NUMERICAL_FAILURE_EXIT)
