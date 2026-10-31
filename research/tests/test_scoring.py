"""Known statistical examples and scoring distinctions fixed before evidence."""
import json
import math

import pytest

from stage1.score import IncompleteRun, aggregate_calibration, score_seed
from stage1.statistics import cp, fisher, holm, jt, mwu


def test_fisher_complete_separation():
    # Conditional probability of all five successes falling in the first arm.
    assert fisher(5, 5, 0, 5, "greater") == pytest.approx(1 / math.comb(10, 5))


def test_mwu_two_sided_without_ties():
    p, method = mwu([1, 2, 3], [4, 5, 6], "two-sided")
    assert p == pytest.approx(2 / math.comb(6, 3))
    assert method == "exact"


def test_tied_mwu_two_sided_counts_both_extremes():
    p, method = mwu([0] * 5, [1] * 5, "two-sided")
    assert p == pytest.approx(2 / math.comb(10, 5))
    assert method == "permutation (ties)"
    assert mwu([1] * 5, [0] * 5, "two-sided")[0] == pytest.approx(p)


def test_jt_all_ties_has_half_pair_credit_and_p_one():
    p, statistic = jt([[7, 7], [7, 7], [7, 7]], n_perm=19)
    assert statistic == 6  # Three ordered group pairs, four pairs each, half credit.
    assert p == 1


def test_clopper_pearson_all_failures_and_all_successes():
    assert cp(0, 5) == pytest.approx([0, 1 - 0.025 ** (1 / 5)])
    assert cp(5, 5) == pytest.approx([0.025 ** (1 / 5), 1])


def test_holm_preserves_order_and_planned_missing_test():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.01, None, 0.03], family_size=3) == [0.03, None, 0.06]


def write_run(path, successes, *, first_solved=None, first_goal=None):
    rows = [{"global_step": (i + 1) * 20480, "train/success": value} for i, value in enumerate(successes)]
    (path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (path / "summary.json").write_text(json.dumps({"seed": 900, "total_timesteps": len(rows) * 20480,
        "first_solved_step": first_solved, "first_goal_step": first_goal}))
    return len(rows) * 20480


def test_ever_solved_is_not_final_solve(tmp_path):
    budget = write_run(tmp_path, [1] * 9 + [0], first_solved=20480, first_goal=16)
    score = score_seed(tmp_path, budget)
    assert score["curve_area"] == pytest.approx(0.9)
    assert not score["solved"]
    assert score["success_ever"]
    assert score["first_solved_censored"] == 20480


def test_unsolved_first_step_is_right_censored(tmp_path):
    budget = write_run(tmp_path, [0] * 10)
    score = score_seed(tmp_path, budget)
    assert score["first_solved_censored"] == budget
    assert score["first_solved_is_censored"]
    assert score["cost_usd"] is None


def test_missing_log_row_is_not_zero_success(tmp_path):
    budget = write_run(tmp_path, [0] * 10)
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    (tmp_path / "metrics.jsonl").write_text("\n".join(lines[1:]))
    with pytest.raises(IncompleteRun, match="missing"):
        score_seed(tmp_path, budget)


def task_records(task, area, learnable=True, missing=None):
    specs, scores = [], {}
    for method in ("none", "episodic_count"):
        for budget in (1000000, 2000000):
            for seed in range(900, 905):
                rid = f"{task}-{method}-{budget}-{seed}"
                specs.append(dict(id=rid, purpose="calibration", task=task, method=method,
                                  seed=seed, total_timesteps=budget))
                if (method, budget, seed) == missing:
                    continue
                scores[rid] = dict(seed=seed, nominal_budget=budget, curve_area=area,
                    solved=False, final_success=0, first_solved_censored=budget,
                    first_solved_is_censored=True, success_ever=learnable,
                    wall_seconds=None, process_core_seconds=None, peak_rss_bytes=None,
                    cost_usd=None, cost_estimate_usd=None)
    return {"runs": specs}, scores


@pytest.mark.parametrize("area,learnable,expected", [(0.85, True, "kept"), (0.850001, True, "rejected"), (0.2, False, "rejected")])
def test_training_selection_fixed_threshold_and_learnability(area, learnable, expected):
    manifest, scores = task_records("DoorKey-8x8", area, learnable)
    result = aggregate_calibration(manifest, scores)
    assert next(row for row in result["selection"] if row["task"] == "DoorKey-8x8")["status"] == expected


@pytest.mark.parametrize("area,expected", [(0.05, "kept"), (0.85, "kept"), (0.049999, "rejected"), (0.850001, "rejected")])
def test_held_out_selection_is_inclusive(area, expected):
    manifest, scores = task_records("DoorKey-16x16", area)
    result = aggregate_calibration(manifest, scores)
    assert next(row for row in result["selection"] if row["task"] == "DoorKey-16x16")["status"] == expected


def test_partial_cell_never_selects_task():
    manifest, scores = task_records("DoorKey-8x8", 0.2, missing=("none", 2000000, 904))
    result = aggregate_calibration(manifest, scores)
    choice = next(row for row in result["selection"] if row["task"] == "DoorKey-8x8")
    assert choice["status"] == "incomplete"
    assert choice["kept"] is None
