"""Archived statistical procedures, with general-purpose Holm and two-sided support.

Source: handoff-cross-platform/scripts/score_round5.py, functions cp, mwu,
jt, fisher and the Holm loop. Numerical procedure and permutation seed/count
are retained. ``mwu`` accepts ``two-sided`` for the registered calibration
comparison; its tied-data permutation tests the same U statistic.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta, fisher_exact, mannwhitneyu, permutation_test


def cp(k, n, a=0.05):
    return [0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))]


def mwu(x, y, alt):
    if alt not in {"less", "greater", "two-sided"}:
        raise ValueError("alternative must be less, greater, or two-sided")
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("MWU requires nonempty finite samples")
    if len(set(x) | set(y)) == len(x) + len(y):
        return float(mannwhitneyu(x, y, alternative=alt, method="exact").pvalue), "exact"
    stat = lambda a, b: mannwhitneyu(a, b, alternative=alt, method="asymptotic").statistic
    return float(permutation_test((x, y), stat, permutation_type="independent", alternative=alt,
                                  n_resamples=200_000, random_state=0).pvalue), "permutation (ties)"


def jt(groups, n_perm=200_000, seed=0):
    """Jonckheere-Terpstra; increasing alternative; ties receive half credit."""
    vals = np.concatenate([np.asarray(g, float) for g in groups]); lab = np.concatenate([[i] * len(g) for i, g in enumerate(groups)])
    def J(v, l):
        lt = (v[:, None] < v[None, :]) + 0.5 * (v[:, None] == v[None, :])
        return float((lt * (l[:, None] < l[None, :])).sum())
    obs = J(vals, lab); rng = np.random.default_rng(seed); ge = 0; done = 0
    order = lab[:, None] < lab[None, :]
    while done < n_perm:
        k = min(2000, n_perm - done)
        P_ = np.argsort(rng.random((k, len(vals))), axis=1); v = vals[P_]
        lt = (v[:, :, None] < v[:, None, :]) + 0.5 * (v[:, :, None] == v[:, None, :])
        ge += int(((lt * order).sum(axis=(1, 2)) >= obs - 1e-9).sum()); done += k
    return (ge + 1) / (n_perm + 1), obs


def fisher(k1, n1, k2, n2, alt):
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative=alt)[1])


def holm(p_values, *, family_size=None):
    """Holm-adjust p-values in input order; preserve missing entries as None.

    ``family_size`` retains the full preregistered family when a planned test
    cannot be calculated. Unavailable tests are effectively assigned p=1 for
    adjusting the observed tests, while their own result remains unavailable.
    """
    p_values = list(p_values)
    observed = [(i, float(p)) for i, p in enumerate(p_values) if p is not None]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for _, p in observed):
        raise ValueError("p-values must be finite and in [0, 1]")
    m = len(observed) if family_size is None else family_size
    if not isinstance(m, int) or m < len(observed):
        raise ValueError("family_size must be an integer at least the number of observed tests")
    out = [None] * len(p_values)
    previous = 0.0
    for rank, (index, p) in enumerate(sorted(observed, key=lambda pair: pair[1])):
        previous = max(previous, min(1.0, (m - rank) * p))
        out[index] = previous
    return out
