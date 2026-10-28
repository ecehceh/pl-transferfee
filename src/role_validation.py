"""
Make the role templates defensible instead of hand-waved.

The role weights in roles.py are expert judgement: "a ball-playing centre-back
cares about pass accuracy roughly three times as much as clearances". There's
no labelled ground truth for the "right" weights, so these can't be trained.
What we *can* do is check them against the data and guard the one thing that
would quietly break them — redundancy.

If two of a role's metrics measure almost the same thing (advanced playmakers:
expected assists and big chances created correlate ~0.87), the role
double-counts that trait and a specialist in it dominates the ranking for the
wrong reason. This module:

  redundancy_report   the within-position correlation matrix of a role's
                      metrics, with the worst pairs called out.
  effective_weights   the role's weights deflated by how much each metric
                      *duplicates* the others, so shared information is counted
                      once. Barely changes an uncorrelated role; noticeably
                      shrinks a redundant one.
  weight_stability    a sensitivity check: jitter the weights and see how much
                      the top-N shortlist reshuffles. If the order barely moves,
                      the exact weights matter less than which metrics are in.

Everything is computed on the within-position percentile columns (pct_*), the
same space the scoring uses.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from roles import Role

REDUNDANT_THRESHOLD = 0.7  # |r| above this = the two metrics largely overlap


def _pct_frame(df: pd.DataFrame, role: Role) -> pd.DataFrame:
    """The role's metric percentiles for players in its position group."""
    pool = df[df["pos_group"] == role.pos_group]
    cols = [f"pct_{c}" for c in role.metric_columns()]
    return pool[cols].rename(columns=lambda c: c[len("pct_"):])


def redundancy_report(df: pd.DataFrame, role: Role) -> pd.DataFrame:
    """Every metric pair in the role, with its within-position correlation,
    worst (most redundant) first."""
    corr = _pct_frame(df, role).corr()
    metrics = role.metric_columns()
    rows = []
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            r = corr.iloc[i, j]
            rows.append({
                "metric_a": metrics[i],
                "metric_b": metrics[j],
                "corr": float(r),
                "redundant": bool(abs(r) >= REDUNDANT_THRESHOLD),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.reindex(out["corr"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def effective_weights(df: pd.DataFrame, role: Role) -> dict:
    """Deflate each metric's weight by how much it overlaps the role's other
    metrics, so correlated metrics share one vote instead of several.

    For metric i the redundancy load is the sum of its absolute correlations
    with the role's other metrics; the weight is divided by (1 + load) and the
    result renormalised to the original total. A metric uncorrelated with the
    rest is unchanged; one that duplicates two others is roughly halved.
    """
    metrics = role.metric_columns()
    raw = np.array([role.weights[m] for m in metrics], dtype=float)
    corr = _pct_frame(df, role).corr().to_numpy().copy()
    np.fill_diagonal(corr, 0.0)
    load = np.nan_to_num(np.abs(corr)).sum(axis=1)  # shared information per metric
    deflated = raw / (1.0 + load)
    deflated *= raw.sum() / deflated.sum()  # keep the same total weight
    return dict(zip(metrics, deflated))


def weight_stability(
    df: pd.DataFrame,
    role: Role,
    score_fn,
    n_trials: int = 50,
    jitter: float = 0.25,
    top_n: int = 20,
    seed: int = 0,
) -> dict:
    """How much the ranking moves when the weights are perturbed.

    `score_fn(pool, weights_dict) -> Series` scores the pool under a weight set
    (pass scouting.ability_score wrapped to take a dict). Each trial multiplies
    every weight by a random factor in [1-jitter, 1+jitter] and recomputes the
    ranking; we report the mean Spearman correlation of the full ranking and
    the mean overlap of the top-N against the unperturbed baseline.
    """
    pool = df[df["pos_group"] == role.pos_group].copy()
    rng = np.random.default_rng(seed)
    base = score_fn(pool, role.weights)
    base_order = base.rank(ascending=False)
    base_top = set(base.nlargest(top_n).index)

    rhos, overlaps = [], []
    for _ in range(n_trials):
        factors = rng.uniform(1 - jitter, 1 + jitter, size=len(role.weights))
        w = {m: role.weights[m] * f for m, f in zip(role.weights, factors)}
        s = score_fn(pool, w)
        rho = spearmanr(base_order, s.rank(ascending=False)).statistic
        rhos.append(rho)
        overlaps.append(len(base_top & set(s.nlargest(top_n).index)) / top_n)

    return {
        "role": role.name,
        "n_trials": n_trials,
        "jitter": jitter,
        "mean_rank_corr": float(np.mean(rhos)),
        "mean_top_n_overlap": float(np.mean(overlaps)),
        "top_n": top_n,
    }
