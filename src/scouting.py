"""
The club-fit scoring engine.

Given the scouting table (scout_prep.py), a target role (roles.py) and a budget,
score every candidate in that role's position group on three pillars and rank
them:

  ABILITY  how good the player is at the things the role cares about. The
           weighted average of their within-position percentiles on the role's
           metrics. 0-100; 70 means "better than 70% of players in this
           position at this role's priorities".

  FIT      how closely the player's *shape* matches the role template,
           regardless of overall level. Cosine similarity between the player's
           percentile vector and the role's weight vector, rescaled to 0-100. A
           solid, balanced player can have high ability but mediocre fit if the
           role wants a specialist, and vice versa.

  VALUE    affordability and bang-for-buck. Combines (a) whether the player is
           within budget and (b) how much ability you get per euro compared
           with other candidates in this role. 0-100.

The overall score is a weighted blend of the three, weights adjustable so a club
can say "we care about fit over raw ability" or "value matters most".

Nothing here reads files at import time, so it's cheap to unit-test.
"""

import numpy as np
import pandas as pd

from roles import METRIC_LABEL, Role

PILLARS = ("ability", "fit", "value")
DEFAULT_WEIGHTS = {"ability": 0.4, "fit": 0.35, "value": 0.25}


def _normalise(weights: dict) -> dict:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return {k: v / total for k, v in weights.items()}


def ability_score(df: pd.DataFrame, role: Role) -> pd.Series:
    """Weighted mean of the role's metric percentiles, per player (0-100)."""
    w = _normalise(role.weights)
    pct_cols = {col: f"pct_{col}" for col in w}
    # missing metric -> treated as median (50) so one absent stat doesn't tank a player
    parts = []
    for col, weight in w.items():
        series = df.get(pct_cols[col])
        series = (series if series is not None else pd.Series(np.nan, index=df.index)).fillna(50.0)
        parts.append(series * weight)
    return sum(parts).clip(0, 100)


def fit_score(df: pd.DataFrame, role: Role) -> pd.Series:
    """Cosine similarity between each player's percentile vector (centred at the
    50th percentile) and the role's weight vector, rescaled from [-1, 1] to
    [0, 100]. Measures profile *shape*, not level."""
    w = _normalise(role.weights)
    cols = list(w)
    # player matrix, centred so "average everywhere" is the zero vector
    mat = np.column_stack([
        (df.get(f"pct_{c}", pd.Series(50.0, index=df.index)).fillna(50.0) - 50.0).to_numpy()
        for c in cols
    ])
    target = np.array([w[c] for c in cols])  # all positive: the role's emphasis
    target = target - target.mean()          # centre so it describes a shape
    tnorm = np.linalg.norm(target)
    pnorm = np.linalg.norm(mat, axis=1)
    denom = pnorm * tnorm
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(denom > 0, (mat @ target) / denom, 0.0)
    return pd.Series((cos + 1) / 2 * 100, index=df.index).clip(0, 100)


def value_score(df: pd.DataFrame, ability: pd.Series, budget_eur: float | None) -> pd.Series:
    """Affordability + ability-per-euro, 0-100.

    - affordability: 100 if value <= budget, tapering to 0 at 2x budget. If no
      budget is given, affordability is neutral (100) for everyone.
    - efficiency: ability divided by (log) market value, percentile-ranked
      across the candidate pool, so "most ability per euro" scores highest.
    Players with no value at all get a neutral 50.
    """
    value = df["market_value_eur"]
    has_value = value.notna() & (value > 0)

    if budget_eur and budget_eur > 0:
        ratio = value / budget_eur
        afford = np.clip(2 - ratio, 0, 1) * 100  # 1.0x -> 100, 2.0x -> 0
    else:
        afford = pd.Series(100.0, index=df.index)

    eff_raw = ability / np.log1p(value.where(has_value))
    efficiency = eff_raw.rank(pct=True) * 100
    efficiency = efficiency.fillna(50.0)

    score = 0.5 * pd.Series(afford, index=df.index) + 0.5 * efficiency
    score = score.where(has_value, 50.0)  # unknown value -> neutral
    return score.clip(0, 100)


def shortlist(
    df: pd.DataFrame,
    role: Role,
    budget_eur: float | None = None,
    weights: dict | None = None,
    exclude_club: str | None = None,
    max_age: int | None = None,
    top_n: int = 25,
) -> pd.DataFrame:
    """Rank candidates in the role's position group. Returns a tidy table with
    the three pillar scores, the overall score, and the key identity columns."""
    weights = _normalise(weights or DEFAULT_WEIGHTS)

    pool = df[df["pos_group"] == role.pos_group].copy()
    if exclude_club:
        pool = pool[pool["team"].str.casefold() != exclude_club.casefold()]
    if max_age:
        pool = pool[pool["age"] <= max_age]
    if pool.empty:
        return pool

    pool["ability"] = ability_score(pool, role)
    pool["fit"] = fit_score(pool, role)
    pool["value"] = value_score(pool, pool["ability"], budget_eur)
    pool["overall"] = sum(pool[p] * weights[p] for p in PILLARS)

    pool = pool.sort_values("overall", ascending=False)
    if top_n:
        pool = pool.head(top_n)
    return pool.reset_index(drop=True)


def profile_bars(row: pd.Series, role: Role) -> pd.DataFrame:
    """The FotMob-style per-metric view for one player: the role's metrics with
    the player's raw value and within-position percentile, for drawing bars."""
    rows = []
    for col in role.metric_columns():
        pct = row.get(f"pct_{col}")
        rows.append({
            "metric": METRIC_LABEL.get(col, col),
            "value": row.get(col),
            "percentile": pct,
        })
    return pd.DataFrame(rows).sort_values("percentile", ascending=False, na_position="last")
