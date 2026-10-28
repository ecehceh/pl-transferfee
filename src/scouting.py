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
    deflate_redundancy: bool = True,
) -> pd.DataFrame:
    """Rank candidates in the role's position group. Returns a tidy table with
    the three pillar scores, the overall score, and the key identity columns.

    `deflate_redundancy` (on by default) rebalances the role's metric weights
    so metrics that measure near the same thing (e.g. a playmaker's expected
    assists and big chances created) don't get counted several times. See
    role_validation.effective_weights.
    """
    weights = _normalise(weights or DEFAULT_WEIGHTS)

    if deflate_redundancy:
        # local import avoids a hard dependency for callers that pass weights
        from role_validation import effective_weights
        role = role.with_weights(effective_weights(df, role))

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
    pool = pool.reset_index(drop=True)
    pool["why"] = [explain(row, role, budget_eur) for _, row in pool.iterrows()]
    return pool


def explain(row: pd.Series, role: Role, budget_eur: float | None = None, n_strengths: int = 2) -> str:
    """A one-line reason this player made the shortlist: their standout role
    metrics, a value note, and a short-contract flag. Keeps the shortlist from
    being an unexplained leaderboard."""
    # top strengths = highest-percentile role metrics
    ranked = sorted(
        ((METRIC_LABEL.get(c, c), row.get(f"pct_{c}")) for c in role.metric_columns()),
        key=lambda kv: (kv[1] if pd.notna(kv[1]) else -1),
        reverse=True,
    )
    strong = [f"{label.lower()} (top {100 - pct:.0f}%)" for label, pct in ranked[:n_strengths]
              if pd.notna(pct) and pct >= 60]
    bits = []
    if strong:
        bits.append("strong on " + ", ".join(strong))

    value = row.get("market_value_eur")
    if pd.notna(value) and budget_eur and budget_eur > 0:
        if value <= budget_eur:
            bits.append(f"€{(budget_eur - value) / 1e6:.0f}m under budget")
        else:
            bits.append(f"€{(value - budget_eur) / 1e6:.0f}m over budget")
    if row.get("value", 0) >= 70 and "budget" not in " ".join(bits):
        bits.append("good value for the ability")

    years = row.get("contract_years_left")
    if pd.notna(years) and years <= 1:
        bits.append("contract nearly up — could come cheaper")

    return "; ".join(bits).capitalize() if bits else "Rounded profile for the role."


def format_metric(label: str, value) -> str:
    """Readable raw value for a metric, so the bars show the number a scout
    cares about next to the percentile (as FotMob does)."""
    if pd.isna(value):
        return "–"
    if "%" in label:
        return f"{value:.0f}%"
    if "km/h" in label:
        return f"{value:.1f}"
    if "rating" in label.lower():
        return f"{value:.2f}"
    if "p90" in label or "per 90" in label.lower():
        return f"{value:.2f}"
    return f"{value:.2f}"


def profile_bars(row: pd.Series, role: Role) -> pd.DataFrame:
    """The FotMob-style per-metric view for one player: the role's metrics with
    the player's raw value (formatted) and within-position percentile."""
    rows = []
    for col in role.metric_columns():
        label = METRIC_LABEL.get(col, col)
        value = row.get(col)
        rows.append({
            "metric": label,
            "value": value,
            "value_label": format_metric(label, value),
            "percentile": row.get(f"pct_{col}"),
        })
    return pd.DataFrame(rows).sort_values("percentile", ascending=False, na_position="last")
