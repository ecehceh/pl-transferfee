import numpy as np
import pandas as pd
import pytest

from roles import METRIC_DIRECTION, ROLES, Role, role_choices
from scouting import (
    DEFAULT_WEIGHTS,
    ability_score,
    fit_score,
    profile_bars,
    shortlist,
    value_score,
)

# a tiny two-metric role for arithmetic we can check by hand
TOY = Role(
    name="Toy",
    pos_group="FW",
    description="",
    weights={"np_xg_p90": 3, "xa_p90": 1},
)


def make_pool(n=6, pos_group="FW"):
    """A small candidate pool with controllable percentiles and values."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "player": [f"P{i}" for i in range(n)],
        "team": ["Club"] * n,
        "pos_group": pos_group,
        "age": rng.integers(19, 30, n),
        "market_value_eur": np.linspace(5e6, 100e6, n),
        "np_xg_p90": rng.random(n),
        "xa_p90": rng.random(n),
        "pct_np_xg_p90": np.linspace(10, 90, n),
        "pct_xa_p90": np.linspace(90, 10, n),
    })


# --- roles -------------------------------------------------------------------
def test_every_role_metric_is_in_the_vocabulary():
    for role in ROLES.values():
        for col in role.weights:
            assert col in METRIC_DIRECTION, f"{role.name}: {col} not a known metric"


def test_role_rejects_unknown_metric():
    from roles import _role
    with pytest.raises(ValueError):
        _role("Bad", "FW", "", {"not_a_metric": 1})


def test_role_choices_cover_all_roles():
    assert len(role_choices()) == len(ROLES)


# --- ability -----------------------------------------------------------------
def test_ability_is_weighted_mean_of_percentiles():
    pool = make_pool()
    # weights 3:1 on np_xg (pct 10..90) and xa (pct 90..10)
    ability = ability_score(pool, TOY)
    expected = 0.75 * pool["pct_np_xg_p90"] + 0.25 * pool["pct_xa_p90"]
    assert np.allclose(ability, expected)


def test_ability_treats_missing_metric_as_median():
    pool = make_pool().drop(columns=["pct_xa_p90"])
    ability = ability_score(pool, TOY)
    expected = 0.75 * pool["pct_np_xg_p90"] + 0.25 * 50.0
    assert np.allclose(ability, expected)


# --- fit ---------------------------------------------------------------------
def test_fit_is_high_when_profile_matches_role_shape():
    # one player strong exactly where the role is weighted, weak elsewhere
    pool = pd.DataFrame({
        "pct_np_xg_p90": [95, 50, 5],   # role weights this heavily
        "pct_xa_p90": [5, 50, 95],      # role weights this lightly
    })
    fit = fit_score(pool, TOY)
    assert fit.iloc[0] > fit.iloc[1] > fit.iloc[2]


def test_fit_is_bounded_0_100():
    pool = make_pool(8)
    fit = fit_score(pool, TOY)
    assert fit.between(0, 100).all()


# --- value -------------------------------------------------------------------
def test_value_affordability_penalises_over_budget():
    pool = make_pool()
    ability = ability_score(pool, TOY)
    # budget below the most expensive players: they must score lower on value
    v = value_score(pool, ability, budget_eur=20e6)
    cheap = v[pool["market_value_eur"] <= 20e6].mean()
    dear = v[pool["market_value_eur"] >= 80e6].mean()
    assert cheap > dear


def test_value_neutral_without_budget_or_value():
    pool = make_pool(4)
    pool["market_value_eur"] = np.nan
    ability = ability_score(pool, TOY)
    v = value_score(pool, ability, budget_eur=None)
    assert np.allclose(v, 50.0)


# --- shortlist ---------------------------------------------------------------
def test_shortlist_ranks_by_overall_and_respects_filters():
    pool = make_pool(10)
    pool.loc[0, "age"] = 35  # should be filtered by max_age
    pool.loc[1, "team"] = "Exclude FC"
    out = shortlist(
        pool, TOY, budget_eur=100e6, exclude_club="Exclude FC", max_age=30, top_n=5
    )
    assert len(out) == 5
    assert "Exclude FC" not in out["team"].values
    assert (out["age"] <= 30).all()
    assert out["overall"].is_monotonic_decreasing


def test_shortlist_only_returns_the_roles_position_group():
    pool = pd.concat([make_pool(4, "FW"), make_pool(4, "DF")], ignore_index=True)
    out = shortlist(pool, TOY, top_n=20)
    assert (out["pos_group"] == "FW").all()


def test_shortlist_overall_is_weighted_blend():
    pool = make_pool(5)
    out = shortlist(pool, TOY, budget_eur=100e6, top_n=5)
    blended = sum(out[p] * DEFAULT_WEIGHTS[p] for p in ("ability", "fit", "value"))
    assert np.allclose(out["overall"], blended)


def test_shortlist_empty_pool_returns_empty():
    pool = make_pool(4, "FW")
    out = shortlist(pool, Role("X", "GK", "", {"np_xg_p90": 1}))
    assert out.empty


# --- profile bars ------------------------------------------------------------
def test_profile_bars_lists_role_metrics_sorted_by_percentile():
    pool = make_pool()
    bars = profile_bars(pool.iloc[0], TOY)
    assert set(bars["metric"]) == {"Non-penalty xG p90", "xA p90"}
    pcts = bars["percentile"].dropna()
    assert list(pcts) == sorted(pcts, reverse=True)
