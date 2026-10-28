import numpy as np
import pandas as pd

from role_validation import (
    effective_weights,
    redundancy_report,
    weight_stability,
)
from roles import Role
from scouting import ability_score

ROLE = Role(
    name="Test",
    pos_group="FW",
    description="",
    weights={"np_xg_p90": 2, "shots_p90": 2, "xa_p90": 1},
)


def make_pool(n=200, seed=0):
    """A pool where np_xg and shots are near-duplicates (should be deflated)
    and xa is independent (should be left alone or boosted)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    xg = base + rng.normal(0, 0.1, n)           # np_xg
    shots = base + rng.normal(0, 0.1, n)        # ~ duplicate of np_xg
    xa = rng.normal(0, 1, n)                     # independent
    df = pd.DataFrame({
        "pos_group": "FW",
        "np_xg_p90": xg, "shots_p90": shots, "xa_p90": xa,
    })
    # percentiles the validation works on
    for c in ["np_xg_p90", "shots_p90", "xa_p90"]:
        df[f"pct_{c}"] = df[c].rank(pct=True) * 100
    return df


def test_redundancy_report_flags_the_duplicate_pair():
    rep = redundancy_report(make_pool(), ROLE)
    top = rep.iloc[0]
    assert {top["metric_a"], top["metric_b"]} == {"np_xg_p90", "shots_p90"}
    assert top["redundant"]
    # the independent pair with xa should not be flagged
    assert not rep[rep.apply(lambda r: "xa_p90" in (r["metric_a"], r["metric_b"]), axis=1)]["redundant"].any()


def test_effective_weights_shrink_redundant_and_keep_total():
    df = make_pool()
    eff = effective_weights(df, ROLE)
    # total weight preserved
    assert np.isclose(sum(eff.values()), sum(ROLE.weights.values()))
    # the independent metric gains share relative to the duplicated ones
    assert eff["xa_p90"] > ROLE.weights["xa_p90"]
    assert eff["np_xg_p90"] < ROLE.weights["np_xg_p90"]


def test_effective_weights_barely_touch_an_uncorrelated_role():
    # three independent metrics -> weights should stay close to raw
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({
        "pos_group": ["FW"] * n,
        "pct_np_xg_p90": rng.uniform(0, 100, n),
        "pct_shots_p90": rng.uniform(0, 100, n),
        "pct_xa_p90": rng.uniform(0, 100, n),
    })
    eff = effective_weights(df, ROLE)
    for m in ROLE.weights:
        assert abs(eff[m] - ROLE.weights[m]) < 0.6  # small nudge only


def test_weight_stability_reports_high_agreement_for_stable_role():
    df = make_pool(400)
    stab = weight_stability(
        df, ROLE, lambda pool, w: ability_score(pool, ROLE.with_weights(w)),
        n_trials=20, top_n=20,
    )
    assert 0 <= stab["mean_top_n_overlap"] <= 1
    assert stab["mean_rank_corr"] > 0.8  # jitter shouldn't scramble the order
    assert stab["n_trials"] == 20
