import numpy as np
import pandas as pd

from analysis import bootstrap_median_ci, group_premiums


def make_deals(club, residuals):
    return pd.DataFrame({
        "to_club_name": club,
        "residual_log": residuals,
        "transfer_fee": 10e6,
        "fee_to_value": 1.0,
    })


def test_ci_contains_the_median_and_is_ordered():
    values = np.random.default_rng(1).normal(0.2, 0.3, 40)
    low, high = bootstrap_median_ci(values)
    assert low < np.median(values) < high


def test_ci_is_reproducible():
    values = [0.1, -0.2, 0.4, 0.0, 0.3]
    assert bootstrap_median_ci(values) == bootstrap_median_ci(values)


def test_ci_ignores_nan_and_handles_empty():
    assert bootstrap_median_ci([np.nan, 0.5, 0.5]) == (0.5, 0.5)
    low, high = bootstrap_median_ci([])
    assert np.isnan(low) and np.isnan(high)


def test_group_premiums_flags_consistent_overpayer_as_clear():
    df = pd.concat([
        make_deals("Overpayer", np.full(12, 0.3) + np.linspace(-0.05, 0.05, 12)),
        make_deals("Noisy", np.array([0.9, -0.8, 0.7, -0.9, 0.8, -0.7])),
    ])
    out = group_premiums(df, "to_club_name", min_deals=3).set_index("to_club_name")
    assert out.loc["Overpayer", "is_clear"]
    assert out.loc["Overpayer", "premium"] > 0.3  # exp(0.3) - 1 ~ +35%
    assert not out.loc["Noisy", "is_clear"]


def test_group_premiums_applies_min_deals_and_keeps_columns_when_empty():
    df = make_deals("Tiny", [0.1, 0.2])
    out = group_premiums(df, "to_club_name", min_deals=3)
    assert out.empty
    assert {"to_club_name", "premium", "premium_low", "premium_high", "is_clear"} <= set(out.columns)


def test_group_premiums_supports_multiple_group_columns():
    df = make_deals("A", [0.1, 0.2, 0.3]).assign(seller_tier="Premier League")
    out = group_premiums(df, ["to_club_name", "seller_tier"], min_deals=3)
    assert list(out[["to_club_name", "seller_tier"]].iloc[0]) == ["A", "Premier League"]
