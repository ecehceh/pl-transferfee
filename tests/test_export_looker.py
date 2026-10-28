import numpy as np
import pandas as pd
import pytest

from export_looker import build_exports

NUMERIC = ["log_mv", "age", "seller_relegated", "log_buyer_squad_value", "minutes_365d", "apps_365d"]
CATEGORICAL = ["position", "seller_tier"]
FEATS = {"numeric": NUMERIC, "categorical": CATEGORICAL}


def scored(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """A small stand-in for models/transfers_scored.parquet."""
    rng = np.random.default_rng(seed)
    mv = rng.uniform(1e6, 60e6, n)
    resid = rng.normal(0, 0.3, n)
    df = pd.DataFrame({
        "transfer_id": np.arange(n),
        "player_name": [f"Player {i}" for i in range(n)],
        "to_club_name": rng.choice(["Arsenal", "Fulham", "Brentford"], n),
        "from_club_name": rng.choice(["Benfica", "Brighton", "Lyon", "Ajax"], n),
        "seller_tier": rng.choice(["Premier League", "Other top-5 league", "Other covered league"], n),
        "season": rng.choice([2023, 2024], n),
        "window": rng.choice(["summer", "winter"], n),
        "transfer_date": pd.Timestamp("2024-07-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D"),
        "position": rng.choice(["Attack", "Defender"], n),
        "sub_position": rng.choice(["Centre-Forward", None], n),
        "age": rng.uniform(17, 33, n),
        "buyer_big_six": rng.integers(0, 2, n),
        "buyer_promoted": rng.integers(0, 2, n),
        "seller_relegated": rng.integers(0, 2, n),
        "seller_is_academy": 0,
        "deadline_week": rng.choice([0.0, 1.0, np.nan], n),
        "mv_at_transfer": mv,
        "residual_log": resid,
    })
    df["expected_fee"] = mv * 1.1
    df["transfer_fee"] = df["expected_fee"] * np.exp(resid)
    df["fee_to_value"] = df["transfer_fee"] / mv
    df["premium_vs_model"] = np.expm1(resid)
    for c in NUMERIC + CATEGORICAL:
        df[f"shap_{c}"] = rng.normal(0, 0.1, n)
    return df


METRICS = {
    "n_transfers": 120,
    "baseline_market_value_only": {"r2": 0.64, "mae_log": 0.4, "typical_error_pct": 49.2},
    "full_model": {"r2": 0.74, "mae_log": 0.33, "r2_premium": 0.42, "typical_error_pct": 39.1},
}
TIER_EFFECTS = {
    "reference": "Other top-5 league",
    "confidence": 0.95,
    "specs": {
        name: {
            "n": 120, "reference_deals": 40, "r2": 0.3,
            "effects": [
                {"tier": "Premier League", "deals": 40, "effect": 0.17, "low": 0.06, "high": 0.28, "p_value": 0.01},
                {"tier": "Other covered league", "deals": 40, "effect": -0.05, "low": -0.2, "high": 0.1,
                 "p_value": 0.5},
            ],
        }
        for name in ["player_and_buyer", "plus_seller_wealth"]
    },
}


@pytest.fixture(scope="module")
def exports():
    return build_exports(scored(), METRICS, FEATS, TIER_EFFECTS)


def test_deals_one_row_per_deal_with_readable_fields(exports):
    deals = exports["deals"]
    assert len(deals) == 120 and deals["transfer_id"].is_unique and deals["deal"].is_unique
    assert set(deals["under_21"]) <= {"Yes", "No"}
    assert set(deals["deadline_week"]) == {"Yes", "No", "Unknown"}
    assert deals["transfer_date"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    assert set(deals["season"]) == {"2023/24", "2024/25"}
    assert deals["detailed_position"].notna().all()  # falls back to position


def test_contributions_has_every_factor_and_group_sums_its_members(exports):
    c = exports["contributions"]
    factors, groups = c[c["level"] == "Factor"], c[c["level"] == "Group"]
    assert len(factors) == 120 * len(NUMERIC + CATEGORICAL)
    # Player = log_mv, age, minutes, apps, position; its group row is their sum
    deal0 = factors[factors["transfer_id"] == 0]
    player_sum = deal0.loc[deal0["factor_group"] == "Player", "effect_log"].sum()
    player_row = groups[(groups["transfer_id"] == 0) & (groups["factor"] == "Player")]["effect_log"].item()
    assert player_row == pytest.approx(player_sum, abs=1e-3)
    assert c["deal"].notna().all() and c["season"].notna().all()


def test_club_premiums_cover_both_roles_and_selections(exports):
    clubs = exports["club_premiums"]
    assert set(clubs["role"]) == {"Buying", "Selling"}
    assert set(clubs["selection"]) == {"All deals", "Excluding under-21s"}
    assert (clubs.loc[clubs["role"] == "Buying", "league"] == "Premier League").all()
    assert (clubs["premium_low"] <= clubs["premium"]).all() and (clubs["premium"] <= clubs["premium_high"]).all()
    assert set(clubs["verdict"]) <= {"Not clearly different", "Paid more than expected", "Paid less than expected"}
    all_deals = clubs[clubs["selection"] == "All deals"]
    young_excluded = clubs[clubs["selection"] == "Excluding under-21s"]
    assert all_deals["deals"].sum() > young_excluded["deals"].sum()


def test_league_effects_include_reference_at_zero(exports):
    e = exports["league_effects"]
    ref = e[e["is_reference"] == "Yes"]
    assert len(ref) == 2 and (ref["effect"] == 0).all() and (ref["selling_league"] == "Other top-5 league").all()
    pl = e[(e["selling_league"] == "Premier League")]
    assert (pl["clear"] == "Yes").all()
    assert pl["likely_range"].iloc[0] == "+6% to +28%"


def test_headline_is_one_row_with_fractions(exports):
    h = exports["headline"]
    assert len(h) == 1
    row = h.iloc[0]
    assert row["deals"] == 120
    assert row["domestic_premium"] == pytest.approx(0.17)
    assert row["typical_error_full_model"] == pytest.approx(0.391)
    assert row["buying_clubs_clear_by_chance"] == pytest.approx(0.3)  # 10% of 3 clubs
