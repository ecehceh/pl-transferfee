import numpy as np
import pandas as pd

from seller_effects import REFERENCE_TIER, estimate_tier_effects

TIERS = ["Premier League", REFERENCE_TIER, "Other covered league", "Outside covered leagues"]


# Large, low-noise sample so the checks below test the estimator, not luck:
# the standard error of each league effect is under 1 percentage point.
def synthetic(pl_effect_log: float, n: int = 4000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "seller_tier": rng.choice(TIERS, n),
        "log_mv": rng.normal(16.5, 1.0, n),
        "age": rng.uniform(18, 32, n),
        "position": rng.choice(["Attack", "Midfield", "Defender", "Goalkeeper"], n),
        "season": rng.choice(range(2020, 2026), n),
        "window": rng.choice(["summer", "winter"], n),
        "log_buyer_squad_value": rng.normal(19.5, 0.6, n),
        "buyer_big_six": rng.integers(0, 2, n),
        "log_seller_squad_value": rng.normal(18.5, 1.0, n),
        "seller_relegated": rng.integers(0, 2, n),
        "to_club_id": rng.integers(0, 40, n),  # 40 buying clubs to cluster on
    })
    premium = 0.15 + pl_effect_log * (df["seller_tier"] == "Premier League") + rng.normal(0, 0.15, n)
    df["log_fee"] = df["log_mv"] + premium
    return df


def test_recovers_a_planted_domestic_premium():
    out = estimate_tier_effects(synthetic(pl_effect_log=0.2))
    pl = next(e for e in out["specs"]["player_and_buyer"]["effects"] if e["tier"] == "Premier League")
    assert abs(pl["effect"] - np.expm1(0.2)) < 0.03
    assert pl["low"] > 0


def test_no_premium_gives_effects_near_zero():
    out = estimate_tier_effects(synthetic(pl_effect_log=0.0))
    for spec in out["specs"].values():
        for e in spec["effects"]:
            assert abs(e["effect"]) < 0.03, e
            assert e["low"] < e["effect"] < e["high"]


def test_output_shape_and_reference_tier_excluded():
    out = estimate_tier_effects(synthetic(pl_effect_log=0.1))
    assert out["reference"] == REFERENCE_TIER
    assert set(out["specs"]) == {"player_and_buyer", "plus_seller_wealth"}
    for spec in out["specs"].values():
        tiers = {e["tier"] for e in spec["effects"]}
        assert REFERENCE_TIER not in tiers and len(tiers) == 3
        assert spec["n"] == 4000
        assert spec["clusters"] == 40
