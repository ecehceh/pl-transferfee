"""
Is there a domestic premium? Do PL clubs pay more when they buy from another
PL club than for a comparable player from elsewhere?

The raw fee/value ratio by selling league can't answer that: PL sellers sell
different players (older, pricier, different positions) at different times.
So this fits a plain linear regression of the premium, log(fee / market
value), on the selling league plus controls, and reads off the league terms.
Each term is "how much more (or less) PL clubs paid, relative to buying a
similar player from another top-5 league". Standard errors are
heteroskedasticity-robust (HC1).

Two specifications, because "similar" is a choice:
  player_and_buyer    player value, age, position, season, window, buyer size
  plus_seller_wealth  ... and the selling club's squad value. PL clubs are
                      rich, so this asks whether the premium is about the
                      league itself or just about selling to/from a rich club.

A linear model is used on purpose: one coefficient with a confidence interval
is easier to explain than an average of tree contributions.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

REFERENCE_TIER = "Other top-5 league"
CONFIDENCE = 0.95

BASE_CONTROLS = ["log_mv", "age", "log_buyer_squad_value", "buyer_big_six"]
CATEGORICAL_CONTROLS = ["position", "season", "window"]
SPECS = {
    "player_and_buyer": BASE_CONTROLS,
    "plus_seller_wealth": BASE_CONTROLS + ["log_seller_squad_value"],
}


def _design(df: pd.DataFrame, numeric: list) -> pd.DataFrame:
    tiers = pd.get_dummies(df["seller_tier"], dtype=float).drop(columns=REFERENCE_TIER)
    tiers.columns = [f"tier::{c}" for c in tiers.columns]
    X = pd.concat([
        tiers,
        df[numeric].astype(float),
        (df["age"] ** 2).rename("age_sq"),
        pd.get_dummies(df[CATEGORICAL_CONTROLS].astype(str), drop_first=True, dtype=float),
    ], axis=1)
    return sm.add_constant(X)


def estimate_tier_effects(df: pd.DataFrame) -> dict:
    """Selling-league effects on fee / market value, as fractions (0.1 = +10%)."""
    data = df.copy()
    data["log_fee_to_value"] = data["log_fee"] - data["log_mv"]
    out = {"reference": REFERENCE_TIER, "confidence": CONFIDENCE, "specs": {}}

    for name, numeric in SPECS.items():
        needed = ["log_fee_to_value", "seller_tier"] + numeric + CATEGORICAL_CONTROLS
        d = data.dropna(subset=needed)
        fit = sm.OLS(d["log_fee_to_value"], _design(d, numeric)).fit(cov_type="HC1")
        ci = fit.conf_int(alpha=1 - CONFIDENCE)
        effects = []
        for term in [c for c in fit.params.index if c.startswith("tier::")]:
            effects.append({
                "tier": term.removeprefix("tier::"),
                "deals": int((d["seller_tier"] == term.removeprefix("tier::")).sum()),
                "effect": float(np.expm1(fit.params[term])),
                "low": float(np.expm1(ci.loc[term, 0])),
                "high": float(np.expm1(ci.loc[term, 1])),
                "p_value": float(fit.pvalues[term]),
            })
        out["specs"][name] = {
            "n": int(fit.nobs),
            "reference_deals": int((d["seller_tier"] == REFERENCE_TIER).sum()),
            "r2": float(fit.rsquared),
            "effects": effects,
        }
    return out
