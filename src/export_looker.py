"""
Export the model's results as tidy CSVs for a Looker Studio report.

Looker Studio only displays tables -- it can't run Python -- so everything the
Streamlit app computes on the fly is computed here instead:

  deals.csv           one row per deal: fee, market value, model estimate,
                      and the fields the report filters on
  contributions.csv   what pushed each deal's expected fee up or down, one row
                      per deal x factor, plus one row per deal x factor group
                      (level = "Factor" / "Group")
  club_premiums.csv   per buying and selling club: median premium vs. the model
                      with a 90% bootstrap range. Ranges can't be recomputed
                      when a report filter changes, so they come in fixed
                      selections: "All deals" and "Excluding under-21s"
  league_effects.csv  the domestic-premium regression: one row per selling
                      league and specification, reference league included at 0
  headline.csv        one row of headline numbers for scorecards

Conventions, so Looker Studio picks sensible field types:
  - percentages are fractions (0.17 = +17%): set the field type to Percent
  - money is in euros: set the field type to Currency (EUR)
  - dates are ISO (YYYY-MM-DD); yes/no flags are "Yes"/"No" text

Run after train_model.py:  python src/export_looker.py   -> exports/*.csv
See docs/looker_studio.md for building the report.
"""

import json
import sys

import numpy as np
import pandas as pd

# `python src/export_looker.py` puts src/ on the import path.
from analysis import CI_LEVEL, YOUNG_AGE, add_deal_labels, group_premiums
from config import BASE_DIR, MODELS_DIR
from labels import GROUP_OF, GROUPS, LABELS

EXPORT_DIR = BASE_DIR / "exports"
# same defaults as the app's "minimum deals per club" sliders
MIN_DEALS = {"Buying": 6, "Selling": 4}
SELECTIONS = {
    "All deals": lambda d: d,
    f"Excluding under-{YOUNG_AGE}s": lambda d: d[d["age"] >= YOUNG_AGE],
}
SPEC_LABELS = {
    "player_and_buyer": "Player, timing & buyer controls",
    "plus_seller_wealth": "+ selling club's squad value",
}


def _yes_no(values) -> pd.Series:
    """1/0 flags -> "Yes"/"No"; missing -> "Unknown"."""
    return pd.Series(values).map({1: "Yes", 0: "No"}).fillna("Unknown")


def _optional(df: pd.DataFrame, col: str) -> pd.Series:
    """A column that older model outputs may not have yet (e.g. seller_relegated)."""
    return df[col] if col in df.columns else pd.Series(np.nan, index=df.index)


def _verdict(is_clear: pd.Series, premium: pd.Series) -> np.ndarray:
    return np.select(
        [~is_clear.astype(bool), premium > 0],
        ["Not clearly different", "Paid more than expected"],
        "Paid less than expected",
    )


def _range_text(low: pd.Series, high: pd.Series) -> list:
    return [f"{lo * 100:+.0f}% to {hi * 100:+.0f}%" for lo, hi in zip(low, high)]


# --------------------------------------------------------------------------- tables
def deals_table(scored: pd.DataFrame) -> pd.DataFrame:
    d = add_deal_labels(scored).reset_index(drop=True)
    return pd.DataFrame({
        "transfer_id": d["transfer_id"],
        "deal": d["deal"],
        "player": d["player_name"],
        "buying_club": d["to_club_name"],
        "selling_club": d["from_club_name"],
        "selling_league": d["seller_tier"],
        "season": d["season_label"],
        "window": d["window"].str.capitalize(),
        "transfer_date": pd.to_datetime(d["transfer_date"]).dt.strftime("%Y-%m-%d"),
        "position": d["position"],
        "detailed_position": d["sub_position"].fillna(d["position"]),
        "age": d["age"].round(1),
        f"under_{YOUNG_AGE}": _yes_no((d["age"] < YOUNG_AGE).astype(int)),
        "buyer_big_six": _yes_no(d["buyer_big_six"]),
        "buyer_promoted": _yes_no(d["buyer_promoted"]),
        "seller_relegated": _yes_no(_optional(d, "seller_relegated")),
        "seller_is_academy": _yes_no(_optional(d, "seller_is_academy")),
        "deadline_week": _yes_no(_optional(d, "deadline_week")),
        "fee_eur": d["transfer_fee"].round(0),
        "market_value_eur": d["mv_at_transfer"].round(0),
        "expected_fee_eur": d["expected_fee"].round(0),
        "fee_to_value": d["fee_to_value"].round(3),
        "above_market_value": _yes_no((d["fee_to_value"] > 1).astype(int)),
        "premium_vs_model": d["premium_vs_model"].round(4),
        "paid_vs_model": np.where(d["premium_vs_model"] > 0, "Above model", "Below model"),
    })


def contributions_table(scored: pd.DataFrame, feats: dict, deals: pd.DataFrame) -> pd.DataFrame:
    """Per-deal feature contributions (log scale, premium over market value).
    Group rows sum their members first: contributions add up, so that is the
    group's effect on the deal -- averaging absolute values per factor and
    adding them would overstate groups of overlapping features."""
    feat_cols = [c for c in feats["numeric"] + feats["categorical"] if f"shap_{c}" in scored.columns]
    shap = scored[[f"shap_{c}" for c in feat_cols]].set_axis(feat_cols, axis=1).reset_index(drop=True)
    shap.insert(0, "transfer_id", scored["transfer_id"].to_numpy())

    factors = shap.melt(id_vars="transfer_id", var_name="feature", value_name="effect_log")
    factors["level"] = "Factor"
    factors["factor"] = factors["feature"].map(LABELS).fillna(factors["feature"])
    factors["factor_group"] = factors["feature"].map(GROUP_OF).fillna("Other")

    groups = []
    for group, members in GROUPS.items():
        cols = [c for c in members if c in feat_cols]
        if cols:
            groups.append(pd.DataFrame({
                "transfer_id": shap["transfer_id"], "feature": group, "effect_log": shap[cols].sum(axis=1),
                "level": "Group", "factor": group, "factor_group": group,
            }))
    out = pd.concat([factors, *groups], ignore_index=True)

    out["effect_pct"] = np.expm1(out["effect_log"]).round(4)
    out["abs_effect_log"] = out["effect_log"].abs().round(4)
    out["effect_log"] = out["effect_log"].round(4)
    # repeat the filter fields so this one table can drive its charts on its own
    dims = deals[["transfer_id", "deal", "season", "position", f"under_{YOUNG_AGE}", "buying_club"]]
    out = out.merge(dims, on="transfer_id", how="left")
    return out[["transfer_id", "deal", "season", "position", f"under_{YOUNG_AGE}", "buying_club",
                "level", "factor", "factor_group", "feature", "effect_log", "effect_pct", "abs_effect_log"]]


def club_premiums_table(scored: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for selection, pick in SELECTIONS.items():
        sub = pick(scored)
        for role, cols in [("Buying", ["to_club_name"]), ("Selling", ["from_club_name", "seller_tier"])]:
            g = group_premiums(sub, cols, MIN_DEALS[role])
            g = g.rename(columns={"to_club_name": "club", "from_club_name": "club", "seller_tier": "league"})
            if "league" not in g.columns:
                g["league"] = "Premier League"  # every buyer is a PL club
            g.insert(0, "selection", selection)
            g.insert(0, "role", role)
            g["min_deals"] = MIN_DEALS[role]
            parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["verdict"] = _verdict(out["is_clear"], out["premium"])
    out["clear"] = _yes_no(out["is_clear"].astype(int))
    out["likely_range"] = _range_text(out["premium_low"], out["premium_high"])
    out = out.rename(columns={"total_fee": "total_fee_eur", "median_ratio": "median_fee_to_value"})
    for c in ["premium", "premium_low", "premium_high"]:
        out[c] = out[c].round(4)
    out["median_fee_to_value"] = out["median_fee_to_value"].round(3)
    return out[["role", "selection", "club", "league", "deals", "min_deals", "total_fee_eur",
                "median_fee_to_value", "premium", "premium_low", "premium_high", "likely_range",
                "clear", "verdict"]]


def league_effects_table(tier_effects: dict) -> pd.DataFrame:
    ref = tier_effects["reference"]
    rows = []
    for name, spec in tier_effects["specs"].items():
        label = SPEC_LABELS.get(name, name)
        rows.append({"specification": label, "selling_league": ref, "deals": spec["reference_deals"],
                     "effect": 0.0, "low": 0.0, "high": 0.0, "p_value": np.nan, "is_reference": "Yes"})
        for e in spec["effects"]:
            rows.append({"specification": label, "selling_league": e["tier"], "deals": e["deals"],
                         "effect": e["effect"], "low": e["low"], "high": e["high"],
                         "p_value": e["p_value"], "is_reference": "No"})
    out = pd.DataFrame(rows)
    clear = (out["low"] > 0) | (out["high"] < 0)
    out["clear"] = np.where(out["is_reference"] == "Yes", "Reference", _yes_no(clear.astype(int)))
    out["likely_range"] = np.where(out["is_reference"] == "Yes", "Reference",
                                   _range_text(out["low"], out["high"]))
    for c in ["effect", "low", "high", "p_value"]:
        out[c] = out[c].round(4)
    return out


def headline_table(metrics: dict, deals: pd.DataFrame, clubs: pd.DataFrame, tier_effects: dict) -> pd.DataFrame:
    def pl_effect(spec):
        return next(e for e in tier_effects["specs"][spec]["effects"] if e["tier"] == "Premier League")

    base, full = metrics["baseline_market_value_only"], metrics["full_model"]
    pl, pl_rich = pl_effect("player_and_buyer"), pl_effect("plus_seller_wealth")
    buyers = clubs[(clubs["role"] == "Buying") & (clubs["selection"] == "All deals")]
    seasons = sorted(deals["season"].unique())
    row = {
        "deals": len(deals),
        "first_season": seasons[0],
        "last_season": seasons[-1],
        "total_spent_eur": deals["fee_eur"].sum(),
        "median_fee_to_value": float(deals["fee_to_value"].median()),
        "share_above_market_value": float((deals["fee_to_value"] > 1).mean()),
        "r2_market_value_only": base["r2"],
        "r2_full_model": full["r2"],
        "r2_premium_explained": full["r2_premium"],
        "typical_error_market_value_only": base["typical_error_pct"] / 100,
        "typical_error_full_model": full["typical_error_pct"] / 100,
        "domestic_premium": pl["effect"],
        "domestic_premium_low": pl["low"],
        "domestic_premium_high": pl["high"],
        "domestic_premium_with_seller_wealth": pl_rich["effect"],
        "domestic_premium_with_seller_wealth_low": pl_rich["low"],
        "domestic_premium_with_seller_wealth_high": pl_rich["high"],
        "confidence_level": tier_effects["confidence"],
        "buying_clubs_shown": len(buyers),
        "buying_clubs_clear": int((buyers["clear"] == "Yes").sum()),
        # with a 90% range, ~10% of clubs look "clear" even if all pay the same
        "buying_clubs_clear_by_chance": round((1 - CI_LEVEL) * len(buyers), 1),
    }
    return pd.DataFrame([row]).round(4)


# --------------------------------------------------------------------------- main
def build_exports(scored: pd.DataFrame, metrics: dict, feats: dict, tier_effects: dict) -> dict:
    deals = deals_table(scored)
    clubs = club_premiums_table(scored)
    return {
        "deals": deals,
        "contributions": contributions_table(scored, feats, deals),
        "club_premiums": clubs,
        "league_effects": league_effects_table(tier_effects),
        "headline": headline_table(metrics, deals, clubs, tier_effects),
    }


def main():
    try:
        scored = pd.read_parquet(MODELS_DIR / "transfers_scored.parquet")
        metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
        feats = json.loads((MODELS_DIR / "features.json").read_text())
        tier_effects = json.loads((MODELS_DIR / "seller_tier_effects.json").read_text())
    except FileNotFoundError as e:
        sys.exit(f"{e}\nNo trained model found. Run src/data_prep.py, then src/train_model.py.")

    EXPORT_DIR.mkdir(exist_ok=True)
    for name, table in build_exports(scored, metrics, feats, tier_effects).items():
        path = EXPORT_DIR / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8")
        print(f"  {path.name:<20} {len(table):>7,} rows  {table.shape[1]:>3} columns")
    print(f"Wrote {EXPORT_DIR}. Next: docs/looker_studio.md")


if __name__ == "__main__":
    main()
