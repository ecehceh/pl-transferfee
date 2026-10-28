"""
Model the fee a Premier League club paid, then measure who paid more or less
than expected.

Steps
  1. Baseline: how much of the fee does market value alone explain?
     (linear regression of log fee on log market value)
  2. Full model: XGBoost on player + seller + buyer-context + timing features.
     It predicts the fee RELATIVE to market value (log fee - log value) and
     adds market value back. Trees can't predict outside the range they were
     trained on, so a model of the fee itself topped out around EUR 90m and
     every record deal looked like an overpay; a premium has no such ceiling.
     The buying club's identity is deliberately NOT a feature, so whatever is
     left over (the residual) is the club's own premium or discount.
  3. Out-of-fold predictions (5-fold CV) so every transfer is scored by a model
     that never saw it. Residuals from in-sample predictions would be too small.
     Folds are grouped by player, so a player who moved twice is never in the
     training and test side of the same fold.
  4. Per-feature contributions (XGBoost's built-in SHAP values) from a model
     fit on all rows, to explain what pushes a fee up or down.

  5. Domestic premium: a separate linear regression of the premium on the
     selling league plus controls (see seller_effects.py).

Needs scikit-learn >= 1.6 (GroupKFold with shuffle).

Writes models/model.json, models/features.json, models/metrics.json,
models/seller_tier_effects.json and models/transfers_scored.parquet.
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

# `python src/train_model.py` puts src/ on the import path.
from config import DATA_DIR, MODELS_DIR
from seller_effects import estimate_tier_effects

NUMERIC = [
    "log_mv",
    "season",  # market-wide price level moves year to year
    "mv_trend_1y",
    "days_since_valuation",  # stale values (often young players) mislead
    "age",
    "height_in_cm",
    "apps_365d",
    "minutes_365d",
    "goals_per90",
    "assists_per90",
    "years_at_seller",
    "seller_is_academy",
    "seller_relegated",  # relegated last season: under pressure to sell
    "log_seller_squad_value",
    "seller_ppg_prev",
    "log_buyer_squad_value",
    "buyer_ppg_prev",
    "buyer_promoted",
    "buyer_big_six",
    # 17% of deals are dated 1 July (when they took effect), so the exact
    # days_to_deadline is noise; a final-week flag is the robust version.
    "deadline_week",
]
CATEGORICAL = ["position", "sub_position", "foot", "seller_tier", "window"]
TARGET = "log_fee"
PREMIUM_TARGET = "log_fee_to_value"  # what the full model actually predicts

PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    enable_categorical=True,
    tree_method="hist",
    random_state=42,
)


def make_X(df: pd.DataFrame):
    num = [c for c in NUMERIC if c in df.columns]
    cat = [c for c in CATEGORICAL if c in df.columns]
    X = df[num + cat].copy()
    levels = {}
    for c in cat:
        X[c] = X[c].astype("category")
        levels[c] = X[c].cat.categories.tolist()
    return X, num, cat, levels


def main():
    df = pd.read_parquet(DATA_DIR / "pl_transfers.parquet")
    df = df.dropna(subset=[TARGET, "log_mv"]).reset_index(drop=True)
    print(f"{len(df):,} paid transfers into the Premier League")

    df[PREMIUM_TARGET] = df[TARGET] - df["log_mv"]

    X, num, cat, levels = make_X(df)
    y = df[TARGET].values
    y_premium = df[PREMIUM_TARGET].values
    log_mv = df["log_mv"].values

    kf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_base = np.zeros(len(df))
    oof_premium = np.zeros(len(df))
    for tr_idx, te_idx in kf.split(X, groups=df["player_id"]):
        lin = LinearRegression().fit(df.loc[tr_idx, ["log_mv"]], y[tr_idx])
        oof_base[te_idx] = lin.predict(df.loc[te_idx, ["log_mv"]])

        m = xgb.XGBRegressor(**PARAMS)
        m.fit(X.iloc[tr_idx], y_premium[tr_idx])
        oof_premium[te_idx] = m.predict(X.iloc[te_idx])
    oof_xgb = log_mv + oof_premium  # back to log fee

    metrics = {
        "n_transfers": int(len(df)),
        "baseline_market_value_only": {
            "r2": round(float(r2_score(y, oof_base)), 3),
            "mae_log": round(float(mean_absolute_error(y, oof_base)), 3),
        },
        "full_model": {
            "r2": round(float(r2_score(y, oof_xgb)), 3),
            "mae_log": round(float(mean_absolute_error(y, oof_xgb)), 3),
            # share of the fee-vs-value gap itself that the features explain
            "r2_premium": round(float(r2_score(y_premium, oof_premium)), 3),
        },
        "median_fee_to_value": round(float(df["fee_to_value"].median()), 3),
        "share_paid_above_value": round(float((df["fee_to_value"] > 1).mean()), 3),
    }
    # a log error of 0.3 means "typically off by ~35%" -- easier to read
    for k in ("baseline_market_value_only", "full_model"):
        metrics[k]["typical_error_pct"] = round((np.exp(metrics[k]["mae_log"]) - 1) * 100, 1)
    print(json.dumps(metrics, indent=2))

    # final model on everything, for explanations. Contributions are in
    # premium space: they explain fee / market value, and sum (with the base)
    # to the predicted log premium.
    final = xgb.XGBRegressor(**PARAMS).fit(X, y_premium)
    contribs = final.get_booster().predict(
        xgb.DMatrix(X, enable_categorical=True), pred_contribs=True
    )
    feat_names = num + cat
    contrib_df = pd.DataFrame(contribs[:, :-1], columns=[f"shap_{c}" for c in feat_names])
    contrib_df["shap_base"] = contribs[:, -1]

    scored = pd.concat([df, contrib_df], axis=1)
    scored["expected_log_fee"] = oof_xgb
    scored["expected_fee"] = np.exp(oof_xgb)
    scored["residual_log"] = y - oof_xgb
    # +0.25 -> paid ~28% more than the model expected for that player/seller/situation
    scored["premium_vs_model"] = np.exp(scored["residual_log"]) - 1

    MODELS_DIR.mkdir(exist_ok=True)
    final.save_model(MODELS_DIR / "model.json")
    with open(MODELS_DIR / "features.json", "w") as f:
        json.dump({"numeric": num, "categorical": cat, "category_levels": levels}, f, indent=2)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    scored.to_parquet(MODELS_DIR / "transfers_scored.parquet", index=False)

    tier_effects = estimate_tier_effects(df)
    with open(MODELS_DIR / "seller_tier_effects.json", "w") as f:
        json.dump(tier_effects, f, indent=2)
    for name, spec in tier_effects["specs"].items():
        print(f"Selling-league effects vs. {tier_effects['reference']} ({name}, n={spec['n']}):")
        for e in spec["effects"]:
            print(f"  {e['tier']:<24} {e['effect']:+.0%}  ({e['low']:+.0%} to {e['high']:+.0%}, p={e['p_value']:.3f})")
    print(f"Saved model and scored transfers to {MODELS_DIR}")


if __name__ == "__main__":
    main()
