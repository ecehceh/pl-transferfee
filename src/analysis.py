"""
Summaries used by the app, kept here so they can be tested on their own.

A club's "premium" is the median of its out-of-fold residuals (log fee minus
the model's expected log fee). With only a handful of deals per club that
median is noisy, so every premium comes with a bootstrap interval: if the
interval crosses zero, the data can't tell that club apart from an average one.
"""

import numpy as np
import pandas as pd

N_BOOTSTRAP = 2000
CI_LEVEL = 0.90
SEED = 0


def bootstrap_median_ci(values, n_boot: int = N_BOOTSTRAP, level: float = CI_LEVEL, seed: int = SEED):
    """Percentile bootstrap interval for the median. Returns (low, high)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    medians = np.median(samples, axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(medians, [tail, 100 - tail])
    return float(low), float(high)


def group_premiums(df: pd.DataFrame, group_cols, min_deals: int, resid_col: str = "residual_log") -> pd.DataFrame:
    """Per group: deal count, money moved, median fee/value, and the median
    premium vs. the model with a bootstrap interval, all as fractions (0.1 = +10%)."""
    group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    rows = []
    for key, g in df.groupby(group_cols, observed=True):
        if len(g) < min_deals:
            continue
        key = key if isinstance(key, tuple) else (key,)
        low, high = bootstrap_median_ci(g[resid_col])
        rows.append({
            **dict(zip(group_cols, key)),
            "deals": len(g),
            "total_fee": g["transfer_fee"].sum(),
            "median_ratio": g["fee_to_value"].median(),
            "premium": np.expm1(g[resid_col].median()),
            "premium_low": np.expm1(low),
            "premium_high": np.expm1(high),
        })
    columns = group_cols + ["deals", "total_fee", "median_ratio", "premium", "premium_low", "premium_high"]
    out = pd.DataFrame(rows, columns=columns)
    # "clear" = the whole interval sits on one side of zero
    out["is_clear"] = (out["premium_low"] > 0) | (out["premium_high"] < 0)
    return out
