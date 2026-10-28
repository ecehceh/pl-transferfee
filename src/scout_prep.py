"""
Build the scouting table the app reads.

Takes the raw top-5-leagues player-season dataset (data/scout/player_seasons.csv,
one row per player-season with xG, duels, passing, FIFA attributes) and turns it
into one analysis-ready table:

  - derives the per-90 metrics the role templates reference (roles.py)
  - assigns each player a position group (GK, DF, DM, MF, AM, W, FW)
  - joins a Transfermarkt market value from our existing players.csv, by name,
    so the "value" pillar uses real transfer-market money, not EA's estimate
  - computes a percentile (0-100) for every scouting metric, ranked *within
    position group* so a defender is compared with defenders

Keeps the most recent season per player (a player who appears in both 24/25 and
25/26 is represented by their latest season).

Run:  python src/scout_prep.py   -> data/scout_players.parquet
"""

import json
import re
import unicodedata

import numpy as np
import pandas as pd

# `python src/scout_prep.py` puts src/ on the import path.
from config import DATA_DIR
from roles import METRICS

SCOUT_CSV = DATA_DIR / "scout" / "player_seasons.csv"
PLAYERS_CSV = DATA_DIR / "players.csv"
OUT_PARQUET = DATA_DIR / "scout_players.parquet"
OUT_SUMMARY = DATA_DIR / "scout_summary.json"

# only rank players with a real sample of minutes
MIN_NINETIES = 8

# map FBref pos codes + sub-position to our 7 groups
WIDE_HINTS = ("Winger", "Wing")


def normalize_name(name: str) -> str:
    """Lowercase, strip accents and punctuation, collapse spaces -- for joining
    two datasets that spell names slightly differently."""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z\s]", " ", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def _per90(series: pd.Series, nineties: pd.Series) -> pd.Series:
    """Season total -> per 90, guarding against tiny-minute blow-ups."""
    out = series / nineties.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def position_group(pos: str, sub_position: str | float) -> str:
    """Collapse FBref `pos` and Transfermarkt sub-position into one of
    GK, DF, DM, MF, AM, W, FW. `pos` is like 'MF,FW'; first token leads."""
    if not isinstance(pos, str) or not pos:
        return "MF"
    primary = pos.split(",")[0]
    sub = sub_position if isinstance(sub_position, str) else ""
    if primary == "GK":
        return "GK"
    if primary == "DF":
        # full-backs and centre-backs both live in DF for now
        return "DF"
    if primary == "FW":
        if any(h in sub for h in WIDE_HINTS):
            return "W"
        return "FW"
    # midfield: split by sub-position hints when we have them
    if "Defensive Midfield" in sub:
        return "DM"
    if "Attacking Midfield" in sub or "Second Striker" in sub:
        return "AM"
    if any(h in sub for h in WIDE_HINTS):
        return "W"
    return "MF"


# Volume per-90 metrics split by whether the action happens with the ball or
# without it, for possession adjustment. A player at a high-possession side
# racks up on-ball volume he wouldn't at a weaker club, and gets fewer
# defensive actions; scaling by the team's share of the ball corrects for that.
# Rate/percentage metrics (pass %, duel win %, conversion) are already
# team-neutral and are deliberately left out of both lists.
ON_BALL_P90 = [
    "np_xg_p90", "xa_p90", "shots_p90", "key_passes_p90", "big_chances_created_p90",
    "final_third_passes_p90", "dribbles_p90", "touches_p90",
]
OFF_BALL_P90 = [
    "tackles_won_p90", "interceptions_p90", "clearances_p90", "blocks_p90",
    "ball_recovery_p90", "poss_won_att_third_p90",
]


def build_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the per-90 columns the role templates reference. Built in one
    concat to avoid fragmenting the frame. These complement the per-90s already
    in the raw file: np_xg_p90, xa_p90, shots_p90, key_passes_p90,
    interceptions_p90, tackles_won_p90."""
    n = df["nineties"]
    derived = pd.DataFrame({
        "big_chances_created_p90": _per90(df["ss_bigChancesCreated"], n),
        "final_third_passes_p90": _per90(df["ss_accurateFinalThirdPasses"], n),
        "dribbles_p90": _per90(df["ss_successfulDribbles"], n),
        "touches_p90": _per90(df["ss_touches"], n),
        "dispossessed_p90": _per90(df["ss_dispossessed"], n),
        "possession_lost_p90": _per90(df["ss_possessionLost"], n),
        "clearances_p90": _per90(df["ss_clearances"], n),
        "blocks_p90": _per90(df["ss_outfielderBlocks"].fillna(df["ss_blockedShots"]), n),
        "ball_recovery_p90": _per90(df["ss_ballRecovery"], n),
        "poss_won_att_third_p90": _per90(df["ss_possessionWonAttThird"], n),
        "errors_p90": _per90(df["ss_errorLeadToShot"].fillna(0) + df["ss_errorLeadToGoal"].fillna(0), n),
        "dribbled_past_p90": _per90(df["ss_dribbledPast"], n),
    })
    return pd.concat([df, derived], axis=1)


def possession_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Scale volume per-90s by the team's share of the ball, so a player's
    numbers reflect him rather than his team's style.

    Team possession is proxied by total touches per team-season relative to the
    league-season median (no possession column ships in the raw data). On-ball
    volume is divided by that share, off-ball volume multiplied by it; the
    unadjusted columns are kept as `<col>_raw` so the app can show both.
    """
    team_touches = (
        df.groupby(["league", "season", "team"])["ss_touches"].sum().rename("team_touches").reset_index()
    )
    league_median = (
        team_touches.groupby(["league", "season"])["team_touches"].median().rename("league_median")
    )
    team_touches = team_touches.merge(league_median, on=["league", "season"])
    team_touches["poss_share"] = (team_touches["team_touches"] / team_touches["league_median"]).clip(0.6, 1.6)
    df = df.merge(team_touches[["league", "season", "team", "poss_share"]], on=["league", "season", "team"], how="left")
    share = df["poss_share"].fillna(1.0)

    raw = {}       # keep the unadjusted values as <col>_raw
    adjusted = {}  # overwrite the live column with the adjusted value
    for col in ON_BALL_P90:
        if col in df.columns:
            raw[f"{col}_raw"] = df[col]
            adjusted[col] = df[col] / share
    for col in OFF_BALL_P90:
        if col in df.columns:
            raw[f"{col}_raw"] = df[col]
            adjusted[col] = df[col] * share
    df = df.assign(**adjusted)
    return pd.concat([df, pd.DataFrame(raw)], axis=1)


def add_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Years left on contract at the end of the player's season, from the FIFA
    contract-end year. `season` is like 2425 (24/25) -> reference year 2025."""
    ref_year = 2000 + (df["season"] % 100)  # 2425 -> 25 -> 2025; 2526 -> 2026
    df["contract_years_left"] = (df["fifa_contract_end"] - ref_year).clip(lower=0)
    return df.copy()


def add_market_value(df: pd.DataFrame) -> pd.DataFrame:
    """Join a Transfermarkt market value (EUR) by normalised name. Falls back to
    the FIFA value estimate when Transfermarkt has no match, flagged in
    `value_source`."""
    players = pd.read_csv(PLAYERS_CSV)
    players["name_key"] = players["name"].map(normalize_name)
    # a name can map to several players; keep the highest current value
    tm = (
        players.dropna(subset=["market_value_in_eur"])
        .sort_values("market_value_in_eur", ascending=False)
        .drop_duplicates("name_key")[["name_key", "market_value_in_eur", "sub_position"]]
        .rename(columns={"market_value_in_eur": "tm_value_eur"})
    )
    df["name_key"] = df["player"].map(normalize_name)
    df = df.merge(tm, on="name_key", how="left")
    df["market_value_eur"] = df["tm_value_eur"].fillna(df["fifa_value_eur"])
    df["value_source"] = np.where(
        df["tm_value_eur"].notna(), "Transfermarkt",
        np.where(df["fifa_value_eur"].notna(), "FIFA estimate", "unknown"),
    )
    return df.copy()  # de-fragment after the merge and inserts


def add_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """For each scouting metric, a 0-100 percentile ranked within position
    group, always oriented so higher = better. Built in one concat."""
    cols = {col: direction for (col, direction) in METRICS.values()}
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    pct = pd.DataFrame({
        f"pct_{col}": (df[col] * direction).groupby(df["pos_group"]).rank(pct=True) * 100
        for col, direction in cols.items()
    })
    return pd.concat([df, pct], axis=1)


def main():
    df = pd.read_csv(SCOUT_CSV).copy()  # de-fragment the wide raw frame once
    df["pos_group"] = [position_group(p, None) for p in df["pos"]]
    df = build_derived(df)
    df = possession_adjust(df)  # before percentiles: rank on the adjusted values
    df = add_market_value(df)
    df = add_contract(df)

    # refine group using Transfermarkt sub-position where the join gave one
    df = df.assign(pos_group=[
        position_group(p, sub) for p, sub in zip(df["pos"], df.get("sub_position"))
    ])
    df = add_percentiles(df)

    # keep players with a real sample, latest season per player
    df = df[df["nineties"] >= MIN_NINETIES].copy()
    df["season_rank"] = df.groupby("name_key")["season"].rank(method="first", ascending=False)
    df = df[df["season_rank"] == 1].drop(columns="season_rank")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    summary = {
        "players": int(len(df)),
        "with_transfermarkt_value": int((df["value_source"] == "Transfermarkt").sum()),
        "with_fifa_value_only": int((df["value_source"] == "FIFA estimate").sum()),
        "no_value": int((df["value_source"] == "unknown").sum()),
        "with_contract_years": int(df["contract_years_left"].notna().sum()),
        "position_groups": df["pos_group"].value_counts().to_dict(),
        "leagues": df["league"].value_counts().to_dict(),
        "min_nineties": MIN_NINETIES,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_PARQUET}")


if __name__ == "__main__":
    main()
