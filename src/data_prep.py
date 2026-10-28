"""
Build the modelling table: one row per permanent, paid, disclosed transfer
INTO a Premier League club, from any league, for the seasons in config.SEASONS.

Every feature is computed as of the transfer date (no peeking at later data):

  player   age, position, foot, height, market value at transfer,
           1-year market value trend, days since that value was last updated,
           minutes/goals/assists in the prior 365 days
  buyer    squad value before the window, last season's points per game,
           promoted-this-season flag, big-six flag
  seller   league tier, squad value before the window (excluding the player),
           last season's points per game (if their league is covered),
           how long the player had been at the club (rough leverage proxy),
           whether the player came from a club's academy side (U17-U23)

Academy sides ("Man City U21") have their own Transfermarkt ids and play in no
league the dataset covers, so on their own they look like obscure foreign
clubs. They are mapped to the parent club for league, form and squad value.
  timing   summer/winter window, approximate days before deadline

Reads data/transfermarkt-datasets.duckdb if present, otherwise CSVs in data/.
Writes data/pl_transfers.parquet and data/prep_summary.json.
"""

import json
import re
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import (  # noqa: E402
    BIG_SIX_CLUB_IDS,
    DATA_DIR,
    PREMIER_LEAGUE_ID,
    SEASONS,
    SUMMER_DEADLINES,
    WINTER_DEADLINES,
    seller_tier,
)

DUCKDB_PATH = DATA_DIR / "transfermarkt-datasets.duckdb"
# Youth/academy sides: "Man City U21", "Liverpool U23". Reserve "B"/"II" teams
# are left alone because many play in a real league of their own.
ACADEMY_SUFFIX = re.compile(r"\s+U(1[5-9]|2[0-3])$")
TABLES = ["players", "player_valuations", "appearances", "clubs", "games", "competitions", "transfers"]


# --------------------------------------------------------------------------- loading
def load_tables() -> dict:
    tables = {}
    if DUCKDB_PATH.exists():
        import duckdb

        print(f"Reading tables from {DUCKDB_PATH.name}")
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        for name in TABLES:
            tables[name] = con.execute(f"SELECT * FROM {name}").fetchdf()
        con.close()
    else:
        print(f"No duckdb file, reading CSVs from {DATA_DIR}")
        for name in TABLES:
            for ext in (".csv.gz", ".csv"):
                path = DATA_DIR / f"{name}{ext}"
                if path.exists():
                    tables[name] = pd.read_csv(path, low_memory=False)
                    break
            else:
                raise FileNotFoundError(
                    f"Missing table '{name}'. Put {name}.csv.gz in data/ "
                    "(or the full .duckdb file). See README.md."
                )
    for name, df in tables.items():
        print(f"  {name:<18} {len(df):>10,} rows")
    return tables


# --------------------------------------------------------------------------- helpers
def season_start_year(season_str) -> float:
    """'23/24' -> 2023. Returns NaN for anything unparseable."""
    m = re.match(r"^\s*(\d{2})/(\d{2})\s*$", str(season_str))
    if not m:
        return np.nan
    return 2000 + int(m.group(1))


def club_league_seasons(games: pd.DataFrame, competitions: pd.DataFrame) -> pd.DataFrame:
    """Which domestic league each club played in each season, and its points per game."""
    league_ids = set(competitions.loc[competitions["type"] == "domestic_league", "competition_id"])
    g = games[games["competition_id"].isin(league_ids)].copy()
    g = g.dropna(subset=["home_club_goals", "away_club_goals"])

    home = pd.DataFrame({
        "club_id": g["home_club_id"], "season": g["season"], "league_id": g["competition_id"],
        "gf": g["home_club_goals"], "ga": g["away_club_goals"],
    })
    away = pd.DataFrame({
        "club_id": g["away_club_id"], "season": g["season"], "league_id": g["competition_id"],
        "gf": g["away_club_goals"], "ga": g["home_club_goals"],
    })
    cg = pd.concat([home, away], ignore_index=True)
    cg["points"] = np.select([cg["gf"] > cg["ga"], cg["gf"] == cg["ga"]], [3, 1], 0)

    out = (
        cg.groupby(["club_id", "season", "league_id"], as_index=False)
        .agg(games=("points", "size"), points=("points", "sum"))
    )
    out["ppg"] = out["points"] / out["games"]
    # if a club somehow appears in two leagues in one season, keep the one with more games
    out = out.sort_values("games").drop_duplicates(["club_id", "season"], keep="last")
    out["club_id"] = out["club_id"].astype("int64")
    out["season"] = out["season"].astype("int64")
    return out[["club_id", "season", "league_id", "ppg"]]


def academy_parent_ids(transfers: pd.DataFrame) -> dict:
    """Map academy-side club ids to their parent club's id, by name
    ("Man City U21" -> "Man City"). If a parent name is used by several ids,
    the one that appears in the most transfers wins."""
    names = pd.concat([
        transfers[["from_club_id", "from_club_name"]].set_axis(["club_id", "name"], axis=1),
        transfers[["to_club_id", "to_club_name"]].set_axis(["club_id", "name"], axis=1),
    ]).dropna()
    names["club_id"] = names["club_id"].astype("int64")
    counts = names.groupby(["name", "club_id"]).size().rename("n").reset_index()
    id_by_name = counts.sort_values("n").drop_duplicates("name", keep="last").set_index("name")["club_id"]

    parents = {}
    for club_id, name in names.drop_duplicates("club_id").itertuples(index=False):
        if not ACADEMY_SUFFIX.search(name):
            continue
        parent = id_by_name.get(ACADEMY_SUFFIX.sub("", name))
        if parent is not None and parent != club_id:
            parents[int(club_id)] = int(parent)
    return parents


def days_since_last_valuation(deals: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
    """Days between each deal and the player's last valuation strictly before
    it -- the valuation transfers.market_value_in_eur is taken from. Old
    values (common for young players) are a weaker guide to the fee."""
    left = deals[["transfer_id", "player_id", "transfer_date"]].sort_values("transfer_date")
    right = (
        valuations.dropna(subset=["date"])[["player_id", "date"]]
        .rename(columns={"date": "valuation_date"})
        .sort_values("valuation_date")
    )
    last = pd.merge_asof(left, right, left_on="transfer_date", right_on="valuation_date",
                         by="player_id", direction="backward", allow_exact_matches=False)
    last["days_since_valuation"] = (last["transfer_date"] - last["valuation_date"]).dt.days
    return last[["transfer_id", "days_since_valuation"]]


def window_info(d: pd.Timestamp, season: int):
    """Return (window, reference date for squad snapshots, days to deadline)."""
    if 5 <= d.month <= 10:
        window = "summer"
        ref = pd.Timestamp(date(d.year, 6, 1))
        deadline = SUMMER_DEADLINES.get(season)
    else:
        window = "winter"
        ref = pd.Timestamp(date(d.year if d.month >= 11 else d.year - 1, 12, 1))
        deadline = WINTER_DEADLINES.get(season)
    days = (pd.Timestamp(deadline) - d).days if deadline else np.nan
    return window, ref, days


def squad_values(valuations: pd.DataFrame, ref_dates) -> pd.DataFrame:
    """Club squad value at each reference date: sum of each player's latest
    valuation in the prior 365 days, attributed to the club they were at then."""
    rows = []
    v = valuations.dropna(subset=["current_club_id"])
    for ref in sorted(set(ref_dates)):
        win = v[(v["date"] <= ref) & (v["date"] > ref - pd.Timedelta(days=365))]
        latest = win.sort_values("date").drop_duplicates("player_id", keep="last")
        agg = latest.groupby("current_club_id")["market_value_in_eur"].sum()
        rows.append(pd.DataFrame({"club_id": agg.index.astype("int64"), "ref_date": ref, "squad_value": agg.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["club_id", "ref_date", "squad_value"])


# --------------------------------------------------------------------------- main build
def build(t: dict):
    players, vals, apps = t["players"], t["player_valuations"].copy(), t["appearances"].copy()
    tr = t["transfers"].copy()

    for df, col in [(vals, "date"), (apps, "date"), (tr, "transfer_date")]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    players = players.copy()
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")

    leagues = club_league_seasons(t["games"], t["competitions"])
    league_lookup = leagues.set_index(["club_id", "season"])

    # --- transfers into a PL club, in our seasons -----------------------------
    tr["season"] = tr["transfer_season"].map(season_start_year)
    tr = tr.dropna(subset=["season", "transfer_date", "to_club_id", "from_club_id"])
    tr["season"] = tr["season"].astype("int64")
    tr["to_club_id"] = tr["to_club_id"].astype("int64")
    tr["from_club_id"] = tr["from_club_id"].astype("int64")
    tr = tr[tr["season"].isin(SEASONS)]

    pl_clubs = leagues[leagues["league_id"] == PREMIER_LEAGUE_ID][["club_id", "season"]]
    into_pl = tr.merge(pl_clubs, left_on=["to_club_id", "season"], right_on=["club_id", "season"])
    into_pl = into_pl.drop(columns="club_id")
    into_pl = into_pl[into_pl["from_club_id"] != into_pl["to_club_id"]]

    summary = {
        "all_moves_into_pl": int(len(into_pl)),
        "undisclosed_fee": int(into_pl["transfer_fee"].isna().sum()),
        "free_or_loan": int((into_pl["transfer_fee"] == 0).sum()),
    }
    df = into_pl[(into_pl["transfer_fee"] > 0) & (into_pl["market_value_in_eur"] > 0)].copy()
    summary["paid_with_market_value"] = int(len(df))
    df = df.rename(columns={"market_value_in_eur": "mv_at_transfer"})
    df = df.reset_index(drop=True)
    df["transfer_id"] = df.index

    # --- player attributes ----------------------------------------------------
    pcols = ["player_id", "name", "date_of_birth", "position", "sub_position", "foot", "height_in_cm"]
    df = df.merge(players[pcols], on="player_id", how="left")
    if "player_name" not in df.columns:
        df["player_name"] = df["name"]
    df["player_name"] = df["player_name"].fillna(df["name"]).fillna(df["player_id"].astype(str))
    df = df.drop(columns="name")
    df["age"] = (df["transfer_date"] - df["date_of_birth"]).dt.days / 365.25
    df = df[(df["age"] > 15) & (df["age"] < 42)].copy()

    df["log_fee"] = np.log(df["transfer_fee"])
    df["log_mv"] = np.log(df["mv_at_transfer"])
    df["fee_to_value"] = df["transfer_fee"] / df["mv_at_transfer"]

    # market value ~1 year before the transfer -> trend
    vals_sorted = vals.dropna(subset=["date", "market_value_in_eur"]).sort_values("date")
    lookup = df[["transfer_id", "player_id", "transfer_date"]].copy()
    lookup["date"] = lookup["transfer_date"] - pd.Timedelta(days=365)
    lookup = lookup.sort_values("date")
    past = pd.merge_asof(
        lookup, vals_sorted[["player_id", "date", "market_value_in_eur"]],
        on="date", by="player_id", direction="backward",
    )
    df = df.merge(past[["transfer_id", "market_value_in_eur"]].rename(
        columns={"market_value_in_eur": "mv_1y_before"}), on="transfer_id", how="left")
    df["mv_trend_1y"] = np.log(df["mv_at_transfer"]) - np.log(df["mv_1y_before"].where(df["mv_1y_before"] > 0))
    df = df.merge(days_since_last_valuation(df, vals), on="transfer_id", how="left")

    # performance in the 365 days before the transfer (covered competitions only)
    apps = apps[apps["player_id"].isin(df["player_id"])]
    apps_by_player = {pid: g for pid, g in apps.groupby("player_id")}
    perf = []
    for row in df[["transfer_id", "player_id", "transfer_date"]].itertuples(index=False):
        g = apps_by_player.get(row.player_id)
        if g is None:
            perf.append((row.transfer_id, 0, 0, 0, 0))
            continue
        w = g[(g["date"] < row.transfer_date) & (g["date"] >= row.transfer_date - pd.Timedelta(days=365))]
        perf.append((row.transfer_id, len(w), w["minutes_played"].sum(), w["goals"].sum(), w["assists"].sum()))
    perf = pd.DataFrame(perf, columns=["transfer_id", "apps_365d", "minutes_365d", "goals_365d", "assists_365d"])
    df = df.merge(perf, on="transfer_id", how="left")
    has_mins = df["minutes_365d"] > 0
    df["goals_per90"] = np.where(has_mins, df["goals_365d"] / df["minutes_365d"] * 90, np.nan)
    df["assists_per90"] = np.where(has_mins, df["assists_365d"] / df["minutes_365d"] * 90, np.nan)
    # zero minutes usually means "plays in a league we don't cover", not "never played"
    df.loc[~has_mins, ["minutes_365d", "apps_365d"]] = np.nan

    # --- timing ---------------------------------------------------------------
    info = [window_info(d, s) for d, s in zip(df["transfer_date"], df["season"])]
    df["window"] = [i[0] for i in info]
    df["ref_date"] = [i[1] for i in info]
    df["days_to_deadline"] = [i[2] for i in info]

    # --- buyer / seller context -----------------------------------------------
    def league_of(club_id, season):
        for s in (season, season - 1):
            if (club_id, s) in league_lookup.index:
                return league_lookup.loc[(club_id, s), "league_id"]
        return None

    def prev_ppg(club_id, season):
        key = (club_id, season - 1)
        return league_lookup.loc[key, "ppg"] if key in league_lookup.index else np.nan

    # academy sides stand in for their parent club from here on
    parents = academy_parent_ids(t["transfers"])
    df["seller_is_academy"] = df["from_club_id"].isin(parents).astype(int)
    df["seller_club_id"] = df["from_club_id"].map(parents).fillna(df["from_club_id"]).astype("int64")
    summary["academy_sellers_mapped_to_parent"] = int(df["seller_is_academy"].sum())

    df["seller_league"] = [league_of(c, s) for c, s in zip(df["seller_club_id"], df["season"])]
    df["seller_tier"] = df["seller_league"].map(seller_tier)
    df["seller_ppg_prev"] = [prev_ppg(c, s) for c, s in zip(df["seller_club_id"], df["season"])]
    df["buyer_ppg_prev"] = [
        prev_ppg(c, s) if league_of(c, s - 1) == PREMIER_LEAGUE_ID else np.nan
        for c, s in zip(df["to_club_id"], df["season"])
    ]
    df["buyer_promoted"] = [
        int((c, s - 1) not in league_lookup.index or league_lookup.loc[(c, s - 1), "league_id"] != PREMIER_LEAGUE_ID)
        for c, s in zip(df["to_club_id"], df["season"])
    ]
    df["buyer_big_six"] = df["to_club_id"].isin(BIG_SIX_CLUB_IDS).astype(int)

    sv = squad_values(vals, df["ref_date"].unique())
    df = df.merge(sv.rename(columns={"club_id": "to_club_id", "squad_value": "buyer_squad_value"}),
                  on=["to_club_id", "ref_date"], how="left")
    df = df.merge(sv.rename(columns={"club_id": "seller_club_id", "squad_value": "seller_squad_value"}),
                  on=["seller_club_id", "ref_date"], how="left")
    df["seller_squad_value_ex_player"] = (df["seller_squad_value"] - df["mv_at_transfer"]).clip(lower=0)
    df["log_buyer_squad_value"] = np.log1p(df["buyer_squad_value"])
    df["log_seller_squad_value"] = np.log1p(df["seller_squad_value_ex_player"])

    # tenure at the selling club (rough leverage proxy): days since the player
    # last moved TO the selling club
    all_tr = t["transfers"][["player_id", "transfer_date", "to_club_id"]].copy()
    all_tr["transfer_date"] = pd.to_datetime(all_tr["transfer_date"], errors="coerce")
    all_tr = all_tr.dropna().sort_values("transfer_date")
    all_tr["to_club_id"] = all_tr["to_club_id"].astype("int64")
    arrivals = {}
    for r in all_tr.itertuples(index=False):
        arrivals.setdefault((r.player_id, r.to_club_id), []).append(r.transfer_date)
    tenure = []
    for r in df[["player_id", "from_club_id", "transfer_date"]].itertuples(index=False):
        earlier = [d for d in arrivals.get((r.player_id, r.from_club_id), []) if d < r.transfer_date]
        tenure.append((r.transfer_date - max(earlier)).days / 365.25 if earlier else np.nan)
    df["years_at_seller"] = tenure

    summary["final_rows"] = int(len(df))
    summary["seasons"] = SEASONS
    return df, summary


def main():
    tables = load_tables()
    df, summary = build(tables)
    out = DATA_DIR / "pl_transfers.parquet"
    df.to_parquet(out, index=False)
    with open(DATA_DIR / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
