"""
Club fit finder — pick a club, define the role and playing style you want, and
get a ranked shortlist of players scored on ability, fit and value.

Stats come from the top-5 European leagues (2024/25 & 2025/26); the "value"
pillar uses Transfermarkt market values and the project's fee model. Run:

    python src/scout_prep.py   # once, to build data/scout_players.parquet
    streamlit run app.py

The fee-model analysis that this project grew out of lives on the "Fee analysis"
page in the sidebar.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

sys.path.insert(0, str(BASE / "src"))
from roles import METRIC_LABEL, ROLES, role_choices  # noqa: E402
from scouting import DEFAULT_WEIGHTS, profile_bars, shortlist  # noqa: E402

GOOD = "#0F766E"
MID = "#F59E0B"
POOR = "#DC2626"
NEUTRAL = "#94A3B8"

st.set_page_config(page_title="Club fit finder", page_icon="🎯", layout="wide")


def eur(x):
    if pd.isna(x):
        return "–"
    return f"€{x / 1e6:,.1f}m" if abs(x) >= 1e6 else f"€{x / 1e3:,.0f}k"


def band_color(pct):
    """FotMob-style traffic light for a percentile."""
    if pd.isna(pct):
        return NEUTRAL
    if pct >= 70:
        return GOOD
    if pct >= 40:
        return MID
    return POOR


@st.cache_data
def load():
    return pd.read_parquet(DATA / "scout_players.parquet")


try:
    df = load()
except FileNotFoundError:
    st.error("No scouting data found. Run `python src/scout_prep.py` first.")
    st.stop()

# ----------------------------------------------------------------------- header
st.title("🎯 Club fit finder")
st.write(
    "Pick a club and the kind of player you want, and get a ranked shortlist. Every candidate is "
    "scored three ways: **ability** (how good they are at the job, ranked against others in their "
    "position), **fit** (how closely their statistical profile matches the role you chose) and "
    "**value** (affordability against your budget, plus how much ability you get per euro)."
)
st.caption(
    f"{len(df):,} players from the top-5 European leagues, 2024/25–2025/26. "
    "Stats: FBref, Understat, SofaScore. Values: Transfermarkt (FIFA estimate where missing)."
)

# -------------------------------------------------------------------- the brief
with st.sidebar:
    st.header("The brief")

    clubs = sorted(df["team"].unique())
    default_club = clubs.index("Manchester City") if "Manchester City" in clubs else 0
    club = st.selectbox("Club you're buying for", clubs, index=default_club)

    role_opts = role_choices()
    role_key = st.selectbox(
        "Role / playing style", [k for k, _ in role_opts],
        format_func=lambda k: dict(role_opts)[k],
    )
    role = ROLES[role_key]
    st.caption(role.description)

    budget_m = st.slider("Budget (€m)", 0, 200, 60, step=5,
                         help="Players above your budget score lower on value; above 2× budget, zero.")
    budget = budget_m * 1e6 if budget_m > 0 else None

    max_age = st.slider("Maximum age", 17, 36, 27)
    exclude_own = st.checkbox(f"Exclude players already at {club}", value=True)

    st.subheader("What matters most")
    st.caption("Weights are relative and rescaled to sum to 1.")
    w_ability = st.slider("Ability", 0.0, 1.0, DEFAULT_WEIGHTS["ability"], 0.05)
    w_fit = st.slider("Fit", 0.0, 1.0, DEFAULT_WEIGHTS["fit"], 0.05)
    w_value = st.slider("Value", 0.0, 1.0, DEFAULT_WEIGHTS["value"], 0.05)
    weights = {"ability": w_ability, "fit": w_fit, "value": w_value}
    if sum(weights.values()) == 0:
        st.warning("Set at least one weight above zero.")
        st.stop()

# --------------------------------------------------------------- how club plays
st.subheader(f"How {club} plays")
club_rows = df[df["team"] == club]
if club_rows.empty:
    st.info(f"No players found for {club}.")
else:
    # the club's own profile on this role's metrics, averaged over its squad
    bars = []
    for col in role.metric_columns():
        pct_col = f"pct_{col}"
        if pct_col in club_rows:
            bars.append({"metric": METRIC_LABEL.get(col, col),
                         "percentile": club_rows[pct_col].mean()})
    club_prof = pd.DataFrame(bars).sort_values("percentile", ascending=False)
    st.caption(
        f"Where {club}'s squad already ranks on the metrics this role cares about "
        "(average percentile of its players vs. all players in their positions). "
        "Low bars are where the role could strengthen the side."
    )
    fig = go.Figure(go.Bar(
        x=club_prof["percentile"], y=club_prof["metric"], orientation="h",
        marker_color=[band_color(p) for p in club_prof["percentile"]],
        hovertemplate="%{y}: %{x:.0f}th percentile<extra></extra>",
    ))
    fig.update_layout(height=max(220, 26 * len(club_prof)), xaxis_title="Squad percentile",
                      yaxis_title="", xaxis_range=[0, 100], margin=dict(l=10))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------- shortlist
st.subheader(f"Best {role.name.lower()} fits for {club}")
sl = shortlist(
    df, role, budget_eur=budget, weights=weights,
    exclude_club=club if exclude_own else None, max_age=max_age, top_n=25,
)
if sl.empty:
    st.info("No candidates match this brief. Loosen the budget or age filter.")
    st.stop()

table = sl.assign(
    Value=sl["market_value_eur"].map(eur),
    Overall=sl["overall"].round(0),
    Ability=sl["ability"].round(0),
    Fit=sl["fit"].round(0),
    ValueScore=sl["value"].round(0),
).rename(columns={
    "player": "Player", "team": "Club", "age": "Age", "league": "League",
})[["Player", "Club", "League", "Age", "Value", "Overall", "Ability", "Fit", "ValueScore"]]
table = table.rename(columns={"ValueScore": "Value score"})

st.dataframe(
    table,
    hide_index=True, width="stretch",
    column_config={
        "Overall": st.column_config.ProgressColumn("Overall", min_value=0, max_value=100, format="%d"),
        "Ability": st.column_config.ProgressColumn("Ability", min_value=0, max_value=100, format="%d"),
        "Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%d"),
        "Value score": st.column_config.ProgressColumn("Value score", min_value=0, max_value=100, format="%d"),
    },
)
st.caption(
    "Overall is the weighted blend of the three pillars (weights in the sidebar). Ability and Fit are "
    "ranked within the player's position; Value combines affordability and ability-per-euro. "
    "'Value' column is the Transfermarkt market value, not a predicted fee."
)

# ------------------------------------------------------- one player, FotMob-style
st.subheader("Player profile")
pick = st.selectbox("Show the full profile for", sl["player"].tolist())
row = sl[sl["player"] == pick].iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Overall", f"{row['overall']:.0f}")
c2.metric("Ability", f"{row['ability']:.0f}")
c3.metric("Fit", f"{row['fit']:.0f}")
c4.metric("Value score", f"{row['value']:.0f}")
c5.metric("Market value", eur(row["market_value_eur"]))
st.write(
    f"**{row['player']}** — {row['team']} ({row['league'].split('-')[-1]}), "
    f"age {row['age']:.0f}, {row['pos_group']}. "
    f"Value source: {row.get('value_source', 'unknown')}."
)

bars = profile_bars(row, role)
fig = go.Figure(go.Bar(
    x=bars["percentile"], y=bars["metric"], orientation="h",
    marker_color=[band_color(p) for p in bars["percentile"]],
    customdata=bars["value"],
    hovertemplate="%{y}: %{x:.0f}th pct (raw %{customdata:.2f})<extra></extra>",
))
fig.update_layout(height=max(260, 28 * len(bars)), xaxis_title="Percentile vs. others in position",
                  yaxis_title="", xaxis_range=[0, 100], margin=dict(l=10))
st.plotly_chart(fig, width="stretch")
st.caption(
    "Each bar is where this player ranks against others in the same position on that metric. "
    "Green ≥ 70th percentile, amber 40–70, red below 40 — the same read as a FotMob stat page."
)
