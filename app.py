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
from role_validation import effective_weights, redundancy_report, weight_stability  # noqa: E402
from roles import METRIC_LABEL, ROLES, role_choices  # noqa: E402
from scouting import DEFAULT_WEIGHTS, ability_score, profile_bars, shortlist  # noqa: E402

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

def contract_label(years):
    if pd.isna(years):
        return "–"
    return "≤1 yr" if years <= 1 else f"{years:.0f} yrs"


table = sl.assign(
    Value=sl["market_value_eur"].map(eur),
    Contract=sl.get("contract_years_left", pd.Series(index=sl.index)).map(contract_label),
    Overall=sl["overall"].round(0),
    Ability=sl["ability"].round(0),
    Fit=sl["fit"].round(0),
    ValueScore=sl["value"].round(0),
    Why=sl["why"],
).rename(columns={
    "player": "Player", "team": "Club", "age": "Age", "league": "League",
})[["Player", "Club", "League", "Age", "Value", "Contract",
    "Overall", "Ability", "Fit", "ValueScore", "Why"]]
table = table.rename(columns={"ValueScore": "Value score"})

st.dataframe(
    table,
    hide_index=True, width="stretch",
    column_config={
        "Overall": st.column_config.ProgressColumn("Overall", min_value=0, max_value=100, format="%d"),
        "Ability": st.column_config.ProgressColumn("Ability", min_value=0, max_value=100, format="%d"),
        "Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%d"),
        "Value score": st.column_config.ProgressColumn("Value score", min_value=0, max_value=100, format="%d"),
        "Why": st.column_config.TextColumn("Why", width="large"),
    },
)
st.caption(
    "Overall is the weighted blend of the three pillars (weights in the sidebar). Ability and Fit are "
    "ranked within the player's position; Value combines affordability and ability-per-euro. "
    "'Value' is the Transfermarkt market value (not a predicted fee); 'Contract' is years left on the "
    "player's deal. Volume stats are possession-adjusted, so a player at a high-possession side isn't "
    "flattered by his team's share of the ball."
)

# -------------------------------------------------- how trustworthy is this role?
with st.expander("How this role is weighted, and whether the weights matter"):
    st.write(
        "The role's metric weights are expert judgement, not learned from labels — there's no "
        "ground truth for the 'right' recipe. Two checks keep them honest:"
    )
    raw = role.weights
    eff = effective_weights(df, role)
    ww = pd.DataFrame({
        "Metric": [METRIC_LABEL.get(c, c) for c in raw],
        "Set weight": [raw[c] for c in raw],
        "Effective weight": [round(eff[c], 2) for c in raw],
    }).sort_values("Effective weight", ascending=False)
    st.markdown("**1. Redundancy correction.** When two metrics measure nearly the same thing, the "
                "role would count that trait twice. Weights are deflated by how much each metric "
                "overlaps the others, so shared information is counted once. 'Effective weight' is "
                "what the shortlist actually uses.")
    st.dataframe(ww, hide_index=True, width="stretch")

    rep = redundancy_report(df, role)
    flagged = rep[rep["redundant"]] if not rep.empty else rep
    if not flagged.empty:
        pairs = ", ".join(
            f"{METRIC_LABEL.get(r['metric_a'], r['metric_a'])} ↔ "
            f"{METRIC_LABEL.get(r['metric_b'], r['metric_b'])} (r={r['corr']:.2f})"
            for _, r in flagged.iterrows()
        )
        st.caption(f"Overlapping metric pairs in this role: {pairs}.")
    else:
        st.caption("No strongly overlapping metric pairs in this role.")

    stab = weight_stability(
        df, role, lambda pool, w: ability_score(pool, role.with_weights(w)),
        n_trials=40, top_n=20,
    )
    st.markdown(
        f"**2. Stability.** Jittering every weight by ±25% over {stab['n_trials']} trials leaves the "
        f"ranking almost unchanged (rank correlation **{stab['mean_rank_corr']:.2f}**, and "
        f"**{stab['mean_top_n_overlap']:.0%}** of the top {stab['top_n']} stays in the top {stab['top_n']}). "
        "So the shortlist is driven by *which* metrics define the role, not the exact weights — the "
        "part that's a judgement call matters least."
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

years = row.get("contract_years_left")
contract_txt = (
    "contract unknown" if pd.isna(years)
    else "contract ≤1 year left" if years <= 1
    else f"{years:.0f} years left on contract"
)
st.write(
    f"**{row['player']}** — {row['team']} ({row['league'].split('-')[-1]}), "
    f"age {row['age']:.0f}, {row['pos_group']}, {contract_txt}. "
    f"Value source: {row.get('value_source', 'unknown')}."
)
st.info(f"**Why shortlisted:** {row['why']}")

bars = profile_bars(row, role)
fig = go.Figure(go.Bar(
    x=bars["percentile"], y=bars["metric"], orientation="h",
    marker_color=[band_color(p) for p in bars["percentile"]],
    text=bars["value_label"], textposition="outside", cliponaxis=False,
    hovertemplate="%{y}: %{x:.0f}th percentile<extra></extra>",
))
fig.update_layout(height=max(260, 30 * len(bars)), xaxis_title="Percentile vs. others in position",
                  yaxis_title="", xaxis_range=[0, 108], margin=dict(l=10))
st.plotly_chart(fig, width="stretch")
st.caption(
    "Each bar is where this player ranks against others in the same position on that metric; the number "
    "at the end is the raw value. Green ≥ 70th percentile, amber 40–70, red below 40 — the same read as "
    "a FotMob stat page. Volume metrics are possession-adjusted."
)
