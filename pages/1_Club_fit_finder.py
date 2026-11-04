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

BASE = Path(__file__).resolve().parent.parent  # project root (this file lives in pages/)
DATA = BASE / "data"

sys.path.insert(0, str(BASE / "src"))
from role_validation import effective_weights, redundancy_report, weight_stability  # noqa: E402
from roles import METRIC_LABEL, ROLES, role_choices  # noqa: E402
from scouting import (  # noqa: E402
    DEFAULT_WEIGHTS,
    ability_score,
    fit_breakdown,
    profile_bars,
    shortlist,
)

GOOD = "#0D9488"
MID = "#F59E0B"
POOR = "#E11D48"
NEUTRAL = "#94A3B8"
INK = "#0F172A"


def style_fig(fig, height):
    """One consistent chart look: clean, light grid, brand typography."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=13),
        xaxis=dict(gridcolor="#EDF0F4", zerolinecolor="#E6E9EF"),
        yaxis=dict(gridcolor="#EDF0F4"),
    )
    return fig

# A focused visual layer. Targets stable Streamlit structures (test ids and
# semantic tags), not fragile generated class names, so it survives upgrades.
# shortcut: CSS leans on Streamlit's DOM; recheck if a major Streamlit release
# changes the layout markup.
st.markdown(
    """
    <style>
      :root { --surface:#FFFFFF; --line:#E6E9EF; --ink:#0F172A; --muted:#64748B; --brand:#0D9488; }
      /* roomier, centred reading column */
      .block-container { padding-top: 2.2rem; max-width: 1200px; }
      /* headings: tighter, more confident */
      h1 { font-weight: 800; letter-spacing: -0.02em; }
      h2, h3 { font-weight: 700; letter-spacing: -0.01em; }
      /* metric tiles become real cards with quiet elevation */
      [data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 12px; padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15,23,42,.04);
      }
      [data-testid="stMetricValue"] { font-weight: 800; letter-spacing: -0.02em; }
      [data-testid="stMetricLabel"] { color: var(--muted); text-transform: uppercase;
        font-size: .72rem; letter-spacing: .06em; }
      /* expanders read as panels, not default boxes */
      [data-testid="stExpander"] details {
        border: 1px solid var(--line); border-radius: 12px; background: var(--surface);
      }
      [data-testid="stExpander"] summary { font-weight: 600; }
      /* sidebar: a distinct control rail */
      [data-testid="stSidebar"] { background: #0F172A; }
      /* labels, captions, headings in the rail are light... */
      [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] p, [data-testid="stSidebar"] .stCaption,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * { color: #E2E8F0 !important; }
      /* ...but widget surfaces stay light with DARK text so they're readable */
      [data-testid="stSidebar"] [data-baseweb="select"] div,
      [data-testid="stSidebar"] input { color: #0F172A !important; }
      /* expanders in the rail: translucent panel, light header text */
      [data-testid="stSidebar"] [data-testid="stExpander"] details {
        background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.12);
      }
      [data-testid="stSidebar"] [data-testid="stExpander"] summary,
      [data-testid="stSidebar"] [data-testid="stExpander"] summary * { color: #E2E8F0 !important; }
      /* dataframe: lighter grid, more breathing room */
      [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; }
      /* primary buttons / tags pick up the brand accent */
      .stAlert { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
st.title("Club fit finder")
st.caption(
    "Set the brief in the sidebar, then read down: **1** where the club needs help · "
    "**2** the players who fit · **3** one player up close. "
    f"{len(df):,} players, top-5 leagues 2024/25–2025/26 · FBref, Understat, SofaScore, Transfermarkt."
)

with st.expander("How the three scores work"):
    st.markdown(
        "- **Ability** — how good the player is at this role's metrics, ranked against others in "
        "their position (0–100).\n"
        "- **Fit** — how closely their statistical *profile shape* matches the role, regardless of "
        "overall level.\n"
        "- **Value** — affordability against your budget, plus how much ability you get per euro.\n\n"
        "**Overall** is the weighted blend of the three (weights in the sidebar). Volume stats are "
        "possession-adjusted so a player at a high-possession side isn't flattered by his team's "
        "share of the ball. 'Value' is the Transfermarkt market value, not a predicted fee."
    )

# -------------------------------------------------------------------- the brief
with st.sidebar:
    st.subheader("The brief")

    # 1. Target — who and what are we scouting for
    clubs = sorted(df["team"].unique())
    default_club = clubs.index("Manchester City") if "Manchester City" in clubs else 0
    club = st.selectbox("Club", clubs, index=default_club)

    role_opts = role_choices()
    role_key = st.selectbox(
        "Role / playing style", [k for k, _ in role_opts],
        format_func=lambda k: dict(role_opts)[k],
    )
    role = ROLES[role_key]
    st.caption(role.description)

    # 2. Filters — narrow the candidate pool
    with st.expander("Filters", expanded=True):
        budget_m = st.slider("Budget (€m)", 0, 200, 60, step=5,
                             help="Players above budget score lower on value; above 2× budget, zero.")
        budget = budget_m * 1e6 if budget_m > 0 else None
        max_age = st.slider("Maximum age", 17, 36, 27)
        only_gettable = st.checkbox(
            "Only realistically available",
            help="Keeps players with ≤2 years left on contract (likelier to move, cheaper). "
                 "Unknown contracts are kept.",
        )
        exclude_own = st.checkbox(f"Exclude current {club} players", value=True)
        foot_pick = st.selectbox("Preferred foot", ["Any", "Left", "Right"])
        side_pick = "Any"
        if role.pos_group in ("W", "FB"):
            side_pick = st.selectbox("Side", ["Any", "Left", "Right"])

    # 3. Priorities — how to weight the three pillars
    with st.expander("Priorities", expanded=False):
        st.caption("Relative weights, rescaled to sum to 1.")
        w_ability = st.slider("Ability", 0.0, 1.0, DEFAULT_WEIGHTS["ability"], 0.05)
        w_fit = st.slider("Fit", 0.0, 1.0, DEFAULT_WEIGHTS["fit"], 0.05)
        w_value = st.slider("Value", 0.0, 1.0, DEFAULT_WEIGHTS["value"], 0.05)
    weights = {"ability": w_ability, "fit": w_fit, "value": w_value}
    if sum(weights.values()) == 0:
        st.warning("Set at least one weight above zero.")
        st.stop()

# ============================================================================
# STEP 1 — the need: where is this club strong or thin for the chosen role?
# ============================================================================
st.subheader(f"Step 1 · Where {club} is strong or thin at {role.name.lower()}")
club_rows = df[df["team"] == club]
weakest = None
if club_rows.empty:
    st.info(f"No players found for {club}.")
else:
    bars = [
        {"metric": METRIC_LABEL.get(col, col), "percentile": club_rows[f"pct_{col}"].mean()}
        for col in role.metric_columns() if f"pct_{col}" in club_rows
    ]
    club_prof = pd.DataFrame(bars).sort_values("percentile", ascending=False)
    weakest = club_prof.tail(2)["metric"].tolist() if len(club_prof) else []
    st.caption(
        f"Each bar is {club}'s squad average percentile on a metric this role is judged on, vs. all "
        "players in their positions. Short red/amber bars are where a signing would strengthen the side."
    )
    fig = go.Figure(go.Bar(
        x=club_prof["percentile"], y=club_prof["metric"], orientation="h",
        marker_color=[band_color(p) for p in club_prof["percentile"]],
        hovertemplate="%{y}: %{x:.0f}th percentile<extra></extra>",
    ))
    fig = style_fig(fig, max(220, 28 * len(club_prof)))
    fig.update_layout(xaxis_title="Squad percentile (higher = already strong)",
                      yaxis_title="", xaxis_range=[0, 100])
    st.plotly_chart(fig, width="stretch")

# ============================================================================
# STEP 2 — the answer: players who fit the role (and help where the club is thin)
# ============================================================================
bridge = f" {club} looks thinnest on **{weakest[0].lower()}**." if weakest else ""
st.subheader(f"Step 2 · Best {role.name.lower()} options for {club}")
st.caption(f"Ranked by how well each player fits the brief.{bridge}")

# apply the availability / profile filters before scoring
pool = df
if only_gettable:
    pool = pool[pool["contract_years_left"].isna() | (pool["contract_years_left"] <= 2)]
if foot_pick != "Any":
    pool = pool[pool["foot"] == foot_pick]
if side_pick != "Any":
    pool = pool[pool["side"] == side_pick]

sl = shortlist(
    pool, role, budget_eur=budget, weights=weights,
    exclude_club=club if exclude_own else None, max_age=max_age, top_n=25,
)
if sl.empty:
    st.info("No candidates match this brief. Loosen the budget, age, or availability filters.")
    st.stop()

def contract_label(years):
    if pd.isna(years):
        return "–"
    return "≤1 yr" if years <= 1 else f"{years:.0f} yrs"


TREND_TAG = {"Rising": "↑ Rising", "Fading": "↓ Fading", "Steady": "→ Steady",
             "New / one season": "· 1 season"}

table = sl.assign(
    Value=sl["market_value_eur"].map(eur),
    Contract=sl.get("contract_years_left", pd.Series(index=sl.index)).map(contract_label),
    Form=sl["trajectory"].map(lambda t: TREND_TAG.get(t, t)),
    Overall=sl["overall"].round(0),
    Ability=sl["ability"].round(0),
    Fit=sl["fit"].round(0),
    ValueScore=sl["value"].round(0),
    Why=sl["why"],
).rename(columns={
    "player": "Player", "team": "Club", "age": "Age", "league": "League",
})[["Player", "Club", "League", "Age", "Value", "Contract", "Form",
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
    "Scores 0–100; Ability/Fit ranked within position. 'Form' (Rising / Steady / Fading) is a tag "
    "only and doesn't change the scores. See \"How the three scores work\" at the top for the rest."
)

# ============================================================================
# STEP 3 — the detail: one player's full profile
# ============================================================================
st.subheader("Step 3 · Look at one player in detail")
pick = st.selectbox("Show the full profile for", sl["player"].tolist())
row = sl[sl["player"] == pick].iloc[0]

# Overall leads; the three pillars support it; market value is context.
head, pillars = st.columns([1, 3])
head.metric("Overall fit", f"{row['overall']:.0f}", help="Weighted blend of the three pillars.")
p1, p2, p3 = pillars.columns(3)
p1.metric("Ability", f"{row['ability']:.0f}")
p2.metric("Fit", f"{row['fit']:.0f}")
p3.metric("Value", f"{row['value']:.0f}", help=f"Market value {eur(row['market_value_eur'])}")

years = row.get("contract_years_left")
contract_txt = (
    "contract unknown" if pd.isna(years)
    else "contract ≤1 year left" if years <= 1
    else f"{years:.0f} years left on contract"
)
side_txt = f", {row['side'].lower()} side" if row.get("side") in ("Left", "Right") else ""
foot_txt = f", {row['foot'].lower()}-footed" if pd.notna(row.get("foot")) else ""
st.write(
    f"**{row['player']}** — {row['team']} ({row['league'].split('-')[-1]}), "
    f"age {row['age']:.0f}, {row['pos_group']}{side_txt}{foot_txt}, {contract_txt}. "
    f"Form trend: **{row.get('trajectory', 'unknown')}**. "
    f"Value source: {row.get('value_source', 'unknown')}."
)
st.info(f"**Why shortlisted:** {row['why']}")

prof_col, fit_col = st.columns(2)

with prof_col:
    st.markdown("**Stat profile** — percentile in position, raw value at the end")
    bars = profile_bars(row, role)
    fig = go.Figure(go.Bar(
        x=bars["percentile"], y=bars["metric"], orientation="h",
        marker_color=[band_color(p) for p in bars["percentile"]],
        text=bars["value_label"], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.0f}th percentile<extra></extra>",
    ))
    fig = style_fig(fig, max(260, 34 * len(bars)))
    fig.update_layout(xaxis_title="Percentile vs. position", yaxis_title="", xaxis_range=[0, 108])
    st.plotly_chart(fig, width="stretch")
    st.caption("Green ≥ 70th percentile, amber 40–70, red below 40. Volume stats possession-adjusted.")

with fit_col:
    st.markdown(f"**Why the fit score is {row['fit']:.0f}** — each metric's pull on the fit")
    fb = fit_breakdown(row, role)
    fig = go.Figure(go.Bar(
        x=fb["contribution"], y=fb["metric"], orientation="h",
        marker_color=[GOOD if h else POOR for h in fb["helps_fit"]],
        hovertemplate="%{y}: %{x:+.0f}% of the fit movement<extra></extra>",
    ))
    fig = style_fig(fig, max(260, 34 * len(fb)))
    fig.update_layout(xaxis_title="Share of fit (+ helps role, − against)", yaxis_title="")
    fig.add_vline(x=0, line_color=NEUTRAL, line_dash="dash")
    st.plotly_chart(fig, width="stretch")
    st.caption("Green: matches what the role emphasises. Red: a trait the role doesn't prioritise.")

# -------------------------------------------- methodology (kept last, out of the flow)
st.divider()
with st.expander("Methodology · how this role is weighted, and whether the weights matter"):
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
