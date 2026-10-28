"""
What did Premier League clubs pay, and why?

Run: streamlit run app.py   (after src/data_prep.py and src/train_model.py)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"

sys.path.insert(0, str(BASE / "src"))
from analysis import CI_LEVEL, group_premiums  # noqa: E402

PREMIUM = "#C2410C"   # paid more than expected
DISCOUNT = "#0F766E"  # paid less than expected
NEUTRAL = "#94A3B8"
# Market values for young prospects are the least reliable (often a token
# €1m, updated rarely), so the app lets you set these deals aside.
YOUNG_AGE = 21

LABELS = {
    "log_mv": "Market value at transfer",
    "season": "Season",
    "mv_trend_1y": "Market value trend (1 yr)",
    "days_since_valuation": "Days since value last updated",
    "age": "Age",
    "height_in_cm": "Height",
    "apps_365d": "Appearances (last 12 months)",
    "minutes_365d": "Minutes (last 12 months)",
    "goals_per90": "Goals per 90",
    "assists_per90": "Assists per 90",
    "years_at_seller": "Years at selling club",
    "seller_is_academy": "Sold from an academy side",
    "seller_relegated": "Seller just relegated",
    "log_seller_squad_value": "Selling club squad value",
    "seller_ppg_prev": "Selling club points/game last season",
    "log_buyer_squad_value": "Buying club squad value",
    "buyer_ppg_prev": "Buying club points/game last season",
    "buyer_promoted": "Buyer just promoted",
    "buyer_big_six": "Buyer is big six",
    "days_to_deadline": "Days before deadline",
    "deadline_week": "Signed in deadline week",
    "position": "Position",
    "sub_position": "Detailed position",
    "foot": "Preferred foot",
    "seller_tier": "Selling league",
    "window": "Window (summer/winter)",
}
# Same groups as the README's feature table. Overlapping features (minutes and
# appearances, position and detailed position) split credit between them, so
# the grouped view is the fairer summary.
GROUPS = {
    "Player": ["log_mv", "mv_trend_1y", "days_since_valuation", "age", "height_in_cm", "apps_365d",
               "minutes_365d", "goals_per90", "assists_per90", "position", "sub_position", "foot"],
    "Selling club": ["seller_tier", "seller_relegated", "log_seller_squad_value", "seller_ppg_prev",
                     "years_at_seller", "seller_is_academy"],
    "Buying club": ["log_buyer_squad_value", "buyer_ppg_prev", "buyer_promoted", "buyer_big_six"],
    "Timing": ["season", "window", "deadline_week", "days_to_deadline"],
}

st.set_page_config(page_title="What PL clubs pay, and why", page_icon="⚽", layout="wide")


def eur(x):
    if pd.isna(x):
        return "–"
    return f"€{x / 1e6:,.1f}m" if abs(x) >= 1e6 else f"€{x / 1e3:,.0f}k"


def pct(x):
    return f"{x * 100:+.0f}%"


@st.cache_data
def load():
    df = pd.read_parquet(MODELS / "transfers_scored.parquet")
    metrics = json.loads((MODELS / "metrics.json").read_text())
    feats = json.loads((MODELS / "features.json").read_text())
    tier_effects = json.loads((MODELS / "seller_tier_effects.json").read_text())
    df["season_label"] = df["season"].map(lambda s: f"{s}/{str(s + 1)[-2:]}")
    df["deal"] = df["player_name"] + " → " + df["to_club_name"] + " (" + df["season_label"] + ")"
    # a player can join the same club twice in a season; add the date so both are listed
    dup = df["deal"].duplicated(keep=False)
    df.loc[dup, "deal"] += " " + pd.to_datetime(df.loc[dup, "transfer_date"]).dt.strftime("%d %b %Y")
    return df, metrics, feats, tier_effects


try:
    df, metrics, feats, tier_effects = load()
except FileNotFoundError:
    st.error("No trained model found. Run `python src/data_prep.py`, then `python src/train_model.py`.")
    st.stop()

# ----------------------------------------------------------------------- header
st.title("What Premier League clubs pay for a player, and why")
st.write(
    f"{metrics['n_transfers']:,} paid signings into the Premier League, "
    f"{df['season_label'].min()} to {df['season_label'].max()}. Each fee is compared with the "
    "player's Transfermarkt market value at the time, and with what a model expected "
    "given the player, the selling club and the timing of the deal."
)

with st.sidebar:
    st.header("Filters")
    seasons = sorted(df["season_label"].unique())
    pick_seasons = st.multiselect("Seasons", seasons, default=seasons)
    positions = sorted(df["position"].dropna().unique())
    pick_pos = st.multiselect("Positions", positions, default=positions)
    skip_young = st.checkbox(
        f"Leave out under-{YOUNG_AGE} signings",
        help="Young prospects often carry a token market value that is rarely updated, so their "
             "fee-to-value gaps are huge and say more about the valuation than the deal.",
    )
    view = df[df["season_label"].isin(pick_seasons) & df["position"].isin(pick_pos)]
    if skip_young:
        view = view[view["age"] >= YOUNG_AGE]
    st.caption(f"{len(view):,} deals selected")

if view.empty:
    st.info("No deals match these filters. Add a season or position in the sidebar.")
    st.stop()

tab_gap, tab_buy, tab_sell, tab_why, tab_deal = st.tabs(
    ["Fee vs. value", "Buying clubs", "Selling clubs", "What drives a fee", "Look up a deal"]
)

# ----------------------------------------------------------------- fee vs value
with tab_gap:
    c1, c2, c3 = st.columns(3)
    c1.metric("Median fee ÷ market value", f"{view['fee_to_value'].median():.2f}×")
    c2.metric("Deals above market value", f"{(view['fee_to_value'] > 1).mean() * 100:.0f}%")
    c3.metric("Total spent", eur(view["transfer_fee"].sum()))

    fig = px.scatter(
        view, x="mv_at_transfer", y="transfer_fee", color="seller_tier",
        hover_name="deal", log_x=True, log_y=True, opacity=0.75,
        hover_data={"mv_at_transfer": ":,.0f", "transfer_fee": ":,.0f", "seller_tier": False},
        labels={"mv_at_transfer": "Market value at transfer (€, log scale)",
                "transfer_fee": "Fee paid (€, log scale)", "seller_tier": "Selling league"},
    )
    lo = min(view["mv_at_transfer"].min(), view["transfer_fee"].min())
    hi = max(view["mv_at_transfer"].max(), view["transfer_fee"].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Fee = market value",
                             line=dict(color=NEUTRAL, dash="dash")))
    fig.update_layout(height=520, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("Points above the dashed line: the club paid more than the player's listed market value.")

    b, f = metrics["baseline_market_value_only"], metrics["full_model"]
    st.subheader("How much does market value alone explain?")
    st.write(
        f"Market value on its own explains **{b['r2'] * 100:.0f}%** of the variation in fees "
        f"(typically off by about {b['typical_error_pct']:.0f}%). Adding player form, the selling "
        f"club, the buyer's situation and timing raises that to **{f['r2'] * 100:.0f}%** "
        f"(typically off by about {f['typical_error_pct']:.0f}%). Put another way, the features "
        f"explain **{f['r2_premium'] * 100:.0f}%** of why a fee lands above or below market value. "
        "What remains is the part of the price these numbers can't account for — negotiation, "
        "need, and club habits."
    )

# ----------------------------------------------------------------- buying clubs
with tab_buy:
    st.write(
        "The model knows how rich and successful the buying club is, but not which club it is. "
        "So a club that consistently pays above the model's estimate is paying more than a club "
        "of its size usually would — for urgency, prestige or weak negotiating."
    )
    min_deals = st.slider("Minimum deals per club", 3, 20, 6, key="buy_min")
    g = group_premiums(view, "to_club_name", min_deals).sort_values("premium")
    if g.empty:
        st.info("No club has that many deals in this selection. Lower the minimum.")
    else:
        g["color"] = np.select([~g["is_clear"], g["premium"] > 0], [NEUTRAL, PREMIUM], DISCOUNT)
        fig = go.Figure(go.Bar(
            x=g["premium"] * 100, y=g["to_club_name"], orientation="h", marker_color=g["color"],
            error_x=dict(type="data", symmetric=False, color="#475569", thickness=1.2,
                         array=(g["premium_high"] - g["premium"]) * 100,
                         arrayminus=(g["premium"] - g["premium_low"]) * 100),
            customdata=np.stack([g["deals"], g["total_fee"] / 1e6, g["median_ratio"],
                                 g["premium_low"] * 100, g["premium_high"] * 100], axis=1),
            hovertemplate="%{y}<br>%{x:+.0f}% vs model "
                          "(range %{customdata[3]:+.0f}% to %{customdata[4]:+.0f}%)<br>"
                          "%{customdata[0]} deals, €%{customdata[1]:,.0f}m spent<br>"
                          "median fee/value %{customdata[2]:.2f}×<extra></extra>",
        ))
        fig.update_layout(height=max(350, 26 * len(g)), xaxis_title="Typical fee vs. model estimate (%)",
                          yaxis_title="", margin=dict(l=10))
        st.plotly_chart(fig, width="stretch")
        n_clear = int(g["is_clear"].sum())
        by_chance = (1 - CI_LEVEL) * len(g)
        st.caption(
            f"Median per club, with a {CI_LEVEL:.0%} bootstrap range. Teal: paid less than expected; "
            "orange: paid more. Grey: the range crosses zero, so these deals can't tell the club apart "
            f"from an average buyer. {n_clear} of {len(g)} clubs shown are clearly different. "
            f"Even if every club paid the same, about {by_chance:.0f} would look clearly different by "
            "chance alone, so treat clubs whose range only just clears zero with care."
        )

# ---------------------------------------------------------------- selling clubs
with tab_sell:
    st.subheader("Is there a domestic premium?")
    ref = tier_effects["reference"]
    spec = tier_effects["specs"]["player_and_buyer"]
    rich = tier_effects["specs"]["plus_seller_wealth"]
    pl = next(e for e in spec["effects"] if e["tier"] == "Premier League")
    pl_rich = next(e for e in rich["effects"] if e["tier"] == "Premier League")
    conf = f"{tier_effects['confidence']:.0%}"
    survives_wealth = pl_rich["low"] > 0 or pl_rich["high"] < 0
    st.write(
        f"Compared with a similar player bought from a club in another top-5 league (same market value, "
        f"age, position, season, window, buyer size, and whether the seller was just relegated), "
        f"Premier League clubs paid **{pct(pl['effect'])}** "
        f"when buying from another Premier League club ({conf} interval {pct(pl['low'])} to "
        f"{pct(pl['high'])}). Also controlling for how rich the selling club is gives "
        f"{pct(pl_rich['effect'])} ({pct(pl_rich['low'])} to {pct(pl_rich['high'])})"
        + (" — so the gap is about buying from a league rival, not just from a rich club."
           if survives_wealth else " — so much of the gap may simply reflect how rich PL sellers are.")
    )
    rows = []
    for name, label in [("player_and_buyer", "Player, timing & buyer controls"),
                        ("plus_seller_wealth", "+ selling club's squad value")]:
        for e in tier_effects["specs"][name]["effects"]:
            rows.append({**e, "spec": label})
    eff = pd.DataFrame(rows)
    fig = go.Figure()
    for (label, g), color in zip(eff.groupby("spec", sort=False), [DISCOUNT, PREMIUM]):
        fig.add_trace(go.Scatter(
            x=g["effect"] * 100, y=g["tier"], mode="markers", name=label, offsetgroup=label,
            marker=dict(color=color, size=10),
            error_x=dict(type="data", symmetric=False, array=(g["high"] - g["effect"]) * 100,
                         arrayminus=(g["effect"] - g["low"]) * 100, color=color),
            customdata=np.stack([g["low"] * 100, g["high"] * 100, g["deals"]], axis=1),
            hovertemplate="%{y}: %{x:+.0f}% (%{customdata[0]:+.0f}% to %{customdata[1]:+.0f}%)"
                          "<br>%{customdata[2]} deals<extra>" + label + "</extra>",
        ))
    fig.add_vline(x=0, line_color=NEUTRAL, line_dash="dash")
    fig.update_layout(height=320, scattermode="group", legend=dict(orientation="h", y=-0.3),
                      xaxis_title="Fee vs. market value, compared with other top-5 league sellers (%)")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Linear regression of log(fee ÷ market value) on the selling league plus controls, all "
        f"{spec['n']:,} deals (sidebar filters don't apply). Bars are {conf} intervals with standard "
        "errors clustered by buying club (a club's deals aren't independent of each other); a bar "
        f"crossing the dashed line means no clear difference from {ref.lower()} sellers. A seller's "
        "league is the one the player was playing in when sold."
    )

    st.subheader("By selling league (raw)")
    t = (
        view.groupby("seller_tier")
        .agg(deals=("transfer_id", "size"), median_ratio=("fee_to_value", "median"),
             median_resid=("residual_log", "median"))
        .reset_index()
    )
    t["Fee vs. model"] = (np.exp(t["median_resid"]) - 1).map(pct)
    t["Fee ÷ market value"] = t["median_ratio"].map(lambda x: f"{x:.2f}×")
    st.dataframe(
        t.rename(columns={"seller_tier": "Selling league", "deals": "Deals"})[
            ["Selling league", "Deals", "Fee ÷ market value", "Fee vs. model"]],
        hide_index=True, width="stretch",
    )
    st.caption(
        "Fee vs. model is near zero for each league because league is a model input. The raw fee ÷ "
        "market value mixes the league with who gets sold: clubs outside the covered leagues sell "
        "mostly young, cheaply valued players, so their raw ratio is high even though, like-for-like, "
        "PL clubs pay them less."
    )

    st.subheader("Clubs that sell to the Premier League most")
    min_sold = st.slider("Minimum deals per club", 2, 15, 4, key="sell_min")
    s = group_premiums(view, ["from_club_name", "seller_tier"], min_sold).sort_values("premium", ascending=False)
    if s.empty:
        st.info("No selling club has that many deals in this selection. Lower the minimum.")
    else:
        s["Fee vs. model"] = s["premium"].map(pct)
        s["Likely range"] = [f"{pct(lo)} to {pct(hi)}" for lo, hi in zip(s["premium_low"], s["premium_high"])]
        s["Clear?"] = np.where(s["is_clear"], "yes", "–")
        s["Received"] = s["total_fee"].map(eur)
        s["Fee ÷ market value"] = s["median_ratio"].map(lambda x: f"{x:.2f}×")
        st.dataframe(
            s.rename(columns={"from_club_name": "Club", "seller_tier": "League", "deals": "Deals"})[
                ["Club", "League", "Deals", "Received", "Fee ÷ market value", "Fee vs. model",
                 "Likely range", "Clear?"]],
            hide_index=True, width="stretch",
        )
        st.caption(
            f"Sorted by how much more than expected PL clubs paid them. 'Likely range' is a {CI_LEVEL:.0%} "
            "bootstrap range; 'Clear?' means it doesn't cross zero."
        )

# ------------------------------------------------------------- what drives a fee
with tab_why:
    feat_cols = feats["numeric"] + feats["categorical"]
    level = st.radio("Show", ["By group", "By factor"], horizontal=True)
    if level == "By group":
        # contributions add up, so a group's effect on a deal is the sum of its
        # members' -- then averaged in absolute size, like the per-factor view
        rows = []
        for group, members in GROUPS.items():
            cols = [f"shap_{c}" for c in members if c in feat_cols]
            if cols:
                rows.append({"label": group, "impact": view[cols].sum(axis=1).abs().mean()})
        imp = pd.DataFrame(rows)
        height = 300
    else:
        imp = pd.DataFrame({
            "label": [LABELS.get(c, c) for c in feat_cols],
            "impact": [view[f"shap_{c}"].abs().mean() for c in feat_cols],
        })
        height = 580
    imp = imp.sort_values("impact")
    fig = px.bar(imp, x="impact", y="label", orientation="h",
                 labels={"impact": "Average effect on fee ÷ market value (log scale)", "label": ""},
                 color_discrete_sequence=[DISCOUNT])
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The model starts from the player's market value and predicts how far above or below it the fee "
        "will land. This shows how much each factor moves that premium on average, in either direction. "
        "Related factors (e.g. minutes and appearances) share credit, so the group view is the fairer "
        "comparison."
    )

    st.subheader("How one factor pushes the price")
    choices = feats["numeric"] + feats["categorical"]
    pick = st.selectbox("Factor", choices, format_func=lambda c: LABELS.get(c, c))
    effect = (np.exp(view[f"shap_{pick}"]) - 1) * 100
    # log market value is unreadable on an axis; show it in euros on a log scale
    x_col = "mv_at_transfer" if pick == "log_mv" else pick
    if pick in feats["categorical"]:
        fig = px.box(view.assign(effect=effect), x=pick, y="effect", points="all", hover_name="deal",
                     labels={pick: LABELS.get(pick, pick), "effect": "Effect on fee vs. market value (%)"},
                     color_discrete_sequence=[DISCOUNT])
    else:
        fig = px.scatter(view.assign(effect=effect), x=x_col, y="effect", hover_name="deal", opacity=0.7,
                         log_x=pick == "log_mv",
                         labels={x_col: LABELS.get(pick, pick), "effect": "Effect on fee vs. market value (%)"},
                         color_discrete_sequence=[DISCOUNT])
    fig.add_hline(y=0, line_color=NEUTRAL, line_dash="dash")
    st.plotly_chart(fig, width="stretch")
    st.caption("Each dot is a deal. Above the line, this factor raised the expected fee for that deal.")

# --------------------------------------------------------------- deal explorer
with tab_deal:
    deal = st.selectbox("Deal", sorted(view["deal"].unique()), index=None, placeholder="Search a player or club")
    if deal:
        r = view[view["deal"] == deal].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fee paid", eur(r["transfer_fee"]))
        c2.metric("Market value then", eur(r["mv_at_transfer"]))
        c3.metric("Model expected", eur(r["expected_fee"]))
        c4.metric("Paid vs. model", pct(r["premium_vs_model"]))
        st.write(
            f"{r['player_name']}, {r['age']:.0f}, {r['sub_position'] or r['position']}, "
            f"from {r['from_club_name']} ({r['seller_tier'].lower()}) in the {r['window']} window."
        )

        feat_cols = feats["numeric"] + feats["categorical"]
        parts = pd.DataFrame({
            "label": [LABELS.get(c, c) for c in feat_cols],
            "effect": [(np.exp(r[f"shap_{c}"]) - 1) * 100 for c in feat_cols],
        })
        parts = parts.reindex(parts["effect"].abs().sort_values().index).tail(10)
        parts["color"] = np.where(parts["effect"] > 0, PREMIUM, DISCOUNT)
        fig = go.Figure(go.Bar(x=parts["effect"], y=parts["label"], orientation="h",
                               marker_color=parts["color"],
                               hovertemplate="%{y}: %{x:+.0f}%<extra></extra>"))
        fig.update_layout(height=420, xaxis_title="Effect on the expected fee (%)", yaxis_title="",
                          title="What pushed the expected fee above or below market value")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Breakdown from the model trained on all deals. 'Model expected' above comes from a model "
            "that never saw this deal, so the two can differ slightly."
        )
