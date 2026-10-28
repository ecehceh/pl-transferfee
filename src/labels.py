"""Readable names for model features, shared by the Streamlit app and the
Looker Studio export so both show the same wording."""

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
GROUP_OF = {feature: group for group, members in GROUPS.items() for feature in members}
