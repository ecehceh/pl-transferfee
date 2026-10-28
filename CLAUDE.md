# Project context for Claude Code

This is a data analyst portfolio project, started in a claude.ai chat and moved
here. The user is building it to show their data analysis skills, so favour
clear, explainable analysis over squeezing out accuracy, and explain choices
as you go.

## The story

"What do Premier League clubs pay for a player, and why?" — how a club values
a player and what drives the fee it agrees to. Not a generic market value
predictor; the user explicitly moved away from that framing.

Decisions already made (don't revisit unless the user asks):
- Scope: Premier League only, transfers INTO PL clubs from any league
  (PL-to-PL included), seasons 2020/21 to 2025/26.
- Target: the actual transfer fee (log), not Transfermarkt market value.
  Market value at the time of the deal is a feature and the baseline.
  (2026-10-07) Internally the XGBoost model predicts log(fee / market value)
  and adds log market value back; metrics are still on log fee. Predicting
  the fee directly capped estimates at ~€90m and made every record deal look
  overpaid.
- Both buying and selling clubs are part of the story.
- Buying club identity is deliberately NOT a model feature, so out-of-fold
  residuals = that club's own premium/discount vs. a similar-sized club.
  Buyer size/form/big-six ARE features.
- Deliverable: Streamlit app (app.py).

## Data

transfermarkt-datasets (github.com/dcaribou/transfermarkt-datasets), current
to 6 July 2026. Download instructions are in README.md; files go in data/.
Tables used: transfers, players, player_valuations, appearances, clubs,
games, competitions.

Known schema quirks (verified against the repo's dbt models):
- transfers.transfer_fee: NULL = undisclosed; 0 = free transfer OR loan
  (any non-€ fee string becomes 0). Only fee > 0 is modelled.
- transfers.market_value_in_eur is the value at the time of the transfer.
- transfers.transfer_season is like "23/24".
- players.contract_expiration_date is the CURRENT contract only — not usable
  historically. years_at_seller is a weak proxy for seller leverage.
- player_valuations.current_club_id is the club at that valuation date (used
  for historical squad values).
- appearances only cover leagues the dataset scrapes; no-minutes is treated
  as missing, not zero.
- Which league a club was in each season is derived from games
  (competitions.type == "domestic_league"); PL id is "GB1".

## Status

- Pipeline: src/download_data.py (Kaggle mirror; r2.dev is blocked on the
  user's network) -> src/data_prep.py -> src/train_model.py -> app.py.
  src/analysis.py holds the bootstrap club summaries; tests in tests/.
- Runs on the REAL data (Kaggle version 679). 696 paid deals; 1,209 free/loan
  and 175 undisclosed dropped. Big-six ids verified against clubs. MV at
  transfer equals the last valuation BEFORE the deal in 100% of rows (no leak).
- v2 model (2026-10-07): premium target, season feature, GroupKFold by player.
  R² log fee: baseline 0.639, full 0.734; full model explains 0.405 of the
  log(fee/value) gap. Record deals now sane (Wirtz -4%, Isak +13%; Grealish
  +108% and Enzo +84% remain — known release-clause / deadline-day deals).
  v1 outputs kept in models/*_v1.* for comparison.
- v3 model (2026-10-07, current): + days_since_valuation, seller_is_academy;
  academy sides (U15-U23 suffix) mapped to parent club for seller league/form/
  squad value (12 deals; PL sellers 182 -> 193). R² log fee 0.743, premium
  R² 0.425. Academy outliers shrank (Kellyman +317% -> +134%, Edozie +345% ->
  +173%); non-academy prospects (Penders, Coulibaly) remain extreme —
  days_since_valuation is a weak feature. v2 outputs in models/*_v2.*.
- Buying-club premiums (v3, >=6 deals): 8/27 clear of zero — discount:
  Fulham, Forest; premium: Newcastle, Ipswich, Sunderland, Leicester, and
  borderline Chelsea, Bournemouth (lower bound ~0). Excluding U21 signings,
  Chelsea is no longer clear: its premium comes from buying young prospects.
- Domestic premium (models/seller_tier_effects.json, OLS + HC1): buying from a
  PL club costs +17% (95% CI +6% to +28%) vs. a similar player from another
  top-5 league; +16% (+5% to +27%) controlling for seller squad value. Sellers
  outside covered leagues: -24% like-for-like, despite the highest RAW ratio
  (1.58x) — they sell young, cheaply valued players.

- v4 changes (2026-10-28, branch improve-analysis-and-portfolio; NOT yet
  re-run on real data):
  - seller league = league the player was playing in when sold (summer:
    season-1, winter: season; no fallback to a neighbouring season). The
    old code took the upcoming season first, so promoted clubs counted as PL
    sellers; it could also have shifted the domestic premium.
  - new seller_relegated feature (also a control in the domestic-premium OLS).
  - days_to_deadline replaced in the model by deadline_week (0-7 days).
  - domestic-premium SEs clustered by buying club (was HC1).
  - app: grouped feature importance, chance-level note on club premiums.
  - scikit-learn >= 1.6 required (GroupKFold shuffle).

## Next steps

1. Re-run data_prep.py and train_model.py; compare with v3 (keep v3 outputs
   as models/*_v3.*) and update README "Key findings" + this status.
2. Commit the four app files in models/ (whitelisted in .gitignore), deploy
   to Streamlit Community Cloud, add a screenshot to the README.
