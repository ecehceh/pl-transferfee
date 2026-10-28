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
- Deliverables: a Looker Studio report (main portfolio view, built by the
  user from src/export_looker.py CSVs via Google Sheets, guide in
  docs/looker_studio.md) and the Streamlit app (app.py, kept as the fully
  interactive version). Statistics are computed in Python; Looker Studio
  only filters and charts them. Feature labels/groups live in src/labels.py
  so both stay in sync.

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

## New direction (2026-10-28): club fit finder

The user is extending the project from "explain the fee" into a FotMob-inspired
**player recruitment tool**: pick a club, define a target role + playing style,
and get a ranked shortlist scoring each candidate on **ability** (stat profile),
**fit** (how the profile matches the role) and **value** (affordability + the
fee model's view). The fee model stays as the "value" engine and lives on a
secondary page.

Decisions made:
- Stats source: `m-mahadi/top5-football-dataset` (top-5 leagues, 2024/25 &
  2025/26; xG/xA, duels, aerials, passing splits, FIFA attributes). Downloaded
  to `data/scout/player_seasons.csv`. It has no transfer fees on purpose, so
  the "value" pillar joins our Transfermarkt market values (from players.csv)
  by normalised name, falling back to the FIFA estimate.
- New modules:
  - `src/roles.py` — scouting metric vocabulary (label -> column + direction)
    and a library of ~12 role templates (weighted metric vectors) by position
    group (GK/DF/DM/MF/AM/W/FW).
  - `src/scout_prep.py` — builds `data/scout_players.parquet`: per-90 metrics,
    position groups, joined market value, within-position percentiles. Keeps
    the latest season per player, min 8 nineties.
  - `src/scouting.py` — the scoring engine: ability (weighted percentile mean),
    fit (cosine similarity of the percentile shape to the role vector), value
    (affordability + ability-per-euro), and `shortlist()` to rank. No file I/O
    at import, unit-tested in tests/test_scouting.py.
- App restructured into Streamlit multipage: `app.py` is now the club fit
  finder (FotMob-style percentile bars, traffic-light colours); the old fee
  analysis moved to `pages/2_Fee_analysis.py` (its BASE now points at the repo
  root via `.parent.parent`).
- `.gitignore` now also lets `data/scout_players.parquet` and
  `data/scout_summary.json` through so the app deploys without the raw CSV.

Caveats to keep honest in any write-up: FIFA/EA values are estimates not fees;
name joins are imperfect (~93% matched to Transfermarkt); the fee model was
trained on moves INTO the PL, so using it to judge a hypothetical signing is an
approximation.

## Earlier next steps (fee-model deliverables, still open)

1. Re-run data_prep.py and train_model.py — DONE 2026-10-28 (696 deals, R²
   0.639 -> 0.742, premium R² 0.423). README key findings updated and the four
   app files in models/ committed.
2. Run src/export_looker.py — DONE (exports/ CSVs produced). Building the
   Looker Studio report (docs/looker_studio.md), publishing, and adding the
   link + screenshot remain the user's manual steps.
