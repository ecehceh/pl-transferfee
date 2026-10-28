# What Premier League clubs pay for a player, and why

When a Premier League club signs a player, the fee rarely matches the player's
listed market value. This project asks why: how much of a fee is explained by
the player, how much by the club selling, and how much by the buying club's own
habits — and which clubs consistently pay more or less than they "should".

Scope: every permanent, paid, disclosed transfer **into** a Premier League club,
from any league, 2020/21 to 2025/26.

**Explore the results:** Looker Studio report *(link to be added once
published)*. There's also a fuller interactive version in Streamlit (`app.py`),
where every number, ranges included, follows the filters.

## Key findings

> Figures below are from the final model: 696 paid deals, with the
> selling-league fix, relegated-seller flag and buyer-clustered standard
> errors.

- **Market value explains most of a fee, but not all of it.** Market value
  alone explains 64% of the variation in (log) fees; adding player form, the
  selling club, the buyer's situation and timing raises that to 74%. The
  features explain about 42% of *why* a fee lands above or below market value.
- **There is a domestic premium.** Buying from another Premier League club
  costs about **+19%** (90% CI +9% to +30%) more than a similar player from
  another top-5 league, and +18% even after controlling for how rich the
  selling club is, so it's about buying from a rival, not just from a rich
  club.
- **Some buyers really do pay more or less.** After accounting for the
  player, seller, timing and the buyer's size, 8 of 27 clubs stand clear of
  zero: **Fulham** gets a discount (−26%); **Chelsea**, **West Ham**,
  **Bournemouth**, **Ipswich**, **Leicester**, **Newcastle** and
  **Sunderland** pay a premium.
- **Chelsea's premium comes from buying young.** Leave out under-21
  signings and Chelsea is no longer clearly different from an average buyer
  (+11%, range crosses zero).
- **The "cheap" sellers aren't cheap like-for-like.** Clubs outside the
  covered leagues have the highest raw fee-to-value ratio (1.43×) because
  they sell young players with low listed values. Compared like-for-like, PL
  clubs pay them about 26% *less*.

## Club fit finder (scouting tool)

Beyond explaining fees, the app now answers a recruitment question: **pick a
club, choose the role and playing style you want, and get a ranked shortlist of
players.** Each candidate is scored three ways:

- **Ability** — how good they are at the things the role cares about, ranked by
  percentile *within their position* (the same read as a FotMob stat page).
- **Fit** — how closely their statistical profile matches the role template,
  regardless of overall level (a specialist can fit better than a better
  all-rounder).
- **Value** — affordability against your budget, plus how much ability you get
  per euro, using Transfermarkt market values and the fee model.

Stats come from the top-5 European leagues (2024/25 & 2025/26): xG, xA, duels,
aerials, passing splits and more, from FBref, Understat and SofaScore via the
[top5-football-dataset](https://github.com/m-mahadi/top5-football-dataset).
Transfermarkt market values are joined on from this project's own data. It ships
with ~12 role templates (ball-playing centre-back, deep-lying playmaker, inside
forward, poacher, pressing forward, and more); the weights are in
`src/roles.py` and easy to tweak.

## The questions

1. **Fee vs. value.** How far do fees drift from Transfermarkt market value, and
   how much does market value alone explain?
2. **Buying clubs.** After accounting for the player, seller, timing and the
   buyer's size, which clubs still pay a premium, and which get bargains?
3. **Selling clubs.** Do some leagues and clubs extract more from PL buyers?
   Is there a domestic premium on PL-to-PL deals?
4. **Drivers.** Which factors push a fee up or down the most?

## Method

- **Target:** log of the transfer fee.
- **Baseline:** linear regression on log market value only.
- **Full model:** XGBoost on player, seller, buyer-context and timing features.
  It predicts the *premium* — log(fee ÷ market value) — and adds market value
  back. Predicting the fee directly capped every estimate at about €90m
  (trees can't extrapolate), which made all record deals look like overpays.
  The buying club's *identity* is deliberately left out, so the leftover
  (residual) for each deal is that club's own premium or discount compared
  with a similar-sized club buying the same player.
- **Out-of-fold scoring:** 5-fold cross-validation grouped by player, so every
  deal is scored by a model that never saw it — or any other move by the same
  player.
- **Uncertainty:** each club's median premium comes with a 90% bootstrap
  range; clubs whose range crosses zero are shown as "not clearly different".
- **Explanations:** per-deal feature contributions (XGBoost's built-in SHAP
  values), which explain the premium over market value.
- **Domestic premium:** a separate linear regression (statsmodels OLS) of
  log(fee ÷ market value) on the selling league plus controls, so "PL clubs
  pay more to PL sellers" is a like-for-like estimate with a confidence
  interval rather than a raw average. Standard errors are clustered by buying
  club, since one club's deals aren't independent of each other. Run twice:
  with and without the selling club's squad value as a control.
- **Selling league** is the league the player was playing in when sold: last
  season's league for summer deals, the current one for winter deals. (Using
  the upcoming season's league would count a just-promoted club as a PL
  seller.) Clubs relegated the season before are flagged, because they sell
  under pressure and their summer sales count as PL sales.
- **Many comparisons:** with 27 clubs and 90% ranges, about 3 would look
  "clearly different" by chance alone; the app says so next to the chart.
- **Academy sides:** "Man City U21"-style clubs are mapped to their parent
  club; otherwise academy sales look like sales from obscure foreign clubs.

| Group | Features (all as of the transfer date) |
|---|---|
| Player | market value at transfer, 1-year value trend, days since that value was last updated, age, position, foot, height, minutes/goals/assists per 90 in the prior 12 months |
| Seller | league tier, relegated last season, squad value (excluding the player), last season's points per game, player's years at the club, sold from an academy side |
| Buyer | squad value before the window, last season's points per game, promoted flag, big-six flag |
| Timing | season (market-wide price level), summer/winter window, signed in deadline week |

## Run it

### 1. Get the data

If the r2.dev links below are blocked on your network, use the Kaggle mirror:

```bash
python src/download_data.py   # kagglehub -> copies the 7 CSVs into data/
```

Otherwise, from the [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)
project. Easiest is the single DuckDB file:

```bash
curl -L https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb \
  -o data/transfermarkt-datasets.duckdb
```

Or download these CSVs into `data/` (same URL pattern, e.g. `.../data/transfers.csv.gz`):
`transfers`, `players`, `player_valuations`, `appearances`, `clubs`, `games`, `competitions`.

### 2. Install, build, train, explore

```bash
pip install -r requirements.txt
python src/data_prep.py     # -> data/pl_transfers.parquet, data/prep_summary.json
python src/train_model.py   # -> models/ (model, metrics, scored transfers)
python src/scout_prep.py    # -> data/scout_players.parquet (for the fit finder)
streamlit run app.py
```

`prep_summary.json` records how many moves were dropped as free/loan or
undisclosed — worth quoting in a write-up.

### Scouting data

The club fit finder reads the top-5-leagues player dataset. Fetch it once into
`data/scout/`:

```bash
curl -L https://raw.githubusercontent.com/m-mahadi/top5-football-dataset/master/data/master/player_seasons.csv \
  -o data/scout/player_seasons.csv
```

Then `python src/scout_prep.py` joins it to the Transfermarkt values and writes
`data/scout_players.parquet`, which the app uses.

### 3. Looker Studio report

```bash
python src/export_looker.py   # -> exports/*.csv
```

Load the CSVs into Google Sheets and build the report by following
[`docs/looker_studio.md`](docs/looker_studio.md). Looker Studio can't run
Python, so the statistics (bootstrap ranges, the domestic-premium regression,
feature contributions) are computed by the export. The report only filters
and charts them, and club ranges come as fixed selections ("All deals",
"Excluding under-21s") rather than following every filter.

### 4. Test

```bash
pip install pytest
pytest -q
```

CI (`.github/workflows/ci.yml`) runs `ruff` and the tests on every pull request.

### 5. Deploy the Streamlit app (optional)

The app only reads four small files from `models/`: `transfers_scored.parquet`,
`metrics.json`, `features.json` and `seller_tier_effects.json`. `.gitignore`
lets these through, so after training, commit them and deploy `app.py` on
[Streamlit Community Cloud](https://streamlit.io/cloud). The raw data never has
to leave your machine.

## Data caveats (read before drawing conclusions)

- **Free transfers and loans both appear as a fee of 0** in the source data,
  and undisclosed fees are missing. Only paid, disclosed deals are modelled,
  so the results describe the visible part of the market.
- **Contract length at the time of sale isn't available** — the dataset only
  holds each player's *current* contract. "Years at selling club" is used as a
  rough stand-in for the seller's leverage, but it's a weak one.
- **Performance stats only exist for leagues the dataset covers.** Players
  signed from the Championship or most non-European leagues have no minutes or
  goals; the model treats this as missing, not zero.
- **Transfer dates are often the date a move took effect** (e.g. 1 July, 17% of
  deals), not when it was agreed. So the model only uses a coarse "signed in
  deadline week" flag, and even that misses deals agreed late but dated earlier.
- **Market value is itself an estimate**, partly informed by fees. Treat
  "paid above market value" as "paid above the crowd's estimate".
- **Young prospects' market values are the least reliable** — often a token
  €1m that is rarely updated — so their fee-to-value gaps can reach 20×. The
  app can leave out under-21 signings.
- Deadline dates in `src/config.py` are approximate.
- The source pipeline paused in July 2026, so the latest window is incomplete.

## Project structure

```
pl-transferfee/
├── app.py                 # Streamlit app: the club fit finder (scouting tool)
├── pages/
│   └── 2_Fee_analysis.py  # the fee-vs-value analysis this project grew from
├── src/
│   ├── config.py          # seasons, league tiers, big-six ids, deadlines
│   ├── download_data.py   # fetch the CSVs from the Kaggle mirror
│   ├── data_prep.py       # build one row per transfer, with features
│   ├── train_model.py     # baseline + XGBoost, CV residuals, contributions
│   ├── seller_effects.py  # domestic premium regression (OLS)
│   ├── analysis.py        # per-club premiums with bootstrap ranges
│   ├── labels.py          # readable feature names and groups
│   ├── export_looker.py   # CSVs for the Looker Studio report
│   ├── roles.py           # scouting metric vocabulary + role templates
│   ├── scout_prep.py      # build the scouting table (per-90s, percentiles, values)
│   └── scouting.py        # ability / fit / value scoring and ranking
├── docs/looker_studio.md  # how to build the Looker Studio report
├── tests/                 # pytest unit tests
├── data/                  # downloaded data (gitignored, except scout_players.parquet)
├── models/                # generated outputs (gitignored, except the app's 4 files)
├── exports/               # Looker Studio CSVs (gitignored)
├── pyproject.toml         # pytest and ruff settings
└── .streamlit/config.toml # app theme
```

## Ideas for v2

- Split the model by position group (defenders and forwards are priced on
  different things).
- Track whether "overpaid" signings went on to justify the fee (minutes played,
  resale value).
- Go further on selling clubs' financial pressure (relegation is now a
  feature; club losses and PSR/financial-fair-play pressure are not).
