# What Premier League clubs pay for a player, and why

When a Premier League club signs a player, the fee rarely matches the player's
listed market value. This project asks why: how much of a fee is explained by
the player, how much by the club selling, and how much by the buying club's own
habits — and which clubs consistently pay more or less than they "should".

Scope: every permanent, paid, disclosed transfer **into** a Premier League club,
from any league, 2020/21 to 2025/26.

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
- **Domestic premium:** a separate linear regression (statsmodels OLS, robust
  standard errors) of log(fee ÷ market value) on the selling league plus
  controls, so "PL clubs pay more to PL sellers" is a like-for-like estimate
  with a confidence interval rather than a raw average. Run twice: with and
  without the selling club's squad value as a control.
- **Academy sides:** "Man City U21"-style clubs are mapped to their parent
  club; otherwise academy sales look like sales from obscure foreign clubs.

| Group | Features (all as of the transfer date) |
|---|---|
| Player | market value at transfer, 1-year value trend, days since that value was last updated, age, position, foot, height, minutes/goals/assists per 90 in the prior 12 months |
| Seller | league tier, squad value (excluding the player), last season's points per game, player's years at the club, sold from an academy side |
| Buyer | squad value before the window, last season's points per game, promoted flag, big-six flag |
| Timing | season (market-wide price level), summer/winter window, approximate days before deadline |

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
streamlit run app.py
```

`prep_summary.json` records how many moves were dropped as free/loan or
undisclosed — worth quoting in a write-up.

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
- **Transfer dates are often the date a move took effect** (e.g. 1 July), not
  when it was agreed, so the deadline-day feature is noisy.
- **Market value is itself an estimate**, partly informed by fees. Treat
  "paid above market value" as "paid above the crowd's estimate".
- **Young prospects' market values are the least reliable** — often a token
  €1m that is rarely updated — so their fee-to-value gaps can reach 20×. The
  app can leave out under-21 signings.
- Deadline dates in `src/config.py` are approximate.
- The source pipeline paused in July 2026, so the latest window is incomplete.

## Project structure

```
pl-transfer-fees/
├── app.py                 # Streamlit app
├── src/
│   ├── config.py          # seasons, league tiers, big-six ids, deadlines
│   ├── data_prep.py       # build one row per transfer, with features
│   └── train_model.py     # baseline + XGBoost, CV residuals, contributions
├── data/                  # downloaded data (gitignored)
├── models/                # generated outputs (gitignored)
└── .streamlit/config.toml # app theme
```

## Ideas for v2

- Split the model by position group (defenders and forwards are priced on
  different things).
- Track whether "overpaid" signings went on to justify the fee (minutes played,
  resale value).
- Add selling clubs' financial pressure, e.g. relegated that season.
