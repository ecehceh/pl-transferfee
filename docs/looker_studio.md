# Building the Looker Studio report

The report shows the same story as the Streamlit app, from five CSVs that
`src/export_looker.py` writes. Looker Studio can't run Python, so anything
statistical (bootstrap ranges, the regression, feature contributions) is
computed in advance. The report only filters and charts it. Budget about an
hour for the first build.

## 1. Export the data

```bash
python src/data_prep.py
python src/train_model.py
python src/export_looker.py    # -> exports/*.csv
```

| File | One row per | Used for |
|---|---|---|
| `deals.csv` | deal (~700) | scatter, KPIs, selling-league table, deal lookup |
| `contributions.csv` | deal × factor, plus deal × group (~20k) | "what drives a fee", deal breakdown |
| `club_premiums.csv` | role × selection × club | buying and selling club premiums |
| `league_effects.csv` | specification × selling league | domestic premium |
| `headline.csv` | (one row) | headline numbers that don't change with filters |

Conventions: percentages are fractions (`0.17` = +17%), money is in euros,
dates are `YYYY-MM-DD`, and flags are `Yes`/`No`.

## 2. Load into Google Sheets

1. Create a new spreadsheet, e.g. *PL transfer fees – report data*.
2. **First, File → Settings → Locale → United States.** The CSVs use `.` as
   the decimal point. In a locale that uses a decimal comma (Indonesia, most
   of Europe), `0.17` would be read as text or a date.
3. For each CSV: **File → Import → Upload**, then choose *Insert new sheet(s)*,
   separator *Detect automatically*, and tick *Convert text to numbers, dates,
   and formulas*. Name the tabs `deals`, `contributions`, `club_premiums`,
   `league_effects` and `headline`, then delete the empty `Sheet1`.

**Refreshing later:** re-run the export, then import each CSV again with
*Replace current sheet* into the same tab. The tab names stay the same, so the
report keeps working. Looker Studio caches Sheets data for up to 15 minutes. If
you added columns, open the data source in Looker Studio and click
*Refresh fields*.

## 3. Connect the data in Looker Studio

At [lookerstudio.google.com](https://lookerstudio.google.com), create a
**Blank report**. Use **Add data → Google Sheets** five times, picking the
spreadsheet and one tab each time, with *Use first row as headers* ticked.
Rename each data source after its tab.

Then open each data source (**Resource → Manage added data sources → Edit**)
and set these field types. Everything else can stay as detected.

| Data source | Field(s) | Type | Default aggregation |
|---|---|---|---|
| deals | `fee_eur`, `market_value_eur`, `expected_fee_eur` | Currency → EUR | Sum |
| deals | `premium_vs_model` | Percent | Median |
| deals | `fee_to_value` | Number | Median |
| deals | `transfer_id` | Text (it's an id, not a number) | — |
| deals | `transfer_date` | Date (YYYYMMDD) | — |
| contributions | `effect_pct` | Percent | Sum |
| contributions | `abs_effect_log`, `effect_log` | Number | Average |
| club_premiums | `premium`, `premium_low`, `premium_high` | Percent | Average |
| club_premiums | `total_fee_eur` | Currency → EUR | Sum |
| league_effects | `effect`, `low`, `high` | Percent | Average |
| headline | every `domestic_premium*`, `share_*`, `typical_error_*`, `r2_*` | Percent | Max |
| headline | `total_spent_eur` | Currency → EUR | Max |

The `headline` table has one row, so *Max* just returns its value.

Add these **calculated fields** to `deals` (*Add a field* in the data source):

| Name | Formula | Type |
|---|---|---|
| Deals | `COUNT_DISTINCT(transfer_id)` | Number |
| Share above market value | `SUM(CASE WHEN fee_to_value > 1 THEN 1 ELSE 0 END) / COUNT(transfer_id)` | Percent |
| Overall fee ÷ value | `SUM(fee_eur) / SUM(market_value_eur)` | Number |

## 4. Theme

Use **Theme and layout → Customize** to match the app:

| Role | Colour |
|---|---|
| Accent / "paid less than expected" | `#0F766E` (teal) |
| "Paid more than expected" | `#C2410C` (orange) |
| "Not clearly different" / reference lines | `#94A3B8` (grey) |
| Text | `#1E293B` |
| Page background | `#FFFFFF`; card background `#F1F5F4` |

To colour by verdict, use `verdict` as the chart's **breakdown dimension**, then
set each value's colour in **Style → Colour by → Dimension values**. Set them
once and every chart using that field picks them up.

## 5. Pages

Put a title and one or two sentences of context at the top of every page.
Copy the wording from the app (`app.py`), which has already been through
review.

### Page 1: Fee vs. value

- **Filter controls** (drop-down lists, all from `deals`): `season`,
  `position`, `under_21`. Put them in the page header and make them
  report-level (*right-click → Make report-level*), so they apply on every
  page.
- **Scorecards** (`deals`): *Deals*, *Median of fee_to_value* (label "Median fee
  ÷ market value"), *Share above market value*, *Sum of fee_eur* (label
  "Total spent", compact numbers).
- **Scatter chart** (`deals`): dimension `deal`; X = `market_value_eur`,
  Y = `fee_to_value` (both aggregation *Sum*, which equals the value with one
  row per deal); bubble colour = `selling_league`. In *Style*, tick **Log
  scale** on both axes, and add a **reference line**: Y, constant value `1`,
  grey, dashed, labelled "Fee = market value".
  (Looker Studio can't draw a y = x diagonal on a fee-vs-value chart; fee ÷ value
  against a line at 1× shows the same thing.)
- **Text box with the model numbers** (scorecards from `headline`):
  `r2_market_value_only` "Market value alone explains", `r2_full_model` "With
  all features", `r2_premium_explained` "Share of the fee-vs-value gap
  explained".

### Page 2: Buying clubs

- **Drop-down list** on `club_premiums.selection`, single select, default
  "All deals". Explain beside it: "Ranges are computed in Python and don't
  follow the season/position filters. Choose a fixed selection here."
- **Bar chart** (`club_premiums`, chart filter `role` = `Buying`):
  dimension `club`, metric `premium`, **breakdown dimension `verdict`**,
  stacked, horizontal, sorted by `premium` ascending, up to 30 bars. Each club has a
  single verdict, so every bar gets one colour. Add a reference line at `0`.
- **Table** under it, with the same source and filter: `club`, `deals`,
  `total_fee_eur`, `median_fee_to_value`, `premium`, `likely_range`, `clear`.
  Add conditional formatting on `verdict` that matches the colours.
- **Caption:** "Median premium vs. the model with a 90% bootstrap range.
  With ~27 clubs, about 3 would look 'clearly different' by chance alone."
  You can put the actual numbers in scorecards from `headline`:
  `buying_clubs_clear` and `buying_clubs_clear_by_chance`.

There are no error bars in Looker Studio, so the range sits in the table.

### Page 3: Selling clubs

- **Domestic premium scorecards** (`headline`): `domestic_premium`, plus
  `domestic_premium_low` / `domestic_premium_high` as the "95% interval".
- **Bar chart** (`league_effects`): dimension `selling_league`, breakdown
  dimension `specification` (grouped, not stacked), metric `effect`, with a
  reference line at `0`. Show `likely_range` and `deals` in a table next to it.
  Caption: "Linear regression of log(fee ÷ market value), standard errors
  clustered by buying club; all deals, so page filters don't apply."
- **Raw table by selling league** (`deals`): dimension `selling_league`;
  metrics *Deals*, *Median fee_to_value*, *Median premium_vs_model* (label
  "Fee vs. model"). Copy the app's caption about why raw ratios mislead.
- **Selling-club table** (`club_premiums`, filters `role` = `Selling` and the
  page's `selection` drop-down): `club`, `league`, `deals`, `total_fee_eur`,
  `median_fee_to_value`, `premium`, `likely_range`, `clear`, sorted by
  `premium` descending.

### Page 4: What drives a fee

- **Drop-down list** on `contributions.level`, single select, default
  `Group`.
- **Bar chart** (`contributions`): dimension `factor`, metric
  `abs_effect_log` (*Average*), sorted descending, horizontal, teal. Label the
  axis "Average effect on fee ÷ market value (log scale)".
- The page-1 filters also apply here, because `contributions` has
  `season`, `position` and `under_21` columns with the same names. If they
  don't filter it in your version, add copies of those controls built on
  `contributions`.

### Page 5: Look up a deal

- **Drop-down list** on `contributions.deal`, single select, with search
  enabled. Set a default deal (e.g. a record signing) so the page never
  opens empty.
- **Scorecards** (`deals`): `fee_eur`, `market_value_eur`,
  `expected_fee_eur`, `premium_vs_model`. They follow the drop-down because
  both sources have a `deal` field.
- **Bar chart** (`contributions`, chart filter `level` = `Factor`):
  dimension `factor`, metric `effect_pct` (*Sum*), **sorted by
  `abs_effect_log` descending** (you can sort by a metric that isn't
  displayed), 10 bars, horizontal. Title: "What pushed the expected fee
  above or below market value".

### Page 6: Method & caveats

Text only. Adapt the README's *Method* and *Data caveats* sections; the free
transfer/loan and young-player valuation caveats matter most.

## 6. Publish

1. **Share → Manage access → Anyone on the internet can view.**
2. Keep each data source's *Data credentials* as **Owner's credentials** (the
   default). That way viewers don't need access to the spreadsheet.
3. Copy the report link into the README, and save a screenshot as
   `docs/report.png`.
