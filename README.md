# AI Travel Analyst — Flight Price Exploration (Part 1)

**MIC AIML Department Recruitment 2026 — Data Science & Visualization Track**

## Project Overview

This project explores a real-world flight pricing dataset to identify the
structural factors that actually drive airfare, and translates those findings
into concrete, honest recommendations for travelers. Rather than treating
"5 visualizations" as a checklist, each chart here builds toward a single
central question: **which factors genuinely explain flight price variation,
and which ones only look important at a glance?**

## Problem Statement

Flight pricing appears volatile and unpredictable to travelers, but it is
largely governed by identifiable structural factors — distance, airline
pricing strategy, and booking timing. Most travelers have no visibility into
*which* of these factors dominates for their specific route, so they either
overpay or book at the wrong time. This project decomposes flight pricing
into its actual drivers, using both visual and quantitative methods, so the
resulting recommendations are grounded in evidence rather than assumption.

## Installation Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Aashik-sketch/mic-3.git
cd mic-3

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the cleaning script (produces flight_pricing_clean.csv)
python 01_cleaning_preprocessing.py

# 5. Run the exploration & visualization script (produces all charts)
python 02_eda_visualizations.py
```

## Dataset Used

`flight_pricing_dataset.csv` — 100,000 raw flight records with 18 columns:
airline, source/destination, departure date & time, duration, stops,
distance, travel class, days before departure, season, weekday, aircraft
type, booking channel, passenger count, and price.

The raw data was intentionally messy in realistic ways: mixed time formats
(`8:10 PM` vs `12:10`), mixed duration formats (`1.67` decimal hours vs
`0h 45m` vs `177 min`), mixed stop formats (`non-stop` vs `1 stop` vs `1`),
and inconsistent location naming (`Bangalore Airport` vs `BLR` vs
`Bangalore`).

## Methodology

**1. Cleaning & Preprocessing** (`01_cleaning_preprocessing.py`)
- Removed exact duplicate rows
- Standardized airline names (case inconsistency)
- Mapped all source/destination values (city names, airport-suffixed names,
  and IATA codes) to one canonical city name
- Parsed `Total_Stops` into a consistent integer scale
- Converted all `Duration` formats into a single `Duration_min` numeric field
- Fixed a unit-suffix bug where some `Distance_km` values carried a literal
  `"km"` string (e.g. `"298.6 km"`), which silently became missing values
  under a naive numeric conversion
- Normalized `Departure_Time`/`Arrival_Time` to 24-hour format and derived
  `Departure_Hour` as a numeric feature
- Backfilled missing `Season`/`Weekday` from `Departure_Date` where possible
- Dropped rows with missing `Price` (the target variable)
- Flagged (not deleted) statistical outliers via the IQR method, since
  unusually cheap/expensive flights are a genuine analytical signal, not
  noise to discard

**2. Exploration & Visualization** (`02_eda_visualizations.py`)
Six charts plus one quantitative ranking, described in Results below.

**3. Factor Identification**
Two independent methods were used so findings aren't dependent on a single
technique:
- **Pearson correlation**, visualized as a clustered heatmap
- **Mutual information regression**, which also captures non-linear
  relationships that correlation can miss

## Technologies Used

- **Python 3** — pandas, NumPy for data cleaning and transformation
- **Matplotlib & Seaborn** — static statistical visualizations
- **Plotly** — interactive Sankey diagram
- **Scikit-learn** — mutual information ranking (`mutual_info_regression`)

## Results

**Chart 1 — Ridgeline plot (price distribution per airline):**
Most international carriers (Emirates, Qatar Airways, Singapore Airlines,
etc.) show a tightly clustered, high-end price distribution. Vistara stands
out with a distinctly different, left-shifted, bimodal distribution —
suggesting a mix of domestic short-haul and international long-haul routes
under one airline.

**Chart 2 — Price elasticity curve (price vs. days before departure):**
Average fares stay roughly flat from 180 days out to about 31 days out, then
rise sharply inside the 14-day mark — nearly 50% higher in the final 0–3 day
window compared to 31+ days out. This is a clear "panic zone" across almost
every airline in the dataset.

**Chart 3 — Violin plot (price vs. number of stops):**
Counter-intuitively, direct (0-stop) flights are the *cheapest* on average,
and price rises with each additional stop. This initially looks surprising,
but Chart 4 explains why.

**Chart 4 — Clustered correlation heatmap:**
`Distance_km` and `Duration_min` correlate at 0.99 with each other (as
expected) and at ~0.68–0.69 with `Price`. `Total_Stops` correlates with
`Price` at only 0.12 — far weaker. This reveals that stops don't directly
cause higher prices; in this dataset, multi-stop flights are simply
correlated with longer international routes, which are the real driver.

**Chart 5 — Price variance within each booking window:**
Even within a single days-before-departure bucket, the price standard
deviation (roughly ₹47,000–58,000) is nearly as large as the mean price
itself. This is directly relevant to a common traveler question: *if I book
early, could the price still drop later?* The answer, shown here, is yes —
not because of any error in the analysis, but because genuine price variance
exists even under similar booking conditions.

**Chart 6 — Sankey diagram (top routes by volume, colored by price):**
Dubai appears repeatedly as a hub source across several top routes (to
Bangkok, Mumbai, Jaipur, Pune, Sydney, and London), while Bangkok is the most
common destination across multiple origins — consistent with its role as a
major connecting hub.

**Chart 7 / Mutual Information Ranking — the quantitative answer to
"what drives price":**

| Rank | Feature | Mutual Info Score |
|------|---------|-------------------|
| 1 | Distance_km | ~1.02 |
| 2 | Duration_min | ~0.76 |
| 3 | Airline | ~0.48 |
| 4 | Destination | ~0.33 |
| 5 | Source | ~0.33 |
| 6 | Travel_Class | ~0.14 |
| 7 | Days_Before_Departure | ~0.03 |
| 8 | Total_Stops | ~0.02 |

This ranking is the key honest finding of the project: **Distance and
Duration are, by a wide margin, the strongest drivers of price.** Airline
choice has a real, independent effect (0.48) on top of route distance.
Interestingly, `Days_Before_Departure` and `Total_Stops` rank low here even
though Charts 2 and 3 show visually dramatic patterns for both — this isn't
a contradiction. It means their effect on price is real but secondary to
distance, and it's a reminder that visually striking trends and overall
statistical importance are two different things worth checking against
each other.

## Insights & Recommendations

1. **Book at least 2–3 weeks ahead where possible.** Prices rise sharply
   inside the 14-day mark across nearly every airline in this dataset.
2. **Don't optimize for "fewer stops" alone.** Check actual route distance
   first — the stop count is largely a side-effect of route length, not an
   independent price driver.
3. **Compare airlines directly on the same route.** Airline choice carries
   a real, independent pricing effect (MI score 0.48) beyond what distance
   alone explains.
4. **Treat any single price prediction as a range, not a guarantee.** The
   variance within any booking window is large enough that a genuinely
   cheaper fare can still appear after you've already booked — this
   reflects real-world pricing volatility that a static historical dataset
   cannot fully eliminate, not a flaw in this analysis.

## Challenges Faced

- The raw dataset mixed at least three different formats each for airline
  names, stop counts, durations, and location identifiers, requiring
  format-aware parsing rather than a single `pd.to_numeric()` call.
- A subtle bug was found during testing: some `Distance_km` values carried a
  literal `"km"` unit suffix (e.g. `"298.6 km"`), which silently converted
  to missing values under naive numeric coercion. This was caught by
  comparing null counts before and after cleaning, not assumed to be clean
  by default.
- Several airport/city codes (e.g. `AMD`, `FRA`, `JAI`) were missing from
  the initial city-name mapping and were only found by inspecting the
  cleaned output's unique values after the first cleaning pass — a reminder
  to validate cleaning output rather than trust it blindly.

## Future Improvements

- Extend the mutual information analysis with SHAP values on a trained
  model (Part 2) to explain individual flight-level predictions, not just
  global feature importance.
- Incorporate real-time data (where available) to distinguish genuine
  historical pricing patterns from live inventory-driven price changes.
- Add an interactive Streamlit dashboard as a stretch goal, allowing a user
  to filter by route and see the relevant elasticity curve and variance
  band directly.

## Repository Structure

```
mic-3/
├── 01_cleaning_preprocessing.py     # Raw data cleaning script
├── 02_eda_visualizations.py         # Visualization + factor analysis script
├── flight_pricing_dataset.csv       # Raw dataset
├── flight_pricing_clean.csv         # Cleaned dataset (script output)
├── mutual_info_ranking.csv          # Quantitative factor ranking (script output)
├── chart1_ridgeline.png
├── chart2_elasticity.png
├── chart3_violin.png
├── chart4_heatmap.png
├── chart5_variance_band.png
├── chart6_sankey.html
├── chart7_mutual_info_bar.png
├── requirements.txt
└── README.md
```
