"""
AI Travel Analyst — Part 1: Data Cleaning & Preprocessing
============================================================
Cleans the raw flight_pricing_dataset.csv and produces a clean,
analysis-ready dataframe saved to flight_pricing_clean.csv

Run this in Google Colab or locally. Just update RAW_PATH below.
"""

import pandas as pd
import numpy as np
import re

pd.set_option('display.max_columns', None)

# ------------------------------------------------------------------
# 0. LOAD
# ------------------------------------------------------------------
RAW_PATH = "flight_pricing_dataset.csv"   # <- change path if needed
df = pd.read_csv(RAW_PATH)

print(f"Raw shape: {df.shape}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Null counts:\n{df.isnull().sum()}")

# ------------------------------------------------------------------
# 1. DROP EXACT DUPLICATES
# ------------------------------------------------------------------
df = df.drop_duplicates()

# ------------------------------------------------------------------
# 2. STANDARDIZE AIRLINE NAMES
#    (case inconsistency: 'AIR INDIA', 'air india', 'Air India' etc.)
# ------------------------------------------------------------------
df['Airline'] = df['Airline'].str.strip().str.title()
# fix known multi-word edge cases title() can mangle (none needed here,
# but keep the pattern in case new airlines appear)
airline_fix_map = {
    "Airasia India": "AirAsia India",
}
df['Airline'] = df['Airline'].replace(airline_fix_map)

# ------------------------------------------------------------------
# 3. STANDARDIZE CITY / AIRPORT CODE NAMES (Source & Destination)
#    Data mixes: full city name, "<City> Airport", and IATA codes.
#    Map everything to one canonical city name.
# ------------------------------------------------------------------
code_to_city = {
    "HYD": "Hyderabad", "BOM": "Mumbai", "DEL": "Delhi", "BLR": "Bangalore",
    "PNQ": "Pune", "GOI": "Goa", "CCU": "Kolkata", "MAA": "Chennai",
    "DXB": "Dubai", "DOH": "Doha", "SIN": "Singapore", "BKK": "Bangkok",
    "JFK": "New York", "LHR": "London", "SYD": "Sydney",
    "AMD": "Ahmedabad", "FRA": "Frankfurt", "JAI": "Jaipur",
}

def canonical_city(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip()
    # strip "Airport" suffix e.g. "Bangalore Airport" -> "Bangalore"
    v = re.sub(r"\s*Airport$", "", v, flags=re.IGNORECASE).strip()
    # map IATA code to city name if it matches (exact, case-insensitive)
    upper = v.upper()
    if upper in code_to_city:
        return code_to_city[upper]
    return v.title()

df['Source'] = df['Source'].apply(canonical_city)
df['Destination'] = df['Destination'].apply(canonical_city)

# ------------------------------------------------------------------
# 4. TOTAL_STOPS -> integer
#    Mixed formats: '0', 'non-stop', '1 stop', '1', '2 stops', '2'
# ------------------------------------------------------------------
def parse_stops(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if v in ("non-stop", "nonstop", "0"):
        return 0
    match = re.search(r"\d+", v)
    return int(match.group()) if match else np.nan

df['Total_Stops'] = df['Total_Stops'].apply(parse_stops)

# ------------------------------------------------------------------
# 5. DURATION -> convert everything to duration in minutes
#    Mixed formats: '1.67' (hours, decimal), '0h 45m', '177 min'
# ------------------------------------------------------------------
def parse_duration_to_minutes(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()

    # format: "177 min"
    m = re.match(r"^(\d+(\.\d+)?)\s*min$", v)
    if m:
        return float(m.group(1))

    # format: "16h 47m" or "0h 45m" or "3h 20m"
    m = re.match(r"^(\d+)h\s*(\d+)?m?$", v)
    if m:
        hours = int(m.group(1))
        mins = int(m.group(2)) if m.group(2) else 0
        return hours * 60 + mins

    # format: plain decimal hours e.g. "1.67", "14.80"
    m = re.match(r"^\d+(\.\d+)?$", v)
    if m:
        return float(v) * 60

    return np.nan

df['Duration_min'] = df['Duration'].apply(parse_duration_to_minutes)
df = df.drop(columns=['Duration'])

# ------------------------------------------------------------------
# 6. DISTANCE_KM -> numeric
#    Some rows carry a "km" unit suffix e.g. "298.6 km" instead of
#    a plain number — strip that before converting.
# ------------------------------------------------------------------
df['Distance_km'] = (
    df['Distance_km'].astype(str)
    .str.replace(r"\s*km$", "", regex=True, case=False)
    .str.strip()
)
df['Distance_km'] = pd.to_numeric(df['Distance_km'], errors='coerce')

# ------------------------------------------------------------------
# 7. PASSENGER_COUNT -> numeric (mixed digits and number-words)
# ------------------------------------------------------------------
word_to_num = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
}

def parse_passenger_count(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if v in word_to_num:
        return word_to_num[v]
    try:
        return int(v)
    except ValueError:
        return np.nan

df['Passenger_Count'] = df['Passenger_Count'].apply(parse_passenger_count)

# ------------------------------------------------------------------
# 8. DAYS_BEFORE_DEPARTURE & PRICE -> numeric
# ------------------------------------------------------------------
df['Days_Before_Departure'] = pd.to_numeric(df['Days_Before_Departure'], errors='coerce')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# ------------------------------------------------------------------
# 9. DEPARTURE_DATE -> datetime, derive Month
#    (Weekday column already exists but we can cross-check/derive too)
# ------------------------------------------------------------------
df['Departure_Date'] = pd.to_datetime(df['Departure_Date'], errors='coerce')
df['Month'] = df['Departure_Date'].dt.month_name()

# ------------------------------------------------------------------
# 10. DEPARTURE_TIME & ARRIVAL_TIME -> normalize to 24h time
#     Mixed formats: '8:10 PM', '12:10' (24h already), '07:05'
# ------------------------------------------------------------------
def parse_time(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip()
    for fmt in ("%I:%M %p", "%H:%M"):
        try:
            return pd.to_datetime(v, format=fmt).strftime("%H:%M")
        except ValueError:
            continue
    return np.nan

df['Departure_Time'] = df['Departure_Time'].apply(parse_time)
df['Arrival_Time'] = df['Arrival_Time'].apply(parse_time)

# derive departure hour as numeric feature (useful for viz/model later)
df['Departure_Hour'] = pd.to_datetime(
    df['Departure_Time'], format="%H:%M", errors='coerce'
).dt.hour

# ------------------------------------------------------------------
# 11. CATEGORICAL CLEANUP (strip whitespace / fix casing consistently)
# ------------------------------------------------------------------
categorical_cols = ['Travel_Class', 'Season', 'Weekday', 'Aircraft_Type', 'Booking_Channel']
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip().str.title()
    df[col] = df[col].replace('Nan', np.nan)   # str.title() turns NaN -> 'Nan'

# fill missing Season/Weekday from Departure_Date where possible
season_map = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Autumn", 11: "Autumn",
}
derived_season = df['Departure_Date'].dt.month.map(season_map)
df['Season'] = df['Season'].fillna(derived_season)
derived_weekday = df['Departure_Date'].dt.day_name()
df['Weekday'] = df['Weekday'].fillna(derived_weekday)

# ------------------------------------------------------------------
# 12. HANDLE REMAINING MISSING VALUES
#     Strategy: drop rows where Price (target) is missing —
#     can't use them for analysis or modeling.
#     For other columns, keep NaN visible rather than silently
#     imputing, since Part 1 should report data quality honestly.
# ------------------------------------------------------------------
before = len(df)
df = df.dropna(subset=['Price'])
print(f"\nDropped {before - len(df)} rows with missing Price (target variable).")

# ------------------------------------------------------------------
# 13. OUTLIER FLAGGING (IQR method) — flag, don't delete
#     Kept deliberately, since anomalies (e.g. surprisingly cheap
#     long-haul flights) are a genuinely interesting analysis angle,
#     not just noise to remove.
# ------------------------------------------------------------------
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Price_Outlier'] = (df['Price'] < lower_bound) | (df['Price'] > upper_bound)
print(f"Flagged {df['Price_Outlier'].sum()} price outliers "
      f"(bounds: {lower_bound:.0f} to {upper_bound:.0f}) — kept, not removed.")

# ------------------------------------------------------------------
# 14. FINAL SANITY CHECKS
# ------------------------------------------------------------------
print(f"\nFinal shape: {df.shape}")
print(f"\nRemaining nulls per column:\n{df.isnull().sum()}")
print(f"\nDtypes:\n{df.dtypes}")

# ------------------------------------------------------------------
# 15. SAVE CLEAN DATASET
# ------------------------------------------------------------------
df.to_csv("flight_pricing_clean.csv", index=False)
print("\nSaved cleaned dataset -> flight_pricing_clean.csv")
