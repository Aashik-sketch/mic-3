from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

COLUMN_ALIASES = {
    "airline": ["airline", "carrier", "operator"],
    "date_of_journey": ["date_of_journey", "journey_date", "flight_date", "date"],
    "source": ["source", "origin", "from", "departure_city", "departure"],
    "destination": ["destination", "dest", "to", "arrival_city", "arrival"],
    "route": ["route", "itinerary", "flight_route"],
    "dep_time": ["dep_time", "departure_time", "departure_datetime"],
    "arrival_time": ["arrival_time", "arr_time", "arrival_datetime"],
    "duration": ["duration", "travel_duration", "flight_duration"],
    "total_stops": ["total_stops", "stops", "number_of_stops", "stop_count"],
    "additional_info": ["additional_info", "additional_information", "notes"],
    "price": ["price", "fare", "ticket_price", "flight_price", "amount", "target"],
}

@dataclass
class ModelResult:
    pipeline: Pipeline
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    feature_columns: list[str]

def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

def standardise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [_normalise_name(column) for column in data.columns]
    rename_map: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_name = _normalise_name(alias)
            if alias_name in data.columns:
                rename_map[alias_name] = canonical
                break
    return data.rename(columns=rename_map)

def _parse_duration(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).lower().strip()
    hours = re.search(r"(\d+(?:\.\d+)?)\s*h", text)
    minutes = re.search(r"(\d+(?:\.\d+)?)\s*m", text)
    if hours or minutes:
        total = 0.0
        if hours:
            total += float(hours.group(1))
        if minutes:
            total += float(minutes.group(1)) / 60
        return total
    if ":" in text:
        parts = text.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60
        except (ValueError, IndexError):
            return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan

def _parse_stops(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).lower().strip()
    if "non-stop" in text or "non stop" in text or text in {"0", "direct"}:
        return 0.0
    match = re.search(r"(\d+)", text)
    return float(match.group(1)) + 1 if match and "stop" in text else (float(match.group(1)) if match else np.nan)

def _extract_hour(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d{1,2})", str(value))
    return float(match.group(1)) if match else np.nan

def clean_flight_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = standardise_columns(frame)
    original_rows = len(data)
    data = data.drop_duplicates().reset_index(drop=True)

    if "price" not in data.columns:
        raise ValueError("The dataset needs a price column, such as Price, Fare, or Ticket_Price.")

    for column in ["date_of_journey", "dep_time", "arrival_time"]:
        if column in data.columns:
            data[column] = data[column].astype(str).str.strip()

    if "date_of_journey" in data.columns:
        parsed_dates = pd.to_datetime(data["date_of_journey"], dayfirst=True, errors="coerce")
        data["journey_day"] = parsed_dates.dt.day
        data["journey_month"] = parsed_dates.dt.month
        data["journey_weekday"] = parsed_dates.dt.dayofweek
        data["journey_date"] = parsed_dates

    if "dep_time" in data.columns:
        data["departure_hour"] = data["dep_time"].map(_extract_hour)
    if "arrival_time" in data.columns:
        data["arrival_hour"] = data["arrival_time"].map(_extract_hour)
    if "duration" in data.columns:
        data["duration_hours"] = data["duration"].map(_parse_duration)
    if "total_stops" in data.columns:
        data["stops_count"] = data["total_stops"].map(_parse_stops)

    data["price"] = pd.to_numeric(data["price"].astype(str).str.replace(r"[^0-9.-]", "", regex=True), errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["price"])
    data = data[data["price"] > 0].copy()

    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].fillna("Unknown").astype(str).str.strip()
    for column in data.select_dtypes(include=np.number).columns:
        if column != "price":
            data[column] = data[column].fillna(data[column].median())

    stats = {
        "original_rows": original_rows,
        "clean_rows": len(data),
        "duplicates_removed": original_rows - len(frame.drop_duplicates()),
        "missing_values_after_cleaning": int(data.isna().sum().sum()),
        "columns": list(data.columns),
    }
    return data.reset_index(drop=True), stats

def _usable_features(data: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    ignored = {"price", "journey_date", "date_of_journey", "dep_time", "arrival_time", "duration", "total_stops", "route"}
    features = [column for column in data.columns if column not in ignored]
    features = [column for column in features if data[column].nunique(dropna=False) > 1]
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(data[column])]
    categorical = [column for column in features if column not in numeric]
    return features, numeric, categorical

def train_price_model(data: pd.DataFrame, random_state: int = 42) -> ModelResult:
    features, numeric, categorical = _usable_features(data)
    if len(data) < 10 or not features:
        raise ValueError("At least 10 usable rows and one predictive feature are required to train the model.")

    x_train, x_test, y_train, y_test = train_test_split(
        data[features], data["price"], test_size=0.2, random_state=random_state
    )
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", transformer),
            ("model", RandomForestRegressor(n_estimators=250, random_state=random_state, n_jobs=-1, min_samples_leaf=2)),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    importance = permutation_importance(
        pipeline, x_test, y_test, n_repeats=5, random_state=random_state, n_jobs=-1
    )
    importance_frame = pd.DataFrame(
        {"feature": features, "importance": importance.importances_mean}
    ).sort_values("importance", ascending=False)
    return ModelResult(pipeline, metrics, importance_frame, features)

def price_by_group(data: pd.DataFrame, column: str, minimum_rows: int = 3) -> pd.DataFrame:
    if column not in data.columns:
        return pd.DataFrame(columns=[column, "average_price", "median_price", "flights"])
    result = (
        data.groupby(column, dropna=False)["price"]
        .agg(average_price="mean", median_price="median", flights="size")
        .reset_index()
    )
    return result[result["flights"] >= minimum_rows].sort_values("average_price", ascending=False)

def booking_time_analysis(data: pd.DataFrame) -> pd.DataFrame:
    if "departure_hour" not in data.columns:
        return pd.DataFrame(columns=["time_window", "average_price", "flights", "saving_vs_peak"])
    bins = [-1, 5, 11, 16, 20, 24]
    labels = ["Late night", "Morning", "Afternoon", "Evening", "Night"]
    working = data.copy()
    working["time_window"] = pd.cut(working["departure_hour"], bins=bins, labels=labels)
    result = working.groupby("time_window", observed=False)["price"].agg(average_price="mean", flights="size").reset_index()
    peak = result["average_price"].max() if not result.empty else 0
    result["saving_vs_peak"] = peak - result["average_price"]
    return result.dropna(subset=["average_price"])

def build_recommendations(data: pd.DataFrame) -> list[str]:
    recommendations: list[str] = []
    by_airline = price_by_group(data, "airline")
    if not by_airline.empty:
        cheapest = by_airline.sort_values("average_price").iloc[0]
        recommendations.append(f"Compare {cheapest['airline']} first: it has the lowest average fare in this dataset ({cheapest['average_price']:.0f}).")
    time_data = booking_time_analysis(data)
    if not time_data.empty:
        cheapest_window = time_data.sort_values("average_price").iloc[0]
        recommendations.append(f"The {str(cheapest_window['time_window']).lower()} departure window is the cheapest on average, saving about {cheapest_window['saving_vs_peak']:.0f} versus the most expensive window.")
    if "stops_count" in data.columns:
        direct = data.loc[data["stops_count"] == 0, "price"].mean()
        connecting = data.loc[data["stops_count"] > 0, "price"].mean()
        if pd.notna(direct) and pd.notna(connecting):
            direction = "direct flights cost more" if direct > connecting else "connecting flights cost more"
            recommendations.append(f"Check the stop count carefully: {direction} on average in this dataset.")
    recommendations.append("Use the model estimate as a comparison signal, not a guaranteed quote; airline availability and demand can change the final fare.")
    return recommendations
