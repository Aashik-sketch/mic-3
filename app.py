from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.travel_analysis import (
    booking_time_analysis,
    build_recommendations,
    clean_flight_data,
    price_by_group,
    train_price_model,
)

st.set_page_config(page_title="AI Travel Analyst", page_icon="✈", layout="wide")

st.markdown("""
<style>
:root { --ink:#102a43; --muted:#627d98; --blue:#0b7285; --mint:#d8f3dc; --sand:#fff7ed; }
.block-container { max-width: 1400px; padding-top: 2rem; }
.hero { background: linear-gradient(120deg,#e6fffa 0%,#f0f9ff 52%,#fff7ed 100%); border:1px solid #cfe8e3; padding: 2.2rem 2.4rem; border-radius: 24px; margin-bottom: 1.4rem; }
.hero h1 { color: var(--ink); font-size: 3rem; letter-spacing:-.04em; margin:0; }
.hero p { color: var(--muted); font-size:1.08rem; max-width:760px; margin:.6rem 0 0; }
.insight { background: #f7fbfc; border-left: 4px solid var(--blue); padding: .9rem 1rem; border-radius: 8px; margin: .6rem 0; color:#243b53; }
[data-testid="stMetricValue"] { color: var(--ink); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>AI Travel Analyst</h1><p>Clean flight data, understand what moves fare prices, and turn historical patterns into practical travel decisions.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Start with your dataset")
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    st.caption("Expected target column: Price, Fare, Ticket_Price, or a similar name.")
    st.divider()
    st.markdown("**What this app covers**")
    st.markdown("1. Data quality and preprocessing\n2. Exploratory visualizations\n3. Explainable price prediction\n4. Cheapest booking-time analysis")

if uploaded is None:
    st.info("Upload the challenge flight-price dataset in the left panel to begin.")
    st.markdown("### Recommended dataset shape")
    st.dataframe(pd.DataFrame({"Airline": ["Example Air"], "Date_of_Journey": ["01/03/2019"], "Source": ["City A"], "Destination": ["City B"], "Duration": ["2h 40m"], "Total_Stops": ["non-stop"], "Price": [120]}), use_container_width=True, hide_index=True)
    st.stop()

try:
    raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
    cleaned, stats = clean_flight_data(raw)
except Exception as error:
    st.error(f"The file could not be processed: {error}")
    st.stop()

with st.spinner("Training an explainable fare model..."):
    try:
        model_result = train_price_model(cleaned)
    except Exception as error:
        st.error(f"The model needs more usable data: {error}")
        st.stop()

st.success(f"Ready. Cleaned {stats['clean_rows']:,} usable records from {stats['original_rows']:,} uploaded rows.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Usable flights", f"{len(cleaned):,}")
m2.metric("Average fare", f"{cleaned['price'].mean():,.0f}")
m3.metric("Median fare", f"{cleaned['price'].median():,.0f}")
m4.metric("Model R²", f"{model_result.metrics['r2']:.2f}")

explore_tab, model_tab, data_tab = st.tabs(["Explore patterns", "Predict & explain", "Cleaned data"])

with explore_tab:
    st.subheader("What drives the fare?")
    left, right = st.columns(2)
    with left:
        group_column = "airline" if "airline" in cleaned.columns else ("source" if "source" in cleaned.columns else None)
        if group_column:
            grouped = price_by_group(cleaned, group_column)
            fig = px.bar(grouped.sort_values("average_price"), x=group_column, y="average_price", color="average_price", color_continuous_scale="Teal", title=f"Average fare by {group_column.replace('_', ' ').title()}")
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Average fare")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No airline or source column was detected for this chart.")
    with right:
        time_data = booking_time_analysis(cleaned)
        if not time_data.empty:
            fig = px.line(time_data, x="time_window", y="average_price", markers=True, title="Advanced analysis: departure-time pricing", color_discrete_sequence=["#0b7285"])
            fig.update_layout(xaxis_title="Departure window", yaxis_title="Average fare")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("A departure-time column is needed for the advanced analysis.")

    st.subheader("Decision-ready insights")
    for recommendation in build_recommendations(cleaned):
        st.markdown(f'<div class="insight">{recommendation}</div>', unsafe_allow_html=True)

    if "price" in cleaned.columns:
        fig = px.histogram(cleaned, x="price", nbins=35, marginal="box", title="Fare distribution", color_discrete_sequence=["#f59f00"])
        fig.update_layout(xaxis_title="Fare", yaxis_title="Number of flights")
        st.plotly_chart(fig, use_container_width=True)

with model_tab:
    st.subheader("Fare prediction")
    st.caption("The model learns from the uploaded historical data. Change the inputs below to compare a scenario with the learned fare patterns.")
    input_data: dict[str, object] = {}
    input_columns = st.columns(3)
    for index, column in enumerate(model_result.feature_columns):
        with input_columns[index % 3]:
            series = cleaned[column]
            if pd.api.types.is_numeric_dtype(series):
                input_data[column] = st.number_input(column.replace("_", " ").title(), value=float(series.median()))
            else:
                options = sorted(series.astype(str).dropna().unique().tolist())
                input_data[column] = st.selectbox(column.replace("_", " ").title(), options, key=f"predict_{column}")
    prediction = model_result.pipeline.predict(pd.DataFrame([input_data]))[0]
    st.metric("Estimated fare", f"{prediction:,.0f}")

    st.subheader("Model quality")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean absolute error", f"{model_result.metrics['mae']:,.0f}")
    c2.metric("Root mean squared error", f"{model_result.metrics['rmse']:,.0f}")
    c3.metric("R² score", f"{model_result.metrics['r2']:.2f}")
    st.caption("Permutation importance shows how much model performance changes when each original feature is shuffled.")
    importance = model_result.feature_importance.head(12).sort_values("importance")
    fig = px.bar(importance, x="importance", y="feature", orientation="h", title="Key features behind predictions", color="importance", color_continuous_scale="Teal")
    st.plotly_chart(fig, use_container_width=True)

with data_tab:
    st.subheader("Cleaning report")
    st.json(stats)
    st.dataframe(cleaned.head(100), use_container_width=True, hide_index=True)
    csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned dataset", csv_bytes, "cleaned_flight_prices.csv", "text/csv")
