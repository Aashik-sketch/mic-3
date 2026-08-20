"""
AI Travel Analyst — Part 1: Exploration & Visualization
==========================================================
Run this AFTER 01_cleaning_preprocessing.py has produced
flight_pricing_clean.csv.

Environment notes:
- In VS Code: run `pip install pandas matplotlib seaborn scikit-learn plotly`
  once in your terminal (inside a virtual environment is recommended).
- In Colab: plotly is pre-installed, everything else too.

Produces 6 charts + a mutual-information factor ranking, all saved
to disk so you can drop them straight into your notebook / README.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

sns.set_style("whitegrid")

# ------------------------------------------------------------------
# 0. LOAD CLEANED DATA
# ------------------------------------------------------------------
df = pd.read_csv("flight_pricing_clean.csv")
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# ==================================================================
# CHART 1 — RIDGELINE: price distribution shape per airline
# ==================================================================
def chart1_ridgeline(df):
    plot_df = df.dropna(subset=["Airline", "Price"]).copy()
    top_airlines = plot_df["Airline"].value_counts().head(8).index
    plot_df = plot_df[plot_df["Airline"].isin(top_airlines)]
    plot_df["log_price"] = np.log1p(plot_df["Price"])

    fig, axes = plt.subplots(len(top_airlines), 1, figsize=(8, 10), sharex=True)
    palette = sns.color_palette("mako", len(top_airlines))

    for i, airline in enumerate(top_airlines):
        subset = plot_df[plot_df["Airline"] == airline]["log_price"]
        sns.kdeplot(subset, ax=axes[i], fill=True, color=palette[i], alpha=0.8, linewidth=1)
        axes[i].set_ylabel("")
        axes[i].set_yticks([])
        axes[i].text(-0.02, 0.2, airline, transform=axes[i].transAxes, ha="right", fontsize=9)
        axes[i].spines[["left", "right", "top"]].set_visible(False)

    axes[-1].set_xlabel("log(Price)")
    fig.suptitle("Price Distribution by Airline (Ridgeline)", y=0.92)
    plt.tight_layout()
    plt.savefig("chart1_ridgeline.png", dpi=150)
    plt.close()
    print("Saved chart1_ridgeline.png")


# ==================================================================
# CHART 2 — ELASTICITY CURVE: price vs days-before-departure
#            (the "panic zone")
# ==================================================================
def chart2_elasticity(df):
    plot_df = df.dropna(subset=["Days_Before_Departure", "Price", "Airline"]).copy()
    top_airlines = plot_df["Airline"].value_counts().head(6).index
    plot_df = plot_df[plot_df["Airline"].isin(top_airlines)]

    plot_df["days_bin"] = pd.cut(
        plot_df["Days_Before_Departure"],
        bins=[-1, 3, 7, 14, 30, 60, 90, 180],
        labels=["0-3", "4-7", "8-14", "15-30", "31-60", "61-90", "91-180"],
    )
    trend = plot_df.groupby(["Airline", "days_bin"], observed=True)["Price"].mean().reset_index()

    plt.figure(figsize=(9, 6))
    sns.lineplot(data=trend, x="days_bin", y="Price", hue="Airline", marker="o")
    plt.xlabel("Days Before Departure")
    plt.ylabel("Average Price")
    plt.title("Price Elasticity: How Fares Change as Departure Approaches")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("chart2_elasticity.png", dpi=150)
    plt.close()
    print("Saved chart2_elasticity.png")


# ==================================================================
# CHART 3 — VIOLIN: price by number of stops
# ==================================================================
def chart3_violin(df):
    plot_df = df.dropna(subset=["Total_Stops", "Price"]).copy()
    plot_df["Total_Stops"] = plot_df["Total_Stops"].astype(int)
    plot_df = plot_df[plot_df["Price"] < plot_df["Price"].quantile(0.95)]

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=plot_df, x="Total_Stops", y="Price", hue="Total_Stops",
                    palette="mako", legend=False)
    plt.xlabel("Number of Stops")
    plt.ylabel("Price (95th percentile clipped for readability)")
    plt.title("Price Distribution by Number of Stops")
    plt.tight_layout()
    plt.savefig("chart3_violin.png", dpi=150)
    plt.close()
    print("Saved chart3_violin.png")


# ==================================================================
# CHART 4 — CLUSTERED CORRELATION HEATMAP
# ==================================================================
def chart4_heatmap(df):
    numeric_cols = ["Price", "Distance_km", "Duration_min", "Total_Stops",
                     "Days_Before_Departure", "Passenger_Count", "Departure_Hour"]
    corr_df = df[numeric_cols].dropna()
    corr = corr_df.corr()

    g = sns.clustermap(corr, cmap="vlag", center=0, annot=True, fmt=".2f",
                        figsize=(8, 8), linewidths=0.5)
    g.fig.suptitle("Clustered Correlation Heatmap (Numeric Features)", y=1.02)
    g.savefig("chart4_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved chart4_heatmap.png")


# ==================================================================
# CHART 5 — VARIANCE BAND: price spread within each booking window
#            (explains why a cheaper fare can appear after booking)
# ==================================================================
def chart5_variance_band(df):
    plot_df = df.dropna(subset=["Days_Before_Departure", "Price"]).copy()
    plot_df["days_bin"] = pd.cut(
        plot_df["Days_Before_Departure"],
        bins=[-1, 3, 7, 14, 30, 60, 90, 180],
        labels=["0-3", "4-7", "8-14", "15-30", "31-60", "61-90", "91-180"],
    )
    plot_df = plot_df[plot_df["Price"] < plot_df["Price"].quantile(0.97)]

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=plot_df, x="days_bin", y="Price", color="#4C72B0")
    plt.xlabel("Days Before Departure")
    plt.ylabel("Price (97th percentile clipped)")
    plt.title("Price Variance Within Each Booking Window\n"
               "(Why a Cheaper Fare Can Still Appear After You Book)")
    plt.tight_layout()
    plt.savefig("chart5_variance_band.png", dpi=150)
    plt.close()
    print("Saved chart5_variance_band.png")

    # print the actual spread numbers for your insights writeup
    spread = plot_df.groupby("days_bin", observed=True)["Price"].agg(["mean", "std"])
    print("\nPrice spread per booking window:\n", spread)


# ==================================================================
# CHART 6 — SANKEY: route flow colored by average price
#            (requires plotly — works in Colab out of the box;
#             in VS Code run: pip install plotly)
# ==================================================================
def chart6_sankey(df):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — skipping Sankey. "
              "Run: pip install plotly")
        return

    plot_df = df.dropna(subset=["Source", "Destination", "Price"]).copy()
    route_stats = (
        plot_df.groupby(["Source", "Destination"])
        .agg(count=("Price", "size"), avg_price=("Price", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(12)  # top 12 routes for readability
    )

    sources = route_stats["Source"].tolist()
    destinations = route_stats["Destination"].tolist()
    all_nodes = list(pd.unique(sources + destinations))
    node_index = {name: i for i, name in enumerate(all_nodes)}

    # color links by price: darker/redder = more expensive
    max_price = route_stats["avg_price"].max()
    link_colors = [
        f"rgba({int(255 * (p / max_price))}, 80, {int(255 * (1 - p / max_price))}, 0.6)"
        for p in route_stats["avg_price"]
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=all_nodes,
            color="#4C72B0",
        ),
        link=dict(
            source=[node_index[s] for s in sources],
            target=[node_index[d] for d in destinations],
            value=route_stats["count"].tolist(),
            color=link_colors,
            customdata=route_stats["avg_price"].round(0).tolist(),
            hovertemplate="%{source.label} → %{target.label}<br>"
                          "Flights: %{value}<br>Avg Price: ₹%{customdata}<extra></extra>",
        ),
    )])
    fig.update_layout(title_text="Top Routes by Flight Volume, Colored by Average Price",
                       font_size=11)
    fig.write_html("chart6_sankey.html")
    print("Saved chart6_sankey.html (open in browser)")


# ==================================================================
# MUTUAL INFORMATION — quantitative factor ranking
#   (backs up "identify major factors" with numbers, not eyeballing)
# ==================================================================
def mutual_info_ranking(df):
    feature_cols = ["Airline", "Source", "Destination", "Total_Stops", "Distance_km",
                     "Travel_Class", "Days_Before_Departure", "Season", "Weekday",
                     "Booking_Channel", "Passenger_Count", "Duration_min", "Departure_Hour"]

    model_df = df[feature_cols + ["Price"]].dropna()
    X = model_df[feature_cols].copy()
    y = model_df["Price"]

    categorical_cols = X.select_dtypes(include="str").columns
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    discrete_mask = [col in categorical_cols or col == "Total_Stops" for col in X.columns]

    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=42)
    mi_df = pd.DataFrame({"Feature": feature_cols, "Mutual_Info": mi_scores})
    mi_df = mi_df.sort_values("Mutual_Info", ascending=False)

    print("\nMutual Information Ranking (higher = stronger driver of price):")
    print(mi_df.to_string(index=False))
    mi_df.to_csv("mutual_info_ranking.csv", index=False)

    # quick bar chart version for the notebook
    plt.figure(figsize=(8, 6))
    sns.barplot(data=mi_df, x="Mutual_Info", y="Feature", hue="Feature",
                palette="mako", legend=False)
    plt.title("Feature Importance via Mutual Information")
    plt.xlabel("Mutual Information Score")
    plt.tight_layout()
    plt.savefig("chart7_mutual_info_bar.png", dpi=150)
    plt.close()
    print("Saved chart7_mutual_info_bar.png")

    return mi_df


# ==================================================================
# RUN EVERYTHING
# ==================================================================
if __name__ == "__main__":
    chart1_ridgeline(df)
    chart2_elasticity(df)
    chart3_violin(df)
    chart4_heatmap(df)
    chart5_variance_band(df)
    chart6_sankey(df)
    mutual_info_ranking(df)
    print("\nAll charts generated. Check the PNG/HTML files in this folder.")
