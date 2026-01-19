# =========================================================
# UIDAI DIGITAL ACTIVITY ANALYTICS PIPELINE (FINAL VERSION)
# Author: Praneet Kamble
# =========================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n🚀 UIDAI DATA PIPELINE STARTED")

# -------------------------------
# Paths
# -------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# -------------------------------
# 1. Load CSVs
# -------------------------------
csv_files = list(DATA_DIR.glob("api_data_aadhar_demographic_*.csv"))
if not csv_files:
    raise ValueError("❌ No demographic CSV files found in /data folder!")

dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

print(f"✅ Loaded {len(df):,} rows from {len(csv_files)} files")

# -------------------------------
# 2. Cleaning
# -------------------------------
df.columns = df.columns.str.strip()

df["state"] = (
    df["state"].astype(str)
    .str.strip()
    .str.title()
)

df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"])

df["demo_age_5_17"] = pd.to_numeric(df["demo_age_5_17"], errors="coerce").fillna(0)
df["demo_age_17_"] = pd.to_numeric(df["demo_age_17_"], errors="coerce").fillna(0)

df["total_updates"] = df["demo_age_5_17"] + df["demo_age_17_"]
df["month"] = df["date"].dt.to_period("M").astype(str)

print("✅ Data cleaning completed")

# -------------------------------
# 3. State Summary + Maturity
# -------------------------------
state_summary = (
    df.groupby("state")[["demo_age_5_17", "demo_age_17_", "total_updates"]]
    .sum()
    .reset_index()
)

# Safe maturity calculation
epsilon = 100
state_summary["maturity_ratio"] = (
    state_summary["demo_age_17_"] /
    (state_summary["demo_age_5_17"] + epsilon)
).round(2)

state_summary["maturity_ratio"] = state_summary["maturity_ratio"].clip(upper=50)

state_summary.to_csv(DATA_DIR / "state_summary.csv", index=False)
print("📄 Exported: state_summary.csv")

# -------------------------------
# 4. Monthly Trend (Chronological)
# -------------------------------
monthly_trend = (
    df.groupby("month")["total_updates"]
    .sum()
    .reset_index()
)

monthly_trend["month_sort"] = pd.to_datetime(monthly_trend["month"])
monthly_trend = monthly_trend.sort_values("month_sort").reset_index(drop=True)
monthly_trend.drop(columns="month_sort", inplace=True)

monthly_trend.to_csv(DATA_DIR / "monthly_trend.csv", index=False)
print("📄 Exported: monthly_trend.csv")

# -------------------------------
# 5. Feature Engineering for ML
# -------------------------------
monthly_trend["t"] = range(len(monthly_trend))
monthly_trend["lag_1"] = monthly_trend["total_updates"].shift(1)
monthly_trend["rolling_3"] = monthly_trend["total_updates"].rolling(3).mean()

monthly_trend = monthly_trend.dropna().reset_index(drop=True)

print("\n--- ML Feature Table Preview ---")
print(monthly_trend.head())

# -------------------------------
# 6. Linear Forecast (Baseline)
# -------------------------------
X_lr = monthly_trend[["t"]]
y_lr = monthly_trend["total_updates"]

lr_model = LinearRegression()
lr_model.fit(X_lr, y_lr)

future_t = np.array([
    len(monthly_trend),
    len(monthly_trend) + 1,
    len(monthly_trend) + 2
]).reshape(-1, 1)

future_predictions = lr_model.predict(future_t).clip(min=0).astype(int)

forecast_df = pd.DataFrame({
    "month": ["Next +1", "Next +2", "Next +3"],
    "predicted_updates": future_predictions
})

forecast_df.to_csv(DATA_DIR / "forecast.csv", index=False)
print("📄 Exported: forecast.csv")

# -------------------------------
# 7. State Growth Calculation
# -------------------------------
monthly_state = (
    df.groupby(["state", "month"])["total_updates"]
    .sum()
    .reset_index()
)

monthly_state["month_sort"] = pd.to_datetime(monthly_state["month"])
monthly_state = monthly_state.sort_values(["state", "month_sort"])

monthly_state["prev"] = monthly_state.groupby("state")["total_updates"].shift(1)
monthly_state["growth_rate"] = (
    (monthly_state["total_updates"] - monthly_state["prev"]) /
    (monthly_state["prev"] + 1)
)

state_growth = (
    monthly_state.groupby("state")["growth_rate"]
    .mean()
    .fillna(0)
    .reset_index()
)

state_summary = state_summary.merge(state_growth, on="state", how="left")

# -------------------------------
# 8. Growth vs Scale Dataset
# -------------------------------
growth_df = state_summary.copy()
growth_df["size_millions"] = growth_df["total_updates"] / 1_000_000
growth_df["growth"] = growth_df["growth_rate"]

growth_df[["state", "growth", "size_millions"]].to_csv(
    DATA_DIR / "growth_size.csv", index=False
)
print("📄 Exported: growth_size.csv")

# -------------------------------
# 9. Demand Stress Index (DSI)
# -------------------------------
dsi_df = state_summary.copy()

dsi_df["norm_size"] = dsi_df["total_updates"] / dsi_df["total_updates"].max()
dsi_df["norm_growth"] = (
    (dsi_df["growth_rate"] - dsi_df["growth_rate"].min()) /
    (dsi_df["growth_rate"].max() - dsi_df["growth_rate"].min() + 1e-6)
)
dsi_df["norm_maturity"] = dsi_df["maturity_ratio"] / dsi_df["maturity_ratio"].max()

dsi_df["DSI"] = (
    0.5 * dsi_df["norm_size"] +
    0.3 * dsi_df["norm_growth"] +
    0.2 * (1 - dsi_df["norm_maturity"])
)

# Early Warning Classification
def classify_risk(dsi):
    if dsi >= 0.65:
        return "CRITICAL"
    elif dsi >= 0.45:
        return "WARNING"
    else:
        return "STABLE"

dsi_df["risk_level"] = dsi_df["DSI"].apply(classify_risk)

dsi_df[[
    "state", "total_updates", "maturity_ratio",
    "growth_rate", "DSI", "risk_level"
]].sort_values("DSI", ascending=False).to_csv(
    DATA_DIR / "demand_stress_index.csv", index=False
)

print("📄 Exported: demand_stress_index.csv")

# -------------------------------
# 10. ML Model Training & Accuracy
# -------------------------------
print("\n--- Training ML Model (Random Forest) ---")

features = ["t", "lag_1", "rolling_3"]
X = monthly_trend[features]
y = monthly_trend["total_updates"]

# Time-based split
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Model Accuracy ---")
print("MAE :", int(mae))
print("RMSE:", int(rmse))

print("\n✅ PIPELINE COMPLETED SUCCESSFULLY 🚀")
