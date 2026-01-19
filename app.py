import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="UIDAI Digital Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Load Data
# -----------------------
DATA_DIR = Path("data")

@st.cache_data
def load_data():
    state_df = pd.read_csv(DATA_DIR / "state_summary.csv")
    monthly_df = pd.read_csv(DATA_DIR / "monthly_trend.csv")
    forecast_df = pd.read_csv(DATA_DIR / "forecast.csv")
    growth_df = pd.read_csv(DATA_DIR / "growth_size.csv")
    dsi_df = pd.read_csv(DATA_DIR / "demand_stress_index.csv")
    return state_df, monthly_df, forecast_df, growth_df, dsi_df

state_df, monthly_df, forecast_df, growth_df, dsi_df = load_data()

# -----------------------
# Column Safety Layer
# -----------------------

# ---- Growth column detection
if "growth_rate" in state_df.columns:
    growth_col = "growth_rate"
elif "growth" in state_df.columns:
    growth_col = "growth"
else:
    growth_col = None

# ---- Size column detection
if "size_millions" in growth_df.columns:
    size_col = "size_millions"
elif "size" in growth_df.columns:
    size_col = "size"
else:
    size_col = None

# -----------------------
# Sidebar Controls
# -----------------------
st.sidebar.title("🎛 Dashboard Controls")

states = ["All"] + sorted(state_df["state"].unique().tolist())
selected_state = st.sidebar.selectbox("📍 Select State", states)

top_n = st.sidebar.slider(
    "🏆 Top N States (DSI Ranking)",
    min_value=5,
    max_value=20,
    value=10
)

st.sidebar.info("Use filters to explore regional risk, growth and demand patterns.")

# -----------------------
# Filtering
# -----------------------
if selected_state == "All":
    filtered_state_df = state_df.copy()
else:
    filtered_state_df = state_df[state_df["state"] == selected_state]

# -----------------------
# Header
# -----------------------
st.title("🇮🇳 UIDAI Aadhaar Digital Intelligence Dashboard")
st.caption("Data-driven insights for national digital infrastructure planning")

# -----------------------
# National Summary
# -----------------------
st.subheader("📊 National Summary")

col1, col2, col3 = st.columns(3)

total_states = state_df["state"].nunique()
total_updates = state_df["total_updates"].sum() / 1_000_000

if growth_col:
    avg_growth = state_df[growth_col].mean()
else:
    avg_growth = 0

col1.metric("Total States", total_states)
col2.metric("Total Updates (Millions)", f"{total_updates:.1f}M")
col3.metric("Avg Growth Rate", f"{avg_growth:.2f}")

st.divider()

# -----------------------
# Monthly Trend Chart
# -----------------------
st.subheader("📈 National Monthly Update Trend")

fig_trend = px.line(
    monthly_df,
    x="month",
    y="total_updates",
    markers=True,
    title="Monthly Aadhaar Updates Trend"
)

fig_trend.update_layout(height=420)
st.plotly_chart(fig_trend, use_container_width=True)

# -----------------------
# Forecast Chart
# -----------------------
st.subheader("🔮 Short-Term Forecast")

fig_forecast = px.line(
    forecast_df,
    x="month",
    y="predicted_updates",
    markers=True,
    title="Predicted Aadhaar Updates (Next Months)"
)

fig_forecast.update_layout(height=420)
st.plotly_chart(fig_forecast, use_container_width=True)

# -----------------------
# Growth vs Demand Bubble Map
# -----------------------
st.subheader("🌍 Growth vs Demand Scale Map")

if size_col and "growth" in growth_df.columns:
    fig_bubble = px.scatter(
        growth_df,
        x="growth",
        y=size_col,
        size=size_col,
        hover_name="state",
        title="State Digital Momentum Map",
        size_max=40
    )
    fig_bubble.update_layout(height=520)
    st.plotly_chart(fig_bubble, use_container_width=True)
else:
    st.warning("Growth vs Demand visualization unavailable due to missing columns.")

# -----------------------
# Demand Stress Index Ranking
# -----------------------
st.subheader("🚨 Demand Stress Index (Top Risk States)")

top_dsi = dsi_df.sort_values("DSI", ascending=False).head(top_n)
st.dataframe(top_dsi, use_container_width=True)

# -----------------------
# AI Generated Insights
# -----------------------
st.subheader("🤖 AI Generated Insights")

if growth_col:
    fastest_growth_state = state_df.sort_values(
        growth_col, ascending=False
    ).iloc[0]["state"]
else:
    fastest_growth_state = "Not Available"

highest_dsi_state = dsi_df.sort_values("DSI", ascending=False).iloc[0]["state"]
highest_volume_state = state_df.sort_values("total_updates", ascending=False).iloc[0]["state"]

insight_text = f"""
✅ **Fastest Growing State:** {fastest_growth_state}  
🔥 **Highest Demand Stress:** {highest_dsi_state}  
📊 **Highest Transaction Volume:** {highest_volume_state}  

📌 **Strategic Insight:**  
States with high growth and high stress should be prioritized for infrastructure scaling, 
capacity upgrades, and proactive monitoring to avoid service congestion.
"""

st.info(insight_text)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("Built for UIDAI Hackathon | Data Pipeline + ML Forecast + Decision Intelligence")
