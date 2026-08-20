import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# Set page config
st.set_page_config(
    page_title="AI Sales Forecasting & Analytics System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Enterprise AI Sales Forecasting & Analytics System")
st.write("Real-time revenue analytics, custom data uploader, predictive modeling, and inventory alert engine.")

# Load pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_sales_model.pkl")

model = load_model()

# --- CUSTOM DATASET UPLOADER ---
st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Custom Sales CSV", type=["csv"])
st.sidebar.caption("💡 Supports custom CSVs with date and revenue/sales columns. Falls back to default dataset automatically.")

@st.cache_data
def process_data(file):
    if file is not None:
        raw_df = pd.read_csv(file)
    else:
        raw_df = pd.read_csv("Data/cleaned_sales.csv")
    
    # Standardize column names (lowercase, stripped spaces)
    raw_df.columns = raw_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Dynamic revenue/sales column detection
    sales_col_candidates = [
        'total_revenue', 'sales', 'revenue', 'amount', 
        'total_sales', 'total_amount', 'price'
    ]
    found_sales_col = None
    for col in sales_col_candidates:
        if col in raw_df.columns:
            found_sales_col = col
            break
            
    # Fallback: calculate sales if units_sold and unit_price exist
    if found_sales_col is None and 'units_sold' in raw_df.columns and 'unit_price' in raw_df.columns:
        raw_df['sales'] = raw_df['units_sold'] * raw_df['unit_price']
        found_sales_col = 'sales'
    elif found_sales_col is None:
        st.error(f"Could not find a sales or revenue column in your CSV. Available columns: {list(raw_df.columns)}")
        st.stop()
    else:
        raw_df = raw_df.rename(columns={found_sales_col: 'sales'})

    # Map item_type to category if category isn't explicitly present
    if 'item_type' in raw_df.columns and 'category' not in raw_df.columns:
        raw_df = raw_df.rename(columns={'item_type': 'category'})
    
    # Flexible date parsing to handle mixed formats automatically
    raw_df['order_date'] = pd.to_datetime(raw_df['order_date'], format='mixed', errors='coerce')
    raw_df = raw_df.dropna(subset=['order_date'])
    
    # Aggregate daily sales
    daily_sales = raw_df.groupby('order_date')['sales'].sum().reset_index()
    daily_sales = daily_sales.sort_values('order_date').reset_index(drop=True)

    # Re-engineer temporal features on the fly
    daily_sales['year'] = daily_sales['order_date'].dt.year
    daily_sales['month'] = daily_sales['order_date'].dt.month
    daily_sales['day'] = daily_sales['order_date'].dt.day
    daily_sales['dayofweek'] = daily_sales['order_date'].dt.dayofweek
    daily_sales['quarter'] = daily_sales['order_date'].dt.quarter
    daily_sales['is_weekend'] = daily_sales['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    daily_sales['sales_lag_1'] = daily_sales['sales'].shift(1)
    daily_sales['sales_lag_7'] = daily_sales['sales'].shift(7)
    daily_sales['sales_lag_30'] = daily_sales['sales'].shift(30)

    daily_sales['rolling_mean_7'] = daily_sales['sales'].shift(1).rolling(window=7).mean()
    daily_sales['rolling_mean_30'] = daily_sales['sales'].shift(1).rolling(window=30).mean()

    daily_sales = daily_sales.dropna().reset_index(drop=True)
    return daily_sales, raw_df

df, cleaned_raw_df = process_data(uploaded_file)

if uploaded_file is not None:
    st.sidebar.success("Custom CSV loaded successfully!")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Dashboard Controls")

# Category Filter
if 'category' in cleaned_raw_df.columns:
    categories = ['All'] + sorted(cleaned_raw_df['category'].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Select Product Category", categories)
    
    if selected_category != 'All':
        st.sidebar.info(f"Filtering for: **{selected_category}**")
        cat_raw = cleaned_raw_df[cleaned_raw_df['category'] == selected_category]
        cat_daily = cat_raw.groupby('order_date')['sales'].sum().reset_index()
        df = df.drop(columns=['sales']).merge(cat_daily, on='order_date', how='inner')

# Date Filter
min_date, max_date = df['order_date'].min(), df['order_date'].max()
date_range = st.sidebar.date_input(
    "Select Historical Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# What-If Analysis Slider
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 What-If Scenario Planner")
promo_boost = st.sidebar.slider("Simulate Promotional Boost (%)", 0, 50, 0)

# Filter dataset by date selection
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = df[(df['order_date'] >= start_date) & (df['order_date'] <= end_date)].copy()
else:
    filtered_df = df.copy()

feature_cols = [
    'year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend',
    'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
    'rolling_mean_7', 'rolling_mean_30'
]

# Predictions
filtered_df['predicted_sales'] = np.clip(model.predict(filtered_df[feature_cols]), 0, None)
if promo_boost > 0:
    filtered_df['predicted_sales'] = filtered_df['predicted_sales'] * (1 + (promo_boost / 100))

# --- AUTOMATED INVENTORY & LOW-STOCK ALERTS ---
avg_sales_baseline = df['sales'].mean()
recent_7_day_pred_avg = filtered_df['predicted_sales'].tail(7).mean() if len(filtered_df) >= 7 else filtered_df['predicted_sales'].mean()

if recent_7_day_pred_avg > (avg_sales_baseline * 1.25):
    st.warning(f"⚠️ **HIGH DEMAND SURGE DETECTED:** Projected average daily sales (${recent_7_day_pred_avg:,.2f}) are 25%+ higher than the historical baseline. **Action:** Restock top inventory items immediately!")
elif recent_7_day_pred_avg < (avg_sales_baseline * 0.75):
    st.info(f"ℹ️ **LOW DEMAND PERIOD EXPECTED:** Projected daily sales (${recent_7_day_pred_avg:,.2f}) are below average. **Action:** Consider launching promotions or holding off on large restocks.")
else:
    st.success("✅ **STABLE INVENTORY DEMAND:** Projected sales align with baseline inventory requirements.")

st.markdown("---")

# --- TOP METRICS CARDS ---
col1, col2, col3, col4 = st.columns(4)

total_sales = filtered_df['sales'].sum()
avg_daily_sales = filtered_df['sales'].mean()
pred_total_sales = filtered_df['predicted_sales'].sum()
diff_percentage = ((pred_total_sales - total_sales) / total_sales) * 100 if total_sales > 0 else 0

col1.metric("Total Revenue", f"${total_sales:,.2f}")
col2.metric("Avg Daily Revenue", f"${avg_daily_sales:,.2f}")
col3.metric("Predicted Revenue", f"${pred_total_sales:,.2f}")
col4.metric("Scenario Lift / Variance", f"{diff_percentage:+.2f}%")

st.markdown("---")

# --- CHART: HISTORICAL VS PREDICTED ---
st.subheader("📊 Revenue Forecast Chart")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered_df['order_date'], 
    y=filtered_df['sales'], 
    mode='lines', 
    name='Actual Sales', 
    line=dict(color='#1f77b4', width=2)
))
fig.add_trace(go.Scatter(
    x=filtered_df['order_date'], 
    y=filtered_df['predicted_sales'], 
    mode='lines', 
    name='Model Prediction', 
    line=dict(color='#ff7f0e', width=2, dash='dash')
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Revenue ($)",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=30, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# --- FUTURE 30-DAY FORECAST ENGINE ---
st.markdown("---")
st.subheader("🔮 Future 30-Day Automated Projection Engine")

if st.button("🚀 Generate Next 30 Days Forecast"):
    last_known_date = df['order_date'].max()
    future_dates = [last_known_date + timedelta(days=i) for i in range(1, 31)]
    
    recent_7_avg = df['sales'].tail(7).mean()
    recent_30_avg = df['sales'].tail(30).mean()
    
    future_records = []
    for dt in future_dates:
        future_records.append({
            'year': dt.year,
            'month': dt.month,
            'day': dt.day,
            'dayofweek': dt.dayofweek,
            'quarter': dt.quarter,
            'is_weekend': 1 if dt.dayofweek >= 5 else 0,
            'sales_lag_1': recent_7_avg,
            'sales_lag_7': recent_7_avg,
            'sales_lag_30': recent_30_avg,
            'rolling_mean_7': recent_7_avg,
            'rolling_mean_30': recent_30_avg
        })
    
    future_df = pd.DataFrame(future_records)
    future_preds = np.clip(model.predict(future_df[feature_cols]), 0, None)
    
    forecast_results = pd.DataFrame({
        'Date': future_dates,
        'Projected_Sales': future_preds
    })
    
    fut_fig = px.line(
        forecast_results, 
        x='Date', 
        y='Projected_Sales', 
        title="30-Day Future Revenue Projection ($)",
        markers=True
    )
    fut_fig.update_traces(line_color='#2ca02c', line_width=3)
    st.plotly_chart(fut_fig, use_container_width=True)
    
    total_projected = forecast_results['Projected_Sales'].sum()
    st.success(f"Estimated Revenue for Next 30 Days: **${total_projected:,.2f}**")