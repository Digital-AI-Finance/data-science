"""Stock Dashboard - Complete Finance Application"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (10, 6),
    'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Stock Dashboard: Complete Finance Application', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Complete stock dashboard code (part 1)
ax1 = axes[0, 0]
ax1.axis('off')

code_part1 = '''
STOCK DASHBOARD APP (Part 1)

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib


# Page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)


# Cache functions
@st.cache_resource
def load_model():
    return joblib.load('models/stock_classifier.joblib')

@st.cache_data
def load_stock_data(ticker, days):
    # Simulate stock data
    np.random.seed(hash(ticker) % 100)
    dates = pd.date_range(end='today', periods=days)
    base = 100 if ticker == 'AAPL' else 150
    prices = base * np.exp(np.cumsum(np.random.randn(days) * 0.02))
    volume = np.random.randint(1e6, 5e6, days)

    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': volume,
        'MA_20': pd.Series(prices).rolling(20).mean(),
        'Returns': pd.Series(prices).pct_change()
    })


# Load model
model = load_model()


# SIDEBAR
st.sidebar.title("Settings")

ticker = st.sidebar.selectbox(
    "Select Stock",
    ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
)

days = st.sidebar.slider("Days of History", 30, 365, 90)

show_ma = st.sidebar.checkbox("Show Moving Average", True)
show_volume = st.sidebar.checkbox("Show Volume", True)
'''

ax1.text(0.02, 0.98, code_part1, transform=ax1.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Dashboard Code (Part 1)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Complete stock dashboard code (part 2)
ax2 = axes[0, 1]
ax2.axis('off')

code_part2 = '''
STOCK DASHBOARD APP (Part 2)

# MAIN CONTENT
st.title(f"{ticker} Stock Analysis")

# Load data
df = load_stock_data(ticker, days)


# METRICS ROW
col1, col2, col3, col4 = st.columns(4)

current_price = df['Close'].iloc[-1]
prev_price = df['Close'].iloc[-2]
change = (current_price - prev_price) / prev_price * 100

col1.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f}%")
col2.metric("Volume", f"{df['Volume'].iloc[-1]/1e6:.1f}M")
col3.metric("52-Week High", f"${df['Close'].max():.2f}")
col4.metric("52-Week Low", f"${df['Close'].min():.2f}")


# PRICE CHART
st.subheader("Price History")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Close'],
    name='Price', line=dict(color='#3333B2')
))

if show_ma:
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['MA_20'],
        name='20-day MA', line=dict(dash='dash', color='#FF7F0E')
    ))

fig.update_layout(height=400, showlegend=True)
st.plotly_chart(fig, use_container_width=True)


# VOLUME CHART
if show_volume:
    st.subheader("Trading Volume")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df['Date'], y=df['Volume']))
    st.plotly_chart(fig_vol, use_container_width=True)
'''

ax2.text(0.02, 0.98, code_part2, transform=ax2.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Dashboard Code (Part 2)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Prediction section code
ax3 = axes[1, 0]
ax3.axis('off')

code_part3 = '''
STOCK DASHBOARD APP (Part 3 - Prediction)

# PREDICTION SECTION
st.header("Price Prediction")

st.write("Enter features for ML prediction:")

col1, col2, col3 = st.columns(3)

with col1:
    pred_price = st.number_input(
        "Current Price",
        value=float(current_price),
        min_value=0.0
    )

with col2:
    pred_volume = st.number_input(
        "Volume (M)",
        value=df['Volume'].iloc[-1] / 1e6,
        min_value=0.0
    )

with col3:
    pred_momentum = st.slider(
        "Momentum",
        -1.0, 1.0,
        float(df['Returns'].iloc[-5:].mean() * 10)
    )


if st.button("Predict Direction", type="primary"):
    X = np.array([[pred_price, pred_volume * 1e6, pred_momentum]])

    with st.spinner("Running prediction..."):
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("Predicted Direction: UP", icon="arrow_upward")
        else:
            st.error("Predicted Direction: DOWN", icon="arrow_downward")

    with col2:
        st.metric("Confidence", f"{max(proba):.1%}")


# FOOTER
st.divider()
st.caption("Data is simulated. Not financial advice.")
'''

ax3.text(0.02, 0.98, code_part3, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Dashboard Code (Part 3)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Dashboard mockup
ax4 = axes[1, 1]
ax4.axis('off')

# Draw dashboard mockup
# Sidebar
ax4.add_patch(plt.Rectangle((0.02, 0.1), 0.18, 0.85, facecolor=MLBLUE, alpha=0.2))
ax4.text(0.11, 0.9, 'Settings', fontsize=9, ha='center', fontweight='bold')
ax4.text(0.11, 0.8, '[AAPL v]', fontsize=7, ha='center')
ax4.text(0.11, 0.7, '[90 days]', fontsize=7, ha='center')
ax4.text(0.11, 0.6, '[x] MA', fontsize=7, ha='center')

# Main content
# Title
ax4.text(0.6, 0.92, 'AAPL Stock Analysis', fontsize=12, ha='center', fontweight='bold')

# Metrics
metrics_x = [0.28, 0.45, 0.62, 0.79]
metrics = [('$150.25', '+2.5%'), ('1.2M', '-5%'), ('$165', ''), ('$120', '')]
for x, (val, delta) in zip(metrics_x, metrics):
    ax4.add_patch(plt.Rectangle((x-0.06, 0.78), 0.12, 0.1, facecolor=MLLAVENDER, alpha=0.5))
    ax4.text(x, 0.85, val, fontsize=8, ha='center', fontweight='bold')
    if delta:
        color = MLGREEN if '+' in delta else MLRED
        ax4.text(x, 0.8, delta, fontsize=7, ha='center', color=color)

# Price chart area
ax4.add_patch(plt.Rectangle((0.23, 0.45), 0.72, 0.28, facecolor='white', edgecolor='gray'))
ax4.text(0.59, 0.7, '[Price Chart]', fontsize=10, ha='center', style='italic')

# Generate mini chart
x_chart = np.linspace(0.25, 0.93, 50)
y_chart = 0.52 + 0.08 * np.sin(np.linspace(0, 4, 50)) + 0.05 * np.cumsum(np.random.randn(50) * 0.1)
ax4.plot(x_chart, y_chart, color=MLBLUE, linewidth=2)

# Prediction section
ax4.add_patch(plt.Rectangle((0.23, 0.12), 0.72, 0.28, facecolor=MLGREEN, alpha=0.15))
ax4.text(0.3, 0.35, 'Prediction', fontsize=9, fontweight='bold')
ax4.text(0.59, 0.25, 'Direction: UP', fontsize=10, ha='center', color=MLGREEN, fontweight='bold')
ax4.text(0.59, 0.18, 'Confidence: 87%', fontsize=9, ha='center')

ax4.set_xlim(0, 1)
ax4.set_ylim(0.05, 1)
ax4.set_title('Dashboard Preview', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
