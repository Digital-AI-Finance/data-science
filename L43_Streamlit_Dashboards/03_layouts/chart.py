"""Streamlit Layouts - Organizing Your App"""
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
fig.suptitle('Streamlit Layouts: Organizing Your App', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Columns
ax1 = axes[0, 0]
ax1.axis('off')

columns_code = '''
COLUMNS - HORIZONTAL LAYOUT

BASIC COLUMNS:
--------------
col1, col2 = st.columns(2)

with col1:
    st.header("Left Column")
    st.write("Content here")

with col2:
    st.header("Right Column")
    st.write("More content")


UNEQUAL COLUMNS:
----------------
col1, col2, col3 = st.columns([2, 1, 1])
# Proportions: 2:1:1

with col1:
    st.write("Wide column (50%)")
with col2:
    st.write("Narrow (25%)")
with col3:
    st.write("Narrow (25%)")


COLUMNS WITH GAP:
-----------------
col1, col2 = st.columns(2, gap="large")
# gap options: "small", "medium", "large"


METRIC CARDS EXAMPLE:
---------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Price", "$150.25", "+2.5%")
col2.metric("Volume", "1.2M", "-5%")
col3.metric("P/E Ratio", "28.5", "+0.3")
col4.metric("Market Cap", "$2.4T", "+1.2%")


CHARTS SIDE BY SIDE:
--------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price")
    st.line_chart(price_data)

with col2:
    st.subheader("Volume")
    st.bar_chart(volume_data)
'''

ax1.text(0.02, 0.98, columns_code, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Columns', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Sidebar and containers
ax2 = axes[0, 1]
ax2.axis('off')

containers_code = '''
SIDEBAR & CONTAINERS

SIDEBAR:
--------
# Add widgets to sidebar
st.sidebar.title("Settings")
st.sidebar.selectbox("Model", ["RF", "XGBoost"])
st.sidebar.slider("Threshold", 0, 100)

# Or use context manager
with st.sidebar:
    st.title("Settings")
    model = st.selectbox("Model", ["RF", "XGBoost"])


EXPANDER:
---------
with st.expander("Advanced Settings"):
    learning_rate = st.number_input("Learning Rate", 0.01)
    epochs = st.number_input("Epochs", 100)
    st.write("These are hidden by default")


CONTAINER:
----------
container = st.container()

# Add content later
container.write("This appears first")
container.write("This appears second")


EMPTY PLACEHOLDER:
------------------
placeholder = st.empty()

# Update dynamically
for i in range(100):
    placeholder.write(f"Progress: {i}%")
    time.sleep(0.1)

placeholder.write("Done!")


TABS:
-----
tab1, tab2, tab3 = st.tabs(["Charts", "Data", "Settings"])

with tab1:
    st.line_chart(data)

with tab2:
    st.dataframe(df)

with tab3:
    st.slider("Option", 0, 10)
'''

ax2.text(0.02, 0.98, containers_code, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Sidebar & Containers', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Layout visualization
ax3 = axes[1, 0]
ax3.axis('off')

# Draw layout mockup
# Sidebar
ax3.add_patch(plt.Rectangle((0.02, 0.1), 0.2, 0.85, facecolor=MLBLUE, alpha=0.3))
ax3.text(0.12, 0.92, 'Sidebar', fontsize=10, ha='center', fontweight='bold')
ax3.text(0.12, 0.8, 'Settings', fontsize=8, ha='center')
ax3.text(0.12, 0.65, '[Selectbox]', fontsize=7, ha='center')
ax3.text(0.12, 0.55, '[Slider]', fontsize=7, ha='center')
ax3.text(0.12, 0.45, '[Checkbox]', fontsize=7, ha='center')

# Main content
ax3.add_patch(plt.Rectangle((0.25, 0.1), 0.72, 0.85, facecolor='white', edgecolor='gray'))

# Header
ax3.add_patch(plt.Rectangle((0.27, 0.85), 0.68, 0.08, facecolor=MLPURPLE, alpha=0.3))
ax3.text(0.61, 0.89, 'Title / Header', fontsize=10, ha='center', fontweight='bold')

# Metrics row (columns)
for i in range(4):
    x = 0.28 + i * 0.17
    ax3.add_patch(plt.Rectangle((x, 0.7), 0.15, 0.1, facecolor=MLGREEN, alpha=0.3))
    ax3.text(x + 0.075, 0.75, f'Metric {i+1}', fontsize=7, ha='center')

# Two column content
ax3.add_patch(plt.Rectangle((0.28, 0.35), 0.33, 0.3, facecolor=MLORANGE, alpha=0.3))
ax3.text(0.445, 0.5, 'Chart 1', fontsize=9, ha='center')

ax3.add_patch(plt.Rectangle((0.63, 0.35), 0.33, 0.3, facecolor=MLORANGE, alpha=0.3))
ax3.text(0.795, 0.5, 'Chart 2', fontsize=9, ha='center')

# Expander
ax3.add_patch(plt.Rectangle((0.28, 0.15), 0.68, 0.15, facecolor=MLLAVENDER, alpha=0.5))
ax3.text(0.61, 0.22, '[Expander: Details]', fontsize=8, ha='center')

ax3.set_xlim(0, 1)
ax3.set_ylim(0.05, 1)
ax3.set_title('Layout Example', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete layout code
ax4 = axes[1, 1]
ax4.axis('off')

layout_example = '''
COMPLETE LAYOUT EXAMPLE

import streamlit as st
import pandas as pd


# Page config
st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)


# Sidebar
with st.sidebar:
    st.title("Settings")
    ticker = st.selectbox("Stock", ['AAPL', 'GOOGL'])
    days = st.slider("Days", 30, 365, 90)
    st.divider()
    st.checkbox("Show volume")


# Main content
st.title("Stock Analysis Dashboard")


# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Price", "$150.25", "+2.5%")
col2.metric("Volume", "1.2M", "-5%")
col3.metric("High", "$155", "+3%")
col4.metric("Low", "$148", "-1%")


# Charts row
col1, col2 = st.columns(2)
with col1:
    st.subheader("Price History")
    st.line_chart(price_data)

with col2:
    st.subheader("Trading Volume")
    st.bar_chart(volume_data)


# Tabs for more content
tab1, tab2 = st.tabs(["Analysis", "Raw Data"])

with tab1:
    st.write("Analysis content...")

with tab2:
    st.dataframe(df)


# Footer expander
with st.expander("About this dashboard"):
    st.write("Built with Streamlit")
'''

ax4.text(0.02, 0.98, layout_example, transform=ax4.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Layout', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
