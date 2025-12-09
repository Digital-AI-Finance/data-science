"""Streamlit Widgets - User Input Components"""
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
fig.suptitle('Streamlit Widgets: User Input Components', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Input widgets
ax1 = axes[0, 0]
ax1.axis('off')

input_widgets = '''
INPUT WIDGETS

TEXT INPUT:
-----------
name = st.text_input("Enter name", value="John")
st.write(f"Hello, {name}")


TEXT AREA:
----------
description = st.text_area("Description", height=100)


NUMBER INPUT:
-------------
price = st.number_input("Price", min_value=0.0, max_value=1000.0)


SLIDER:
-------
threshold = st.slider("Threshold", 0, 100, 50)

# Range slider
range_val = st.slider("Range", 0, 100, (25, 75))


SELECT SLIDER:
--------------
rating = st.select_slider("Rating",
    options=['Poor', 'Fair', 'Good', 'Excellent'])


DATE INPUT:
-----------
from datetime import date
d = st.date_input("Start date", date.today())


TIME INPUT:
-----------
from datetime import time
t = st.time_input("Time", time(9, 0))


COLOR PICKER:
-------------
color = st.color_picker("Pick a color", "#3333B2")


FILE UPLOAD:
------------
uploaded = st.file_uploader("Upload CSV", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
'''

ax1.text(0.02, 0.98, input_widgets, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Input Widgets', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Selection widgets
ax2 = axes[0, 1]
ax2.axis('off')

selection_widgets = '''
SELECTION WIDGETS

SELECTBOX (Dropdown):
---------------------
ticker = st.selectbox(
    "Select Stock",
    options=['AAPL', 'GOOGL', 'MSFT', 'AMZN']
)


MULTISELECT:
------------
selected = st.multiselect(
    "Select Stocks",
    options=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
    default=['AAPL']
)


RADIO BUTTONS:
--------------
model = st.radio(
    "Select Model",
    options=['Random Forest', 'XGBoost', 'Neural Network'],
    horizontal=True  # Display horizontally
)


CHECKBOX:
---------
show_data = st.checkbox("Show raw data", value=False)
if show_data:
    st.dataframe(df)


TOGGLE:
-------
dark_mode = st.toggle("Dark Mode", value=False)


BUTTON:
-------
if st.button("Run Prediction"):
    result = model.predict(X)
    st.write(result)


DOWNLOAD BUTTON:
----------------
st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False),
    file_name="predictions.csv",
    mime="text/csv"
)


FORM (Group Widgets):
---------------------
with st.form("my_form"):
    name = st.text_input("Name")
    age = st.number_input("Age")
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write(f"Name: {name}, Age: {age}")
'''

ax2.text(0.02, 0.98, selection_widgets, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Selection Widgets', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Widget categories visualization
ax3 = axes[1, 0]
ax3.axis('off')

# Create widget category cards
categories = [
    ('Text\nInput', ['text_input', 'text_area', 'number_input'], MLBLUE),
    ('Selection', ['selectbox', 'multiselect', 'radio'], MLGREEN),
    ('Range', ['slider', 'select_slider', 'date_input'], MLORANGE),
    ('Action', ['button', 'form_submit', 'download'], MLPURPLE)
]

for i, (cat, widgets, color) in enumerate(categories):
    x = 0.12 + i * 0.22
    ax3.add_patch(plt.Rectangle((x-0.08, 0.3), 0.2, 0.55, facecolor=color, alpha=0.3))
    ax3.text(x+0.02, 0.8, cat, fontsize=10, ha='center', fontweight='bold')
    for j, w in enumerate(widgets):
        ax3.text(x+0.02, 0.65 - j*0.12, f'st.{w}()', fontsize=8, ha='center', fontfamily='monospace')

ax3.text(0.5, 0.95, 'WIDGET CATEGORIES', fontsize=12, ha='center', fontweight='bold')
ax3.text(0.5, 0.2, 'All widgets return values that update when user interacts',
         fontsize=9, ha='center', style='italic')

ax3.set_xlim(0, 1)
ax3.set_ylim(0.1, 1)
ax3.set_title('Widget Categories', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete widget example
ax4 = axes[1, 1]
ax4.axis('off')

complete_example = '''
COMPLETE WIDGET EXAMPLE

import streamlit as st
import pandas as pd
import numpy as np


st.title("Stock Analysis Dashboard")


# Sidebar widgets
st.sidebar.header("Settings")

ticker = st.sidebar.selectbox(
    "Select Stock",
    ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
)

date_range = st.sidebar.slider(
    "Days of history",
    min_value=30,
    max_value=365,
    value=90
)

show_volume = st.sidebar.checkbox("Show volume", True)


# Main content based on widgets
st.header(f"Analysis for {ticker}")

# Simulate data based on selection
np.random.seed(42)
dates = pd.date_range(end='today', periods=date_range)
prices = 100 + np.cumsum(np.random.randn(date_range))
df = pd.DataFrame({'Date': dates, 'Price': prices})

# Display chart
st.line_chart(df.set_index('Date'))


# Conditional display
if show_volume:
    volume = np.random.randint(1000000, 5000000, date_range)
    st.bar_chart(pd.DataFrame({'Volume': volume}, index=dates))


# Action button
if st.button("Download Report"):
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        f"{ticker}_data.csv"
    )


# The UI updates automatically when widgets change!
'''

ax4.text(0.02, 0.98, complete_example, transform=ax4.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Example', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
