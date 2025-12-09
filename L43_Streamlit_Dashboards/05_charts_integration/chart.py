"""Charts Integration - Visualizations in Streamlit"""
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
fig.suptitle('Charts Integration: Visualizations in Streamlit', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Built-in charts
ax1 = axes[0, 0]
ax1.axis('off')

builtin_charts = '''
STREAMLIT BUILT-IN CHARTS

LINE CHART:
-----------
import streamlit as st
import pandas as pd
import numpy as np

# Simple line chart
data = pd.DataFrame({
    'AAPL': np.random.randn(20).cumsum(),
    'GOOGL': np.random.randn(20).cumsum()
})
st.line_chart(data)


BAR CHART:
----------
st.bar_chart(data)


AREA CHART:
-----------
st.area_chart(data)


SCATTER PLOT:
-------------
st.scatter_chart(data)


BUILT-IN ADVANTAGES:
--------------------
+ Simple, one-line code
+ Automatically interactive
+ Good for quick dashboards

LIMITATIONS:
------------
- Limited customization
- Basic styling options
- Not for complex visualizations


FOR COMPLEX CHARTS:
-------------------
Use matplotlib, plotly, or altair
'''

ax1.text(0.02, 0.98, builtin_charts, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Built-in Charts', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Matplotlib integration
ax2 = axes[0, 1]
ax2.axis('off')

matplotlib_code = '''
MATPLOTLIB IN STREAMLIT

BASIC MATPLOTLIB:
-----------------
import matplotlib.pyplot as plt
import streamlit as st

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_xlabel("X")
ax.set_ylabel("Y")

st.pyplot(fig)  # Display in Streamlit


FINANCE EXAMPLE:
----------------
import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots(figsize=(10, 6))

# Plot stock prices
ax.plot(df['Date'], df['Close'], label='Close', color='#3333B2')
ax.fill_between(df['Date'], df['Close'], alpha=0.3)

# Add moving average
ax.plot(df['Date'], df['MA_20'], label='20-day MA',
        color='#FF7F0E', linestyle='--')

ax.set_title(f"{ticker} Stock Price")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)


WITH SUBPLOTS:
--------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(df['Close'])
ax1.set_title("Price")

ax2.bar(df.index, df['Volume'])
ax2.set_title("Volume")

plt.tight_layout()
st.pyplot(fig)


CLEAR FIGURE:
-------------
plt.close()  # Important to prevent memory leaks!
'''

ax2.text(0.02, 0.98, matplotlib_code, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Matplotlib Integration', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Plotly integration
ax3 = axes[1, 0]
ax3.axis('off')

plotly_code = '''
PLOTLY IN STREAMLIT (Interactive!)

BASIC PLOTLY:
-------------
import plotly.express as px
import streamlit as st

fig = px.line(df, x='Date', y='Close', title='Stock Price')
st.plotly_chart(fig, use_container_width=True)


CANDLESTICK CHART:
------------------
import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )
])
fig.update_layout(title=f"{ticker} Candlestick")
st.plotly_chart(fig)


MULTIPLE TRACES:
----------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Close'],
    name='Close', mode='lines'
))
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['MA_20'],
    name='20-day MA', mode='lines',
    line=dict(dash='dash')
))

st.plotly_chart(fig, use_container_width=True)


WHY PLOTLY?
-----------
+ Interactive (zoom, pan, hover)
+ Professional financial charts
+ Animations possible
+ Great for dashboards
'''

ax3.text(0.02, 0.98, plotly_code, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Plotly Integration', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Chart library comparison
ax4 = axes[1, 1]

# Chart library comparison
libraries = ['st.line_chart', 'Matplotlib', 'Plotly', 'Altair']
ease_of_use = [5, 3, 4, 4]
customization = [2, 5, 4, 4]
interactivity = [3, 2, 5, 5]
performance = [5, 4, 3, 4]

x = np.arange(len(libraries))
width = 0.2

bars1 = ax4.bar(x - 1.5*width, ease_of_use, width, label='Ease of Use', color=MLBLUE, edgecolor='black')
bars2 = ax4.bar(x - 0.5*width, customization, width, label='Customization', color=MLGREEN, edgecolor='black')
bars3 = ax4.bar(x + 0.5*width, interactivity, width, label='Interactivity', color=MLORANGE, edgecolor='black')
bars4 = ax4.bar(x + 1.5*width, performance, width, label='Performance', color=MLPURPLE, edgecolor='black')

ax4.set_xticks(x)
ax4.set_xticklabels(libraries, fontsize=9)
ax4.set_ylabel('Score (1-5)')
ax4.set_title('Chart Library Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=8, loc='lower right')
ax4.set_ylim(0, 6)
ax4.grid(alpha=0.3, axis='y')

# Add recommendations
recommendations = ['Quick\nprototype', 'Publication\nquality', 'Dashboards\n(recommended)', 'Declarative']
for i, rec in enumerate(recommendations):
    ax4.text(i, 0.5, rec, fontsize=7, ha='center', style='italic')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
