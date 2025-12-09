"""Streamlit Introduction - Building Data Apps"""
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
fig.suptitle('Streamlit Introduction: Building Data Apps', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is Streamlit
ax1 = axes[0, 0]
ax1.axis('off')

intro = '''
WHAT IS STREAMLIT?

DEFINITION:
-----------
Open-source Python framework for building
interactive data applications quickly.

"Turn data scripts into web apps in minutes."


KEY FEATURES:
-------------
- No web development knowledge needed
- Pure Python
- Auto-refresh on code changes
- Built-in widgets (sliders, buttons, etc.)
- Easy chart integration
- Free cloud hosting


PERFECT FOR:
------------
- Data dashboards
- ML model demos
- Internal tools
- Prototypes
- Educational apps


COMPARED TO ALTERNATIVES:
-------------------------
Streamlit: Simplest, fastest to build
Dash: More customization, more complex
Flask: Full web framework, most work
Django: Enterprise, overkill for dashboards


INSTALLATION:
-------------
pip install streamlit


RUN AN APP:
-----------
streamlit run app.py


DEFAULT URL:
------------
http://localhost:8501
'''

ax1.text(0.02, 0.98, intro, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What is Streamlit?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: First Streamlit app
ax2 = axes[0, 1]
ax2.axis('off')

first_app = '''
YOUR FIRST STREAMLIT APP

# app.py
import streamlit as st
import pandas as pd
import numpy as np


# Title
st.title("My First Streamlit App")


# Text
st.write("Hello, World!")
st.markdown("## This is a subheader")


# Data
df = pd.DataFrame({
    'Stock': ['AAPL', 'GOOGL', 'MSFT'],
    'Price': [150, 140, 380]
})
st.dataframe(df)


# Chart
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['AAPL', 'GOOGL', 'MSFT']
)
st.line_chart(chart_data)


# Interactive widget
name = st.text_input("Enter your name")
st.write(f"Hello, {name}!")


RUN IT:
-------
streamlit run app.py


THAT'S IT!
----------
~20 lines of Python = interactive web app!


FEATURES INCLUDED:
------------------
- Automatic layout
- Responsive design
- Dark/light mode
- Download buttons
- Shareable URL
'''

ax2.text(0.02, 0.98, first_app, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('First App', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Framework comparison
ax3 = axes[1, 0]

frameworks = ['Streamlit', 'Dash', 'Flask', 'Django']
ease_of_use = [5, 3.5, 3, 2]
time_to_build = [5, 3.5, 2.5, 2]
customization = [3, 4.5, 5, 5]
features = [4, 4.5, 3, 5]

x = np.arange(len(frameworks))
width = 0.2

bars1 = ax3.bar(x - 1.5*width, ease_of_use, width, label='Ease of Use', color=MLBLUE, edgecolor='black')
bars2 = ax3.bar(x - 0.5*width, time_to_build, width, label='Speed to Build', color=MLGREEN, edgecolor='black')
bars3 = ax3.bar(x + 0.5*width, customization, width, label='Customization', color=MLORANGE, edgecolor='black')
bars4 = ax3.bar(x + 1.5*width, features, width, label='Features', color=MLPURPLE, edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(frameworks, fontsize=10)
ax3.set_ylabel('Score (1-5)')
ax3.set_title('Framework Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=8, loc='lower right')
ax3.set_ylim(0, 6)
ax3.grid(alpha=0.3, axis='y')

# Highlight Streamlit for data apps
ax3.annotate('Best for\nData Apps', xy=(0, 5.3), fontsize=9, ha='center', color=MLGREEN)

# Plot 4: Streamlit concepts
ax4 = axes[1, 1]
ax4.axis('off')

concepts = '''
KEY STREAMLIT CONCEPTS

1. SCRIPT EXECUTION:
--------------------
Streamlit runs your script top-to-bottom
every time user interacts with a widget.


2. DATA FLOW:
-------------
Widget changes -> Script reruns -> App updates


3. STATE MANAGEMENT:
--------------------
Use st.session_state to persist data
across reruns.

st.session_state['counter'] = 0


4. CACHING:
-----------
Avoid recomputing expensive operations.

@st.cache_data
def load_data():
    return pd.read_csv('data.csv')


5. LAYOUTS:
-----------
Organize with columns and containers.

col1, col2 = st.columns(2)
with col1:
    st.write("Left")
with col2:
    st.write("Right")


6. WIDGETS:
-----------
st.button()        # Buttons
st.slider()        # Sliders
st.selectbox()     # Dropdowns
st.text_input()    # Text input
st.file_uploader() # File upload


MINDSET:
--------
Think of Streamlit as a Python script
that regenerates the page on each interaction.
'''

ax4.text(0.02, 0.98, concepts, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Key Concepts', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
