"""Streamlit Caching - Performance Optimization"""
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
fig.suptitle('Streamlit Caching: Performance Optimization', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Why caching
ax1 = axes[0, 0]
ax1.axis('off')

why_cache = '''
WHY CACHING?

THE PROBLEM:
------------
Streamlit reruns your ENTIRE script
every time a user interacts with a widget.

Without caching:
- Load data every rerun: 5 seconds
- Train model every rerun: 30 seconds
- User clicks button: Wait 35 seconds!


THE SOLUTION:
-------------
Caching remembers results of expensive functions.

With caching:
- First run: Load data (5s), train model (30s)
- User clicks: Instant! (cached results)


TWO CACHING DECORATORS:
-----------------------
@st.cache_data
- For data: DataFrames, lists, dicts
- Serializable objects
- Default choice

@st.cache_resource
- For models, database connections
- Non-serializable objects
- Shared across users


CACHE BEHAVIOR:
---------------
1. Function called with arguments
2. Check if (function + args) in cache
3. If yes: Return cached result (fast!)
4. If no: Run function, store result


WHEN TO USE:
------------
- Loading CSV/database data
- API calls
- Model training
- Expensive computations
'''

ax1.text(0.02, 0.98, why_cache, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Why Caching?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Cache decorators
ax2 = axes[0, 1]
ax2.axis('off')

cache_code = '''
CACHE DECORATORS

@st.cache_data - FOR DATA:
--------------------------
@st.cache_data
def load_data(filename):
    \"\"\"Cached: only loads once per filename.\"\"\"
    return pd.read_csv(filename)

@st.cache_data
def fetch_stock_data(ticker, days):
    \"\"\"Cached: different result per ticker+days.\"\"\"
    # API call here
    return data

# Use normally - caching is automatic
df = load_data("stocks.csv")  # First call: loads
df = load_data("stocks.csv")  # Cached!


@st.cache_resource - FOR MODELS:
--------------------------------
@st.cache_resource
def load_model():
    \"\"\"Load ML model once, share across users.\"\"\"
    return joblib.load("model.joblib")

@st.cache_resource
def get_db_connection():
    \"\"\"Database connection - shared resource.\"\"\"
    return psycopg2.connect(...)


# Model stays in memory
model = load_model()  # Loads once
predictions = model.predict(X)


CACHE OPTIONS:
--------------
@st.cache_data(ttl=3600)  # Expire after 1 hour
def get_live_data():
    return fetch_from_api()

@st.cache_data(max_entries=100)  # Limit cache size
def process_data(data):
    return expensive_transform(data)

@st.cache_data(show_spinner=False)  # Hide spinner
def quick_function():
    return result
'''

ax2.text(0.02, 0.98, cache_code, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Cache Decorators', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Performance comparison
ax3 = axes[1, 0]

# Simulated performance data
operations = ['Load CSV\n(1M rows)', 'API Call\n(10 stocks)', 'Train Model\n(RF)', 'Process\nData']
without_cache = [5.2, 3.5, 45, 8]  # seconds
with_cache = [0.01, 0.01, 0.01, 0.01]  # cached

x = np.arange(len(operations))
width = 0.35

bars1 = ax3.bar(x - width/2, without_cache, width, label='Without Cache', color=MLRED, edgecolor='black')
bars2 = ax3.bar(x + width/2, with_cache, width, label='With Cache', color=MLGREEN, edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(operations, fontsize=9)
ax3.set_ylabel('Time (seconds)')
ax3.set_title('Performance: Cache vs No Cache', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=9)
ax3.set_yscale('log')
ax3.grid(alpha=0.3, axis='y')

# Add speedup labels
for i, (no_cache, cache) in enumerate(zip(without_cache, with_cache)):
    speedup = no_cache / cache
    ax3.text(i, no_cache + 2, f'{speedup:.0f}x faster', fontsize=8, ha='center', color=MLGREEN)

# Plot 4: Session state
ax4 = axes[1, 1]
ax4.axis('off')

session_state = '''
SESSION STATE

PROBLEM WITH CACHING:
---------------------
Cache is for FUNCTIONS returning the same result.
What about user-specific state that changes?


SESSION STATE - USER STATE:
---------------------------
import streamlit as st

# Initialize
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# Update
if st.button("Increment"):
    st.session_state.counter += 1

# Display
st.write(f"Count: {st.session_state.counter}")


PRACTICAL EXAMPLE:
------------------
# Track selected stocks
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

ticker = st.selectbox("Select stock", ['AAPL', 'GOOGL', 'MSFT'])

if st.button("Add to portfolio"):
    st.session_state.portfolio.append(ticker)

st.write("Your portfolio:", st.session_state.portfolio)


CACHE vs SESSION STATE:
-----------------------
Cache:
- Same result for same inputs
- Shared across users
- For expensive computations

Session State:
- User-specific data
- Changes during session
- For UI state, forms, selections


CLEARING CACHE:
---------------
st.cache_data.clear()     # Clear all data cache
st.cache_resource.clear() # Clear all resource cache

# Or via button
if st.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
'''

ax4.text(0.02, 0.98, session_state, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Session State', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
