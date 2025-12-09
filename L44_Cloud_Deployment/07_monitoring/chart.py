"""Monitoring Deployed Apps - Keeping Track of Performance"""
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
fig.suptitle('Monitoring Deployed Apps: Keeping Track of Performance', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What to monitor
ax1 = axes[0, 0]
ax1.axis('off')

monitoring = '''
WHAT TO MONITOR

APP HEALTH:
-----------
- Is the app running?
- Response time
- Error rate
- Memory usage


STREAMLIT CLOUD PROVIDES:
-------------------------
- App status (running/error)
- Logs (errors and output)
- Resource usage
- Viewer count (analytics)


ACCESSING LOGS:
---------------
1. Go to share.streamlit.io
2. Find your app
3. Click "..." menu
4. Select "Manage app"
5. Click "Logs" tab


LOG INFORMATION:
----------------
- Streamlit version
- Python version
- Package installations
- App startup
- Runtime errors
- User interactions (if logged)


ADDING YOUR OWN LOGGING:
------------------------
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log events
logger.info("App started")
logger.info(f"User selected: {ticker}")
logger.error("Prediction failed!")

# View in cloud logs


SLEEP BEHAVIOR:
---------------
Free tier apps sleep after inactivity.
First visit after sleep = slower load.
'''

ax1.text(0.02, 0.98, monitoring, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What to Monitor', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Handling errors
ax2 = axes[0, 1]
ax2.axis('off')

error_handling = '''
HANDLING PRODUCTION ERRORS

GRACEFUL ERROR HANDLING:
------------------------
import streamlit as st

try:
    result = model.predict(X)
    st.success(f"Prediction: {result}")
except Exception as e:
    st.error("Prediction failed!")
    st.write("Please check your inputs.")
    # Log error for debugging
    st.exception(e)  # Shows traceback


CHECK DEPENDENCIES:
-------------------
def check_model():
    try:
        model = load_model()
        return True
    except FileNotFoundError:
        st.error("Model file not found!")
        st.info("Please ensure model is deployed.")
        return False

if not check_model():
    st.stop()


FALLBACK VALUES:
----------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/stocks.csv")
    except FileNotFoundError:
        st.warning("Using sample data (actual data unavailable)")
        return pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL'],
            'price': [150, 140]
        })


USER-FRIENDLY MESSAGES:
-----------------------
# Bad
st.error("KeyError: 'price'")

# Good
st.error("Missing price data. Please refresh or try again.")


CONTACT INFO:
-------------
st.sidebar.markdown("---")
st.sidebar.info("Issues? Contact: support@example.com")
'''

ax2.text(0.02, 0.98, error_handling, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Error Handling', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Usage metrics visualization
ax3 = axes[1, 0]

# Simulated usage data
hours = np.arange(24)
views = np.array([10, 5, 3, 2, 2, 5, 15, 45, 80, 95, 100, 105,
                  110, 100, 90, 85, 70, 50, 40, 35, 30, 25, 20, 15])

ax3.fill_between(hours, views, alpha=0.3, color=MLBLUE)
ax3.plot(hours, views, color=MLBLUE, linewidth=2)

# Mark peak hours
peak_hours = np.where(views > 80)[0]
ax3.scatter(peak_hours, views[peak_hours], color=MLORANGE, s=100, zorder=5, label='Peak hours')

ax3.set_xlabel('Hour of Day')
ax3.set_ylabel('Page Views')
ax3.set_title('Daily Usage Pattern (Example)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Add stats
total_views = sum(views)
peak_hour = hours[np.argmax(views)]
ax3.text(0.02, 0.98, f'Total: {total_views} views\nPeak: {peak_hour}:00',
         transform=ax3.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 4: Maintenance tasks
ax4 = axes[1, 1]

tasks = [
    ('Check logs daily', 'Find errors early', MLGREEN),
    ('Update dependencies monthly', 'Security + features', MLBLUE),
    ('Monitor memory usage', 'Prevent crashes', MLORANGE),
    ('Review analytics', 'Understand users', MLPURPLE),
    ('Test after updates', 'Catch regressions', MLRED),
    ('Backup configurations', 'Disaster recovery', MLBLUE)
]

y_pos = np.arange(len(tasks))

for i, (task, reason, color) in enumerate(tasks):
    ax4.add_patch(plt.Rectangle((0, i-0.35), 0.55, 0.7, facecolor=color, alpha=0.2))
    ax4.text(0.02, i, task, fontsize=9, va='center', fontweight='bold')
    ax4.text(0.57, i, reason, fontsize=8, va='center', style='italic')

ax4.set_xlim(0, 1)
ax4.set_ylim(-0.5, len(tasks)-0.5)
ax4.invert_yaxis()
ax4.axis('off')
ax4.set_title('Maintenance Tasks', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
