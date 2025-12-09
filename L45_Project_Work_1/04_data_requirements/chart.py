"""Data Requirements - Getting Your Project Data"""
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
fig.suptitle('Data Requirements: Sourcing Your Project Data', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Data sources
ax1 = axes[0, 0]
ax1.axis('off')

sources = '''
DATA SOURCES FOR FINANCE PROJECTS

FREE SOURCES (RECOMMENDED):
---------------------------
Yahoo Finance (yfinance library)
- Stock prices, ETFs, indices
- Historical data (decades)
- Easy Python API
- Example: yf.download("AAPL")

Kaggle Finance Datasets
- Pre-cleaned datasets
- Various formats
- Community validated
- Good documentation

FRED (Federal Reserve)
- Economic indicators
- Interest rates
- GDP, inflation data
- fredapi library


OTHER FREE OPTIONS:
-------------------
Alpha Vantage (limited free)
- Real-time quotes
- Technical indicators
- Requires API key

Quandl (now Nasdaq Data Link)
- Economic data
- Some free datasets
- Premium for full access


SYNTHETIC DATA (OK for project):
--------------------------------
- Generate realistic patterns
- Label clearly as synthetic
- Good for anomaly detection
- Useful for rare events


DATA SIZE GUIDELINES:
---------------------
- Minimum: 1,000 rows
- Recommended: 10,000+ rows
- Time series: 2+ years
- Keep files < 50MB for GitHub
'''

ax1.text(0.02, 0.98, sources, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Data Sources', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Data quality checklist
ax2 = axes[0, 1]

quality_items = [
    'Complete (minimal missing)',
    'Relevant features',
    'Appropriate size',
    'Clean format',
    'Time period adequate',
    'Documented source',
    'Reproducible download',
    'Legal to use'
]

importance = [95, 90, 85, 80, 85, 75, 80, 100]
colors = [MLGREEN if v >= 85 else MLBLUE if v >= 75 else MLORANGE for v in importance]

y_pos = np.arange(len(quality_items))
bars = ax2.barh(y_pos, importance, color=colors, alpha=0.7)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(quality_items, fontsize=9)
ax2.set_xlabel('Importance (%)')
ax2.set_xlim(0, 110)
ax2.set_title('Data Quality Checklist', fontsize=11, fontweight='bold', color=MLPURPLE)

for bar, val in zip(bars, importance):
    ax2.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val}%',
             va='center', fontsize=8)

ax2.axvline(x=85, color='gray', linestyle='--', alpha=0.5)
ax2.text(86, 7.5, 'Critical', fontsize=8, color='gray')

# Plot 3: Data loading code example
ax3 = axes[1, 0]
ax3.axis('off')

code_example = '''
DATA LOADING CODE EXAMPLES

YAHOO FINANCE:
--------------
import yfinance as yf
import pandas as pd

# Download single stock
aapl = yf.download("AAPL",
                   start="2020-01-01",
                   end="2024-01-01")

# Download multiple stocks
tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(tickers,
                   start="2020-01-01")


KAGGLE DATASETS:
----------------
# After downloading from Kaggle website
df = pd.read_csv("data/kaggle_finance.csv")


FROM URL:
---------
url = "https://example.com/data.csv"
df = pd.read_csv(url)


SAVE FOR DEPLOYMENT:
--------------------
# Save a sample for deployment
df_sample = df.tail(1000)
df_sample.to_csv("data/sample.csv",
                 index=False)

# In app, check file exists
from pathlib import Path
data_path = Path("data/sample.csv")
if data_path.exists():
    df = pd.read_csv(data_path)


REQUIREMENTS.TXT:
-----------------
yfinance==0.2.33
pandas==2.0.3
'''

ax3.text(0.02, 0.98, code_example, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Data Loading Code', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Data workflow
ax4 = axes[1, 1]
ax4.axis('off')

# Draw workflow
steps = [
    ('1. ACQUIRE', 'Download data\nfrom source', MLBLUE),
    ('2. EXPLORE', 'Check quality\nUnderstand features', MLORANGE),
    ('3. CLEAN', 'Handle missing\nFix formats', MLGREEN),
    ('4. PREPARE', 'Feature engineering\nTrain/test split', MLPURPLE),
    ('5. SAVE', 'Export for\ndeployment', MLRED)
]

for i, (title, desc, color) in enumerate(steps):
    x = 0.1 + i * 0.18
    ax4.add_patch(plt.Rectangle((x-0.07, 0.35), 0.14, 0.5, facecolor=color, alpha=0.2))
    ax4.text(x, 0.8, title, fontsize=9, ha='center', fontweight='bold', color=color)
    ax4.text(x, 0.55, desc, fontsize=7, ha='center')

    if i < len(steps) - 1:
        ax4.annotate('', xy=(x+0.1, 0.6), xytext=(x+0.05, 0.6),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

ax4.text(0.5, 0.95, 'DATA PIPELINE', fontsize=11, ha='center', fontweight='bold')

# Key notes
ax4.text(0.5, 0.2, 'KEY NOTES:', fontsize=10, ha='center', fontweight='bold')
notes = [
    '- Always keep raw data backup',
    '- Document all transformations',
    '- Use relative paths in code',
    '- Include sample data in repo'
]
for j, note in enumerate(notes):
    ax4.text(0.5, 0.12 - j*0.05, note, fontsize=8, ha='center')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title('Data Workflow', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
