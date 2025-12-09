"""News Sentiment Analysis for Trading"""
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
fig.suptitle('News Sentiment Analysis for Trading', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: News sources and preprocessing
ax1 = axes[0, 0]
ax1.axis('off')

sources = '''
NEWS DATA FOR SENTIMENT ANALYSIS

NEWS SOURCES:
-------------
Free/Academic:
- Yahoo Finance RSS
- Google Finance
- Financial news APIs (limited)
- Web scraping (check TOS!)

Commercial:
- Reuters
- Bloomberg
- RavenPack
- Refinitiv


PREPROCESSING PIPELINE:
-----------------------
1. Remove HTML tags
2. Handle special characters
3. Remove URLs
4. Company name normalization
   ("Apple Inc." -> "AAPL")
5. Date extraction and normalization


KEY FIELDS TO EXTRACT:
----------------------
- Headline (most important!)
- Publication datetime
- Source reliability score
- Company/ticker mentioned
- Article category


HEADLINE VS FULL TEXT:
----------------------
Headlines:
+ Quick to process
+ Dense information
+ Less noise
- Missing context

Full text:
+ More context
+ Captures nuance
- Slower processing
- More noise


DEDUPLICATION:
--------------
Same news from multiple sources!
Use fuzzy matching to deduplicate.

from fuzzywuzzy import fuzz
if fuzz.ratio(headline1, headline2) > 85:
    # Likely duplicate
'''

ax1.text(0.02, 0.98, sources, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('News Data Sources', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Example news analysis
ax2 = axes[0, 1]

# Simulated news headlines and sentiments
headlines = [
    'Apple beats Q3 earnings',
    'Fed signals rate pause',
    'Tech stocks tumble on fears',
    'Microsoft Azure growth slows',
    'Amazon expands cloud services',
    'Tesla recalls 100K vehicles',
    'JP Morgan raises dividend',
    'Oil prices surge on supply'
]

sentiments = [0.82, 0.35, -0.75, -0.45, 0.55, -0.68, 0.72, 0.28]
colors = [MLGREEN if s > 0.2 else MLRED if s < -0.2 else MLBLUE for s in sentiments]

y_pos = np.arange(len(headlines))
bars = ax2.barh(y_pos, sentiments, color=colors, edgecolor='black')

ax2.set_yticks(y_pos)
ax2.set_yticklabels([h[:25] + '...' if len(h) > 25 else h for h in headlines], fontsize=8)
ax2.set_xlabel('Sentiment Score')
ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax2.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(-0.2, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlim(-1, 1)
ax2.set_title('Sample News Sentiment', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.invert_yaxis()
ax2.grid(alpha=0.3, axis='x')

# Plot 3: News sentiment strategy
ax3 = axes[1, 0]
ax3.axis('off')

strategy = '''
NEWS SENTIMENT TRADING STRATEGY

BASIC STRATEGY:
---------------
1. Aggregate daily news sentiment
2. Generate signals based on thresholds
3. Enter positions on sentiment extremes

Signal Rules:
- sentiment > 0.3 -> BUY
- sentiment < -0.3 -> SELL
- otherwise -> HOLD


ENHANCED STRATEGY:
------------------
Combine multiple factors:
- Sentiment level (current)
- Sentiment momentum (change)
- News volume (attention)
- Sentiment volatility


EXAMPLE IMPLEMENTATION:
-----------------------
def generate_signal(sent_mean, sent_ma5, news_count):
    # High volume + strong sentiment
    if news_count > 10 and abs(sent_mean) > 0.5:
        return np.sign(sent_mean) * 2  # Strong signal

    # Sentiment momentum
    if sent_mean > sent_ma5 + 0.2:
        return 1  # Improving sentiment
    elif sent_mean < sent_ma5 - 0.2:
        return -1  # Worsening sentiment

    return 0  # No signal


RISK MANAGEMENT:
----------------
- Don't trade on single news item
- Require confirmation (multiple sources)
- Set position size limits
- Use stop losses


TIMING CONSIDERATIONS:
----------------------
- Pre-market news vs market hours
- Earnings announcements (high impact)
- Weekend news (Monday effect)


BACKTESTING:
------------
- Use out-of-sample data
- Account for lookahead bias
- Include transaction costs
'''

ax3.text(0.02, 0.98, strategy, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Sentiment Trading Strategy', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Strategy backtest visualization
ax4 = axes[1, 1]

# Simulate backtest results
np.random.seed(456)
days = 100
dates = np.arange(days)

# Strategy vs benchmark
benchmark = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.01))
sentiment_returns = np.random.randn(days) * 0.012 + 0.0005  # Slight edge
strategy = 100 * np.exp(np.cumsum(sentiment_returns))

ax4.plot(dates, benchmark, color=MLBLUE, linewidth=2, label='Benchmark (S&P 500)')
ax4.plot(dates, strategy, color=MLGREEN, linewidth=2, label='Sentiment Strategy')

# Add drawdown region
max_so_far = np.maximum.accumulate(strategy)
drawdown = (strategy - max_so_far) / max_so_far
dd_mask = drawdown < -0.03
ax4.fill_between(dates, strategy, max_so_far, where=dd_mask,
                 color=MLRED, alpha=0.2, label='Drawdown')

ax4.set_xlabel('Trading Days')
ax4.set_ylabel('Portfolio Value ($)')
ax4.set_title('Strategy Backtest Results', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=9, loc='upper left')
ax4.grid(alpha=0.3)

# Add performance stats
final_return = (strategy[-1] / 100 - 1) * 100
bench_return = (benchmark[-1] / 100 - 1) * 100
ax4.text(70, 95, f'Strategy: +{final_return:.1f}%\nBenchmark: +{bench_return:.1f}%',
         fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
