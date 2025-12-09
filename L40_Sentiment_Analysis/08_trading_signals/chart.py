"""Trading Signals from Sentiment - Complete Application"""
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
fig.suptitle('Trading Signals from Sentiment Analysis', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Signal generation framework
ax1 = axes[0, 0]
ax1.axis('off')

framework = '''
SENTIMENT-BASED TRADING SIGNALS

SIGNAL TYPES:
-------------
1. THRESHOLD SIGNALS
   Buy when sentiment > threshold
   Sell when sentiment < -threshold

2. MOMENTUM SIGNALS
   Buy when sentiment improving
   Sell when sentiment declining

3. MEAN REVERSION
   Buy when sentiment extremely negative
   Sell when sentiment extremely positive

4. COMBINED SIGNALS
   Multiple conditions must align


SIGNAL STRENGTH:
----------------
Not all signals are equal!

Factors affecting strength:
- Sentiment magnitude
- News volume
- Source reliability
- Consensus across sources
- Historical accuracy


EXAMPLE SIGNAL LOGIC:
---------------------
def compute_signal(row):
    sent = row['sentiment']
    sent_ma = row['sentiment_ma5']
    volume = row['news_count']

    score = 0

    # Level signal
    if sent > 0.3:
        score += 1
    elif sent < -0.3:
        score -= 1

    # Momentum signal
    if sent > sent_ma + 0.1:
        score += 1
    elif sent < sent_ma - 0.1:
        score -= 1

    # Volume confirmation
    if volume > 10 and abs(sent) > 0.4:
        score *= 1.5

    return np.clip(score, -2, 2)


OUTPUT:
-------
-2: Strong Sell
-1: Weak Sell
 0: Hold
+1: Weak Buy
+2: Strong Buy
'''

ax1.text(0.02, 0.98, framework, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Signal Generation Framework', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Signal visualization
ax2 = axes[0, 1]

# Generate synthetic trading data
np.random.seed(789)
days = 50
dates = np.arange(days)

# Sentiment
sentiment = np.sin(dates * 0.15) * 0.4 + np.random.randn(days) * 0.15
sentiment_ma = np.convolve(sentiment, np.ones(5)/5, mode='same')

# Signals based on sentiment
signals = np.zeros(days)
signals[sentiment > 0.3] = 1
signals[sentiment < -0.3] = -1

# Price (somewhat correlated with lagged sentiment)
price = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.01 + sentiment * 0.005))

# Plot price with signals
ax2.plot(dates, price, color=MLBLUE, linewidth=2, label='Price')

# Mark buy/sell signals
buy_signals = np.where(signals == 1)[0]
sell_signals = np.where(signals == -1)[0]

ax2.scatter(buy_signals, price[buy_signals], color=MLGREEN, s=100,
            marker='^', label='Buy Signal', zorder=5)
ax2.scatter(sell_signals, price[sell_signals], color=MLRED, s=100,
            marker='v', label='Sell Signal', zorder=5)

ax2.set_xlabel('Trading Days')
ax2.set_ylabel('Stock Price ($)')
ax2.set_title('Trading Signals on Price Chart', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(alpha=0.3)

# Add sentiment as secondary axis
ax2_sent = ax2.twinx()
ax2_sent.fill_between(dates, sentiment, 0, where=(sentiment >= 0),
                       color=MLGREEN, alpha=0.1)
ax2_sent.fill_between(dates, sentiment, 0, where=(sentiment < 0),
                       color=MLRED, alpha=0.1)
ax2_sent.set_ylabel('Sentiment', color='gray')
ax2_sent.set_ylim(-1, 1)

# Plot 3: Performance metrics
ax3 = axes[1, 0]

# Simulated strategy performance metrics
metrics = {
    'Total Return': ('18.5%', '12.3%'),
    'Sharpe Ratio': ('1.45', '0.89'),
    'Max Drawdown': ('-8.2%', '-12.5%'),
    'Win Rate': ('58%', '51%'),
    'Profit Factor': ('1.72', '1.15'),
    'Avg Trade': ('+0.45%', '+0.18%')
}

cell_text = [[k, v[0], v[1]] for k, v in metrics.items()]
columns = ['Metric', 'Sentiment\nStrategy', 'Buy &\nHold']

table = ax3.table(cellText=cell_text, colLabels=columns,
                  loc='center', cellLoc='center',
                  colColours=[MLLAVENDER, MLGREEN, MLBLUE])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

# Color cells based on comparison
for i in range(len(metrics)):
    for j in range(3):
        cell = table[i+1, j]
        if j == 1:  # Strategy column
            cell.set_facecolor('#E8F5E9')  # Light green
        elif j == 2:  # Benchmark column
            cell.set_facecolor('#E3F2FD')  # Light blue

ax3.axis('off')
ax3.set_title('Strategy Performance Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete implementation
ax4 = axes[1, 1]
ax4.axis('off')

implementation = '''
COMPLETE SENTIMENT TRADING SYSTEM

import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SentimentTrader:
    def __init__(self, buy_threshold=0.3, sell_threshold=-0.3):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.sia = SentimentIntensityAnalyzer()
        self.position = 0  # 0=flat, 1=long, -1=short


    def compute_sentiment(self, headlines):
        \"\"\"Compute aggregated sentiment from headlines.\"\"\"
        if not headlines:
            return 0
        scores = [self.sia.polarity_scores(h)['compound']
                  for h in headlines]
        return np.mean(scores)


    def generate_signal(self, sentiment, sentiment_ma):
        \"\"\"Generate trading signal.\"\"\"
        # Level-based
        if sentiment > self.buy_threshold:
            signal = 1
        elif sentiment < self.sell_threshold:
            signal = -1
        else:
            signal = 0

        # Momentum boost
        if sentiment > sentiment_ma + 0.15:
            signal = max(signal, 1)
        elif sentiment < sentiment_ma - 0.15:
            signal = min(signal, -1)

        return signal


    def update_position(self, signal):
        \"\"\"Update position based on signal.\"\"\"
        if signal == 1 and self.position <= 0:
            action = 'BUY'
            self.position = 1
        elif signal == -1 and self.position >= 0:
            action = 'SELL'
            self.position = -1
        else:
            action = 'HOLD'

        return action


    def run_strategy(self, price_data, news_data):
        \"\"\"Run full backtest.\"\"\"
        results = []
        for date, row in price_data.iterrows():
            headlines = news_data.get(date, [])
            sent = self.compute_sentiment(headlines)
            signal = self.generate_signal(sent, row['sent_ma'])
            action = self.update_position(signal)
            results.append({'date': date, 'action': action})
        return pd.DataFrame(results)
'''

ax4.text(0.02, 0.98, implementation, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
