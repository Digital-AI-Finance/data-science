"""Time Series Sentiment Analysis"""
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
fig.suptitle('Time Series Sentiment Analysis', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Sentiment over time concept
ax1 = axes[0, 0]

# Generate synthetic sentiment time series
np.random.seed(42)
days = 60
dates = np.arange(days)

# Simulate sentiment with trend and events
base_sentiment = np.sin(dates * 0.1) * 0.2
noise = np.random.randn(days) * 0.15
events = np.zeros(days)
events[15] = 0.5   # Positive news
events[35] = -0.6  # Negative news
events[50] = 0.4   # Recovery

sentiment = base_sentiment + noise + events
sentiment = np.clip(sentiment, -1, 1)

# Smooth with rolling average
rolling_sentiment = np.convolve(sentiment, np.ones(5)/5, mode='same')

ax1.fill_between(dates, sentiment, 0, where=(sentiment >= 0),
                 color=MLGREEN, alpha=0.3, label='Positive')
ax1.fill_between(dates, sentiment, 0, where=(sentiment < 0),
                 color=MLRED, alpha=0.3, label='Negative')
ax1.plot(dates, rolling_sentiment, color=MLBLUE, linewidth=2, label='5-day MA')
ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)

# Mark events
ax1.annotate('Positive\nNews', xy=(15, 0.5), xytext=(20, 0.7),
            fontsize=8, arrowprops=dict(arrowstyle='->', color=MLGREEN))
ax1.annotate('Negative\nNews', xy=(35, -0.4), xytext=(40, -0.7),
            fontsize=8, arrowprops=dict(arrowstyle='->', color=MLRED))

ax1.set_xlabel('Days')
ax1.set_ylabel('Sentiment Score')
ax1.set_title('Daily Sentiment Over Time', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(alpha=0.3)
ax1.set_ylim(-1, 1)

# Plot 2: Aggregation methods
ax2 = axes[0, 1]
ax2.axis('off')

aggregation = '''
AGGREGATING SENTIMENT OVER TIME

DAILY AGGREGATION:
------------------
Multiple articles per day -> One score

Methods:
1. Simple average
   daily_sent = mean(article_sentiments)

2. Volume-weighted
   daily_sent = sum(sent * volume) / sum(volume)

3. Recency-weighted
   (more recent articles weighted higher)


ROLLING WINDOWS:
----------------
Smooth out noise with moving averages.

sentiment_ma = df['sentiment'].rolling(window=5).mean()


EXPONENTIAL SMOOTHING:
----------------------
Recent sentiment matters more.

sentiment_ema = df['sentiment'].ewm(span=5).mean()


DECAY FUNCTIONS:
----------------
Sentiment impact decays over time.

def decayed_sentiment(days_ago, half_life=3):
    return sentiment * (0.5 ** (days_ago / half_life))


EXAMPLE CODE:
-------------
# Daily aggregation
daily_sent = df.groupby('date')['sentiment'].agg([
    'mean',    # Average
    'std',     # Volatility
    'count'    # Volume
])

# Rolling sentiment
daily_sent['rolling_7d'] = daily_sent['mean'].rolling(7).mean()


CHOOSING WINDOW SIZE:
---------------------
- 3-5 days: Capture quick sentiment shifts
- 7-14 days: Smooth short-term noise
- 20-30 days: Track longer trends
'''

ax2.text(0.02, 0.98, aggregation, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Aggregation Methods', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Sentiment vs price
ax3 = axes[1, 0]

# Generate correlated sentiment and price
np.random.seed(123)
days = 60
sentiment_ts = np.cumsum(np.random.randn(days) * 0.1)
sentiment_ts = (sentiment_ts - sentiment_ts.mean()) / sentiment_ts.std() * 0.3

# Price follows sentiment with lag
price_base = 100
price_returns = np.zeros(days)
for i in range(2, days):
    price_returns[i] = sentiment_ts[i-2] * 0.02 + np.random.randn() * 0.01
price = price_base * np.exp(np.cumsum(price_returns))

# Plot
ax3_sent = ax3
ax3_price = ax3.twinx()

line1, = ax3_sent.plot(range(days), sentiment_ts, color=MLBLUE, linewidth=2, label='Sentiment')
line2, = ax3_price.plot(range(days), price, color=MLORANGE, linewidth=2, label='Price')

ax3_sent.set_xlabel('Days')
ax3_sent.set_ylabel('Sentiment Score', color=MLBLUE)
ax3_price.set_ylabel('Stock Price ($)', color=MLORANGE)
ax3.set_title('Sentiment vs Price (2-day lag)', fontsize=11, fontweight='bold', color=MLPURPLE)

ax3.legend([line1, line2], ['Sentiment (leading)', 'Price (lagging)'],
           fontsize=8, loc='upper left')
ax3_sent.grid(alpha=0.3)

# Highlight correlation
ax3.annotate('Sentiment\nleads price', xy=(25, 0.2), xytext=(35, 0.4),
            fontsize=9, bbox=dict(facecolor='white', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color=MLPURPLE))

# Plot 4: Complete pipeline code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
TIME SERIES SENTIMENT PIPELINE

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# 1. LOAD NEWS DATA
df = pd.read_csv('financial_news.csv')
df['date'] = pd.to_datetime(df['date'])


# 2. COMPUTE SENTIMENT
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['headline'].apply(
    lambda x: sia.polarity_scores(x)['compound']
)


# 3. DAILY AGGREGATION
daily = df.groupby('date').agg({
    'sentiment': ['mean', 'std', 'count']
}).reset_index()
daily.columns = ['date', 'sentiment_mean', 'sentiment_std', 'news_count']


# 4. CREATE FEATURES
daily['sent_ma5'] = daily['sentiment_mean'].rolling(5).mean()
daily['sent_ma20'] = daily['sentiment_mean'].rolling(20).mean()
daily['sent_momentum'] = daily['sent_ma5'] - daily['sent_ma20']


# 5. SENTIMENT SIGNALS
daily['signal'] = 0
daily.loc[daily['sent_momentum'] > 0.1, 'signal'] = 1   # Bullish
daily.loc[daily['sent_momentum'] < -0.1, 'signal'] = -1  # Bearish


# 6. MERGE WITH PRICE DATA
prices = pd.read_csv('stock_prices.csv')
merged = daily.merge(prices, on='date')


# 7. ANALYZE RELATIONSHIP
correlation = merged['sentiment_mean'].shift(1).corr(merged['returns'])
print(f"Sentiment-Return correlation (1-day lag): {correlation:.3f}")


# 8. VISUALIZE
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(merged['date'], merged['sent_ma5'], label='Sentiment')
ax2 = ax.twinx()
ax2.plot(merged['date'], merged['price'], color='orange', label='Price')
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Pipeline', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
