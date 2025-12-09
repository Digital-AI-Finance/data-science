"""Finance Application - Earnings Call Analysis"""
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
fig.suptitle('Finance Application: Earnings Call Text Analysis', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Problem description
ax1 = axes[0, 0]
ax1.axis('off')

problem = '''
EARNINGS CALL ANALYSIS

THE TASK:
---------
Analyze earnings call transcripts to predict
stock price movement after the call.


DATA:
-----
- Earnings call transcripts (CEO/CFO remarks)
- Analyst Q&A sections
- Historical stock returns after calls


HYPOTHESIS:
-----------
Language in earnings calls contains signals:
- Confident language -> positive returns
- Hedging language -> negative returns
- Forward guidance -> predictive power


APPROACH:
---------
1. Collect transcripts
2. Preprocess text (tokenize, clean)
3. Extract TF-IDF features
4. Train classifier (positive/negative return)
5. Evaluate on held-out data


KEY FEATURES TO EXTRACT:
------------------------
- Sentiment words (positive/negative)
- Uncertainty words (may, might, could)
- Forward-looking statements
- Numerical information
- Management tone in Q&A


CHALLENGES:
-----------
- Long documents (~10,000 words)
- Domain-specific vocabulary
- Subtle sentiment differences
- Market expectations already priced in
'''

ax1.text(0.02, 0.98, problem, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Problem Description', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Top TF-IDF terms visualization
ax2 = axes[0, 1]

# Simulated top terms by sentiment
positive_terms = ['growth', 'strong', 'exceeded', 'opportunity', 'momentum']
negative_terms = ['challenges', 'decline', 'uncertainty', 'pressure', 'headwinds']

positive_scores = [0.45, 0.42, 0.38, 0.35, 0.32]
negative_scores = [0.43, 0.40, 0.37, 0.34, 0.31]

y_pos = np.arange(len(positive_terms))

# Positive terms (right side)
bars1 = ax2.barh(y_pos + 0.2, positive_scores, height=0.35, color=MLGREEN, label='Positive Calls')
# Negative terms (left side, negative values for visualization)
bars2 = ax2.barh(y_pos - 0.2, [-s for s in negative_scores], height=0.35, color=MLRED, label='Negative Calls')

ax2.set_yticks(y_pos)
ax2.set_yticklabels([f'{p} / {n}' for p, n in zip(positive_terms, negative_terms)], fontsize=8)
ax2.axvline(0, color='black', linewidth=0.5)
ax2.set_xlabel('TF-IDF Score')
ax2.set_title('Top Distinctive Terms by Outcome', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)

ax2.set_xlim(-0.5, 0.5)

# Plot 3: Complete analysis code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
COMPLETE EARNINGS CALL ANALYSIS

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


# Load data
df = pd.read_csv('earnings_calls.csv')
# Columns: transcript, return_5day (target)

# Create binary label
df['positive'] = (df['return_5day'] > 0).astype(int)


# Preprocess
def preprocess_transcript(text):
    import re
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    return text

df['clean_text'] = df['transcript'].apply(preprocess_transcript)


# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    stop_words='english'
)

X = vectorizer.fit_transform(df['clean_text'])
y = df['positive']


# Train/test split (temporal!)
train_idx = df['date'] < '2022-01-01'
test_idx = df['date'] >= '2022-01-01'

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


# Evaluate
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")


# Top predictive features
feature_names = vectorizer.get_feature_names_out()
coefs = clf.coef_[0]

top_positive = np.argsort(coefs)[-10:]
top_negative = np.argsort(coefs)[:10]

print("Bullish signals:", [feature_names[i] for i in top_positive])
print("Bearish signals:", [feature_names[i] for i in top_negative])
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=6.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Complete Code', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Results visualization
ax4 = axes[1, 1]

# Simulated model comparison
models = ['Random\nBaseline', 'BoW\nLogistic', 'TF-IDF\nLogistic', 'TF-IDF\n+Bigrams']
auc_scores = [0.50, 0.55, 0.58, 0.61]
colors = ['gray', MLBLUE, MLGREEN, MLGREEN]

bars = ax4.bar(range(len(models)), auc_scores, color=colors, edgecolor='black')

ax4.set_xticks(range(len(models)))
ax4.set_xticklabels(models)
ax4.set_ylabel('AUC-ROC Score')
ax4.set_title('Model Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
ax4.set_ylim(0.4, 0.7)
ax4.grid(alpha=0.3, axis='y')

# Add value labels
for bar, score in zip(bars, auc_scores):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.2f}', ha='center', fontsize=10, fontweight='bold')

# Annotation
ax4.text(2.5, 0.45, 'TF-IDF + bigrams\noutperforms simple BoW', fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
