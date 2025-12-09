"""Sentiment Classification - ML Approach"""
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
fig.suptitle('Sentiment Classification with Machine Learning', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: ML approach overview
ax1 = axes[0, 0]
ax1.axis('off')

overview = '''
ML APPROACH TO SENTIMENT

WORKFLOW:
---------
1. Collect labeled data
2. Preprocess text
3. Extract features (TF-IDF, embeddings)
4. Train classifier
5. Evaluate and tune
6. Deploy


LABELED DATA SOURCES:
---------------------
- Financial PhraseBank (4,840 sentences)
- Sentiment140 (1.6M tweets)
- IMDB reviews (50K reviews)
- Your own labeled data


CLASSIFIERS FOR SENTIMENT:
--------------------------
Simple & Fast:
- Naive Bayes (baseline)
- Logistic Regression (interpretable)

More Powerful:
- SVM (good with TF-IDF)
- Random Forest (handles non-linear)

Best Accuracy:
- Gradient Boosting (XGBoost)
- Neural Networks (if enough data)


FEATURES:
---------
- TF-IDF vectors
- Word embeddings (average)
- Sentiment lexicon counts
- Text statistics (length, punctuation)


TYPICAL ACCURACY:
-----------------
Task            | Accuracy
----------------|----------
Binary (pos/neg)| 80-90%
3-class         | 70-80%
5-class         | 55-65%
'''

ax1.text(0.02, 0.98, overview, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('ML Approach Overview', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Model comparison
ax2 = axes[0, 1]

models = ['Naive\nBayes', 'Logistic\nReg', 'SVM', 'Random\nForest', 'XGBoost']
accuracy = [0.78, 0.82, 0.84, 0.81, 0.86]
train_time = [1, 5, 15, 30, 45]  # relative units

x = np.arange(len(models))
width = 0.35

ax2_twin = ax2.twinx()

bars1 = ax2.bar(x - width/2, accuracy, width, label='Accuracy', color=MLGREEN, edgecolor='black')
bars2 = ax2_twin.bar(x + width/2, train_time, width, label='Train Time', color=MLBLUE, edgecolor='black')

ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=9)
ax2.set_ylabel('Accuracy', color=MLGREEN)
ax2_twin.set_ylabel('Training Time (rel)', color=MLBLUE)
ax2.set_ylim(0.7, 0.9)
ax2.set_title('Model Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
ax2.grid(alpha=0.3, axis='y')

# Plot 3: Complete pipeline code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
COMPLETE SENTIMENT CLASSIFICATION PIPELINE

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# 1. LOAD DATA
df = pd.read_csv('labeled_sentiment.csv')
# Columns: 'text', 'sentiment' (positive, negative, neutral)


# 2. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'],
    test_size=0.2, random_state=42, stratify=df['sentiment']
)


# 3. FEATURE EXTRACTION
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 4. TRAIN MODEL
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'  # Handle imbalanced classes
)
model.fit(X_train_tfidf, y_train)


# 5. EVALUATE
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))


# 6. PREDICT NEW TEXT
def predict_sentiment(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return pred, max(proba)

predict_sentiment("Strong earnings growth!")
# ('positive', 0.89)
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Complete Pipeline Code', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Feature importance
ax4 = axes[1, 1]

# Simulated feature importance for sentiment
features = ['beat', 'miss', 'growth', 'decline', 'strong',
            'weak', 'exceeds', 'below', 'positive', 'negative']
importance = [0.85, -0.82, 0.75, -0.71, 0.68, -0.65, 0.62, -0.58, 0.55, -0.52]

colors = [MLGREEN if i > 0 else MLRED for i in importance]

y_pos = np.arange(len(features))
ax4.barh(y_pos, importance, color=colors, edgecolor='black')

ax4.set_yticks(y_pos)
ax4.set_yticklabels(features, fontsize=10)
ax4.set_xlabel('Feature Coefficient')
ax4.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax4.set_title('Top Features (Logistic Regression)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.grid(alpha=0.3, axis='x')

# Add labels
ax4.text(0.5, 9.5, 'Positive\nIndicators', fontsize=9, color=MLGREEN, fontweight='bold')
ax4.text(-0.5, 9.5, 'Negative\nIndicators', fontsize=9, color=MLRED, ha='right', fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
