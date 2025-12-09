"""Vocabulary Size - Controlling Feature Dimensions"""
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
fig.suptitle('Vocabulary Size: Controlling Feature Dimensions', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Word frequency distribution
ax1 = axes[0, 0]

# Zipf's law: word frequency follows power law
np.random.seed(42)
ranks = np.arange(1, 1001)
frequencies = 10000 / ranks + np.random.randn(1000) * 10

ax1.plot(ranks, frequencies, color=MLBLUE, linewidth=1)
ax1.fill_between(ranks, 0, frequencies, alpha=0.3, color=MLBLUE)

# Mark regions
ax1.axvline(50, color=MLGREEN, linestyle='--', linewidth=2, label='Top 50 (common)')
ax1.axvline(500, color=MLORANGE, linestyle='--', linewidth=2, label='Top 500 (useful)')
ax1.axvline(900, color=MLRED, linestyle='--', linewidth=2, label='Top 900 (rare)')

ax1.set_xlabel('Word Rank')
ax1.set_ylabel('Frequency')
ax1.set_title("Zipf's Law: Word Frequency Distribution", fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=8)
ax1.set_xlim(0, 1000)
ax1.grid(alpha=0.3)

# Annotate
ax1.text(100, 150, 'Common words:\n"the", "is", "and"', fontsize=8)
ax1.text(600, 50, 'Rare words:\n"algorithm", "IPO"', fontsize=8)

# Plot 2: Effect of vocabulary size on performance
ax2 = axes[0, 1]

vocab_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000]
accuracy = [0.72, 0.81, 0.85, 0.88, 0.89, 0.89, 0.88]
training_time = [1, 2, 3, 5, 10, 20, 45]

ax2.plot(vocab_sizes, accuracy, 'o-', color=MLBLUE, linewidth=2, markersize=8, label='Accuracy')

ax2_twin = ax2.twinx()
ax2_twin.plot(vocab_sizes, training_time, 's--', color=MLRED, linewidth=2, markersize=8, label='Training Time')

ax2.set_xlabel('Vocabulary Size')
ax2.set_ylabel('Accuracy', color=MLBLUE)
ax2_twin.set_ylabel('Training Time (seconds)', color=MLRED)

ax2.set_title('Vocab Size vs Performance', fontsize=11, fontweight='bold', color=MLPURPLE)

# Highlight sweet spot
ax2.axvspan(2000, 5000, alpha=0.2, color=MLGREEN)
ax2.text(3500, 0.75, 'Sweet\nspot', ha='center', fontsize=9, color=MLGREEN)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)

# Plot 3: Controlling vocabulary
ax3 = axes[1, 0]
ax3.axis('off')

control = '''
CONTROLLING VOCABULARY SIZE

METHOD 1: max_features
-----------------------
vectorizer = TfidfVectorizer(max_features=5000)
# Keep only top 5000 most frequent words


METHOD 2: min_df (minimum document frequency)
---------------------------------------------
vectorizer = TfidfVectorizer(min_df=5)
# Word must appear in at least 5 documents

vectorizer = TfidfVectorizer(min_df=0.01)
# Word must appear in at least 1% of documents

Removes: rare words, typos, proper nouns


METHOD 3: max_df (maximum document frequency)
---------------------------------------------
vectorizer = TfidfVectorizer(max_df=0.95)
# Word must appear in at most 95% of documents

Removes: very common words (like stopwords)


COMBINED APPROACH (recommended):
--------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,   # Hard limit on vocab
    min_df=2,            # Remove rare words
    max_df=0.95,         # Remove too-common words
    stop_words='english' # Remove standard stopwords
)


INSPECT YOUR VOCABULARY:
------------------------
print(len(vectorizer.vocabulary_))  # Size
print(list(vectorizer.vocabulary_.items())[:20])  # First 20 words


FINDING REMOVED WORDS:
----------------------
# Words removed by min_df
all_words = set(word for doc in documents for word in doc.split())
kept_words = set(vectorizer.vocabulary_.keys())
removed = all_words - kept_words
'''

ax3.text(0.02, 0.98, control, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Controlling Vocabulary', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Recommendations
ax4 = axes[1, 1]
ax4.axis('off')

recommendations = '''
VOCABULARY SIZE RECOMMENDATIONS

DATASET SIZE         | VOCAB SIZE  | min_df | max_df
---------------------|-------------|--------|--------
Small (< 1K docs)    | 500-2000    | 1-2    | 0.95
Medium (1K-10K docs) | 2000-5000   | 2-5    | 0.95
Large (10K-100K docs)| 5000-10000  | 5-10   | 0.90
Very large (> 100K)  | 10000-50000 | 10-20  | 0.85


TASK-SPECIFIC RECOMMENDATIONS:
------------------------------

Text Classification:
- max_features=5000 usually enough
- Include bigrams: ngram_range=(1, 2)
- max_features=10000 with bigrams

Sentiment Analysis:
- Keep more features (sentiment words vary)
- max_features=10000
- Don't aggressively filter with min_df

Topic Modeling:
- Remove very rare and very common
- min_df=5, max_df=0.80
- max_features=5000-10000

Document Similarity:
- More features = better precision
- max_features=20000
- Less filtering


DEBUGGING TIPS:
---------------
1. Start with max_features=5000
2. Check accuracy on validation set
3. Increase if accuracy plateaus early
4. Check for memory issues
5. Print feature_names to sanity check


MEMORY RULE OF THUMB:
---------------------
n_docs x vocab_size x 8 bytes < available RAM
(for dense operations)
'''

ax4.text(0.02, 0.98, recommendations, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Recommendations', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
