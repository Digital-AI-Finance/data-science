"""TF-IDF Formula - Term Frequency Inverse Document Frequency"""
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
fig.suptitle('TF-IDF: Term Frequency - Inverse Document Frequency', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: The formula
ax1 = axes[0, 0]
ax1.axis('off')

formula = '''
TF-IDF FORMULA

TF-IDF = TF(t, d) x IDF(t)


TERM FREQUENCY (TF):
--------------------
How often a word appears in a document.

TF(t, d) = count(t in d) / total_words(d)

or simply: count(t in d)


INVERSE DOCUMENT FREQUENCY (IDF):
---------------------------------
How rare a word is across all documents.

IDF(t) = log(N / df(t))

N = total number of documents
df(t) = number of docs containing term t


THE INTUITION:
--------------
- TF: Words appearing often in a doc are important TO THAT DOC
- IDF: Words appearing in many docs are LESS informative

High TF-IDF = frequent in this doc, rare overall
             = likely a key term for this doc!


EXAMPLE:
--------
"the" appears often but in ALL docs -> low IDF -> low TF-IDF
"algorithm" appears 5x but in few docs -> high IDF -> high TF-IDF


SKLEARN FORMULA:
----------------
TF-IDF = TF x (log((N+1)/(df+1)) + 1)

(Adds smoothing to avoid division by zero)
'''

ax1.text(0.02, 0.98, formula, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('The Formula', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: IDF visualization
ax2 = axes[0, 1]

# Simulate IDF for different document frequencies
N = 1000  # Total documents
df_values = np.array([1, 10, 50, 100, 200, 500, 900, 1000])
idf_values = np.log(N / df_values)

ax2.bar(range(len(df_values)), idf_values, color=MLBLUE, edgecolor='black')
ax2.set_xticks(range(len(df_values)))
ax2.set_xticklabels([f'{df}' for df in df_values], fontsize=8)
ax2.set_xlabel('Document Frequency (# docs containing word)')
ax2.set_ylabel('IDF Score')
ax2.set_title('IDF: Rare Words Have Higher Scores', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3, axis='y')

# Add annotations
ax2.annotate('Rare word\n(high IDF)', xy=(0, idf_values[0]), xytext=(1.5, 6),
            arrowprops=dict(arrowstyle='->', color=MLGREEN), fontsize=9, color=MLGREEN)
ax2.annotate('Common word\n(low IDF)', xy=(7, idf_values[7]), xytext=(5.5, 2),
            arrowprops=dict(arrowstyle='->', color=MLRED), fontsize=9, color=MLRED)

# Plot 3: Numerical example
ax3 = axes[1, 0]
ax3.axis('off')

example = '''
NUMERICAL EXAMPLE

CORPUS:
-------
Doc 1: "Apple earnings beat estimates"
Doc 2: "Apple stock rose today"
Doc 3: "Tech earnings surprised analysts"
Doc 4: "Market earnings season begins"

N = 4 documents

CALCULATING TF-IDF FOR "earnings" IN Doc 1:
-------------------------------------------

Step 1: TF (Term Frequency)
  TF = count("earnings" in Doc1) / words_in_Doc1
  TF = 1 / 4 = 0.25

Step 2: DF (Document Frequency)
  "earnings" appears in: Doc1, Doc3, Doc4
  DF = 3

Step 3: IDF (Inverse Document Frequency)
  IDF = log(N / DF) = log(4 / 3) = 0.29

Step 4: TF-IDF
  TF-IDF = TF x IDF = 0.25 x 0.29 = 0.07


COMPARE WITH "Apple" IN Doc 1:
------------------------------
  TF = 1/4 = 0.25
  DF = 2 (appears in Doc1, Doc2)
  IDF = log(4/2) = 0.69
  TF-IDF = 0.25 x 0.69 = 0.17

"Apple" has HIGHER TF-IDF because it's less common!


TF-IDF RANKING FOR Doc 1:
-------------------------
Apple: 0.17      (most distinctive)
beat: 0.35       (appears only in Doc1!)
estimates: 0.35  (appears only in Doc1!)
earnings: 0.07   (common across docs)
'''

ax3.text(0.02, 0.98, example, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Numerical Example', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: TF-IDF comparison visualization
ax4 = axes[1, 1]

# Compare raw counts vs TF-IDF for same words
words = ['the', 'stock', 'earnings', 'beat', 'algorithm']
raw_counts = [50, 30, 20, 5, 2]
tfidf_scores = [0.02, 0.15, 0.25, 0.45, 0.55]

x = np.arange(len(words))
width = 0.35

bars1 = ax4.bar(x - width/2, raw_counts, width, label='Raw Count', color=MLBLUE, edgecolor='black')
ax4_twin = ax4.twinx()
bars2 = ax4_twin.bar(x + width/2, tfidf_scores, width, label='TF-IDF', color=MLGREEN, edgecolor='black')

ax4.set_xticks(x)
ax4.set_xticklabels(words)
ax4.set_ylabel('Raw Count', color=MLBLUE)
ax4_twin.set_ylabel('TF-IDF Score', color=MLGREEN)

ax4.set_title('Raw Counts vs TF-IDF Scores', fontsize=11, fontweight='bold', color=MLPURPLE)

# Combine legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

ax4.text(2, 45, '"the" is common\n= low TF-IDF\n\n"algorithm" is rare\n= high TF-IDF',
         fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
