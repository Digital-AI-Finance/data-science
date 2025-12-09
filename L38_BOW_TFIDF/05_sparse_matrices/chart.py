"""Sparse Matrices - Memory Efficient Storage"""
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
fig.suptitle('Sparse Matrices: Memory-Efficient Text Representation', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Why sparse matrices
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WHY SPARSE MATRICES?

THE PROBLEM:
------------
Text data creates HUGE matrices!

Example:
- 10,000 documents
- 50,000 vocabulary words
- Matrix: 10,000 x 50,000 = 500,000,000 cells

Dense storage (float64):
  500M x 8 bytes = 4 GB of RAM!


THE SOLUTION:
-------------
Most cells are ZERO (word not in document).
Typical sparsity: 99%+ zeros!

Sparse matrices only store non-zero values.

Same matrix in sparse format:
  ~5 million non-zeros x (8 + 4 + 4) bytes
  = ~80 MB


COUNTVECTORIZER/TFIDFVECTORIZER:
--------------------------------
Return scipy sparse matrices automatically!

X = vectorizer.fit_transform(docs)
print(type(X))  # <class 'scipy.sparse._csr.csr_matrix'>


WHEN TO USE DENSE:
------------------
- Small datasets (< 1000 docs, < 10000 vocab)
- Neural networks (need dense input)
- Operations that don't support sparse

X_dense = X.toarray()  # Convert to numpy array
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Why Sparse Matrices?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual comparison
ax2 = axes[0, 1]

# Create a sample sparse matrix visualization
np.random.seed(42)
matrix = np.zeros((8, 12))
# Add some non-zero entries
for i in range(8):
    indices = np.random.choice(12, size=np.random.randint(1, 4), replace=False)
    matrix[i, indices] = np.random.rand(len(indices))

# Plot as heatmap
im = ax2.imshow(matrix, cmap='Blues', aspect='auto')

# Add grid
ax2.set_xticks(np.arange(12) - 0.5, minor=True)
ax2.set_yticks(np.arange(8) - 0.5, minor=True)
ax2.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

# Labels
ax2.set_xlabel('Vocabulary (words)')
ax2.set_ylabel('Documents')
ax2.set_title('Document-Term Matrix (Most cells = 0)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Calculate sparsity
non_zero = np.count_nonzero(matrix)
total = matrix.size
sparsity = (total - non_zero) / total * 100

ax2.text(5.5, 8.5, f'Sparsity: {sparsity:.1f}%\nNon-zeros: {non_zero}/{total}',
         fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.8))

# Plot 3: Memory comparison
ax3 = axes[1, 0]

sizes = ['1K docs\n10K vocab', '10K docs\n50K vocab', '100K docs\n100K vocab']
dense_mb = [80, 4000, 80000]  # MB
sparse_mb = [0.8, 40, 800]  # MB (assuming 1% non-zeros)

x = np.arange(len(sizes))
width = 0.35

bars1 = ax3.bar(x - width/2, dense_mb, width, label='Dense', color=MLRED, edgecolor='black')
bars2 = ax3.bar(x + width/2, sparse_mb, width, label='Sparse', color=MLGREEN, edgecolor='black')

ax3.set_yscale('log')
ax3.set_xticks(x)
ax3.set_xticklabels(sizes)
ax3.set_ylabel('Memory (MB, log scale)')
ax3.set_title('Memory: Dense vs Sparse', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis='y')

# Add ratio labels
for i, (d, s) in enumerate(zip(dense_mb, sparse_mb)):
    ax3.text(i, max(d, s) * 1.5, f'{d/s:.0f}x', ha='center', fontsize=9, fontweight='bold')

# Plot 4: Working with sparse matrices
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
WORKING WITH SPARSE MATRICES

from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp


# Create sparse matrix
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(documents)

print(type(X))          # scipy.sparse._csr.csr_matrix
print(X.shape)          # (n_docs, 5000)
print(X.nnz)            # Number of non-zero elements
print(X.data[:5])       # First 5 non-zero values


# SPARSITY INFO
sparsity = 1 - (X.nnz / (X.shape[0] * X.shape[1]))
print(f"Sparsity: {sparsity:.2%}")


# MEMORY USAGE
print(f"Dense size: {X.shape[0] * X.shape[1] * 8 / 1e6:.1f} MB")
print(f"Sparse size: {(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6:.1f} MB")


# CONVERT TO DENSE (be careful!)
X_dense = X.toarray()  # Only for small matrices!


# SKLEARN WORKS WITH SPARSE DIRECTLY
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X, y)  # No conversion needed!


# INDEXING SPARSE MATRICES
X[0, :]         # First row (returns sparse)
X[0, :].toarray()  # First row as dense
X[:, 10]        # Column 10


# COMMON OPERATIONS
X_normalized = X / X.sum(axis=1)  # Row normalization
X_subset = X[:100, :]             # First 100 docs
X_combined = sp.vstack([X1, X2])  # Stack matrices


# SAVE/LOAD
sp.save_npz('tfidf_matrix.npz', X)
X_loaded = sp.load_npz('tfidf_matrix.npz')
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Working with Sparse Matrices', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
