"""Finance Classification - Metrics in financial applications"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Classification Metrics in Finance', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Finance-specific metrics
ax1 = axes[0, 0]
ax1.axis('off')

finance_metrics = '''
FINANCE-SPECIFIC CLASSIFICATION METRICS

DIRECTIONAL ACCURACY (Hit Rate)
-------------------------------
= % of correct direction predictions
= (TP + TN) / Total

But wait... this is just accuracy!
In finance, >55% is often considered good.


PROFIT-WEIGHTED ACCURACY
------------------------
Weight each prediction by the profit/loss:

PW_Acc = sum(correct_return) / sum(|all_return|)

This matters more than simple accuracy!


INFORMATION COEFFICIENT (IC)
----------------------------
Correlation between predicted and actual:

IC = corr(y_pred_prob, y_actual)

IC > 0.05 is considered valuable in finance.


SHARPE RATIO OF STRATEGY
------------------------
Strategy return / Strategy volatility

The ultimate test: Does the model make money
on a risk-adjusted basis?


FALSE SIGNAL RATE
-----------------
In trading: wrong direction signals cost money

FalseBuy = bought when price went down
FalseSell = sold when price went up
'''

ax1.text(0.02, 0.98, finance_metrics, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Finance-Specific Metrics', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Confusion matrix for trading
ax2 = axes[0, 1]

# Trading confusion matrix (Up/Down predictions)
cm = np.array([[45, 30], [25, 50]])  # 53% accuracy

im = ax2.imshow(cm, cmap='RdYlGn', aspect='auto')

for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > 40 else 'black'
        if (i == j):  # Correct predictions
            ax2.text(j, i, f'{cm[i, j]}\n(+${cm[i,j]*100})', ha='center', va='center',
                     fontsize=11, fontweight='bold', color=color)
        else:  # Wrong predictions
            ax2.text(j, i, f'{cm[i, j]}\n(-${cm[i,j]*150})', ha='center', va='center',
                     fontsize=11, fontweight='bold', color=color)

ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Predicted Down', 'Predicted Up'])
ax2.set_yticklabels(['Actual Down', 'Actual Up'])
ax2.set_title('Trading Confusion Matrix', fontsize=11, fontweight='bold', color=MLPURPLE)

# Calculate metrics
accuracy = (45 + 50) / 150
profit = (45 + 50) * 100 - (30 + 25) * 150
ax2.text(0.5, -0.15, f'Accuracy: {accuracy:.1%} | Net Profit: ${profit:,}',
         transform=ax2.transAxes, ha='center', fontsize=10, fontweight='bold')

# Plot 3: Strategy performance comparison
ax3 = axes[1, 0]

# Different models with different characteristics
models = ['High Acc\nLow Profit', 'Balanced', 'High Recall\nTrader', 'High Precision\nTrader']
accuracy_scores = [0.62, 0.58, 0.52, 0.55]
sharpe_ratios = [0.5, 1.2, 0.8, 1.5]
max_drawdown = [15, 12, 25, 8]  # in percent

x = np.arange(len(models))
width = 0.25

bars1 = ax3.bar(x - width, accuracy_scores, width, label='Accuracy', color=MLBLUE, edgecolor='black')
bars2 = ax3.bar(x, [s/2 for s in sharpe_ratios], width, label='Sharpe/2', color=MLGREEN, edgecolor='black')
bars3 = ax3.bar(x + width, [d/50 for d in max_drawdown], width, label='MaxDD/50', color=MLRED, edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=9)
ax3.set_title('Model Comparison: Accuracy vs Risk Metrics', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Normalized Score', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, axis='y')

# Mark best choice
ax3.annotate('Best Risk-Adjusted!', xy=(3, 1.5/2), xytext=(2.5, 0.85),
             fontsize=9, color=MLGREEN, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=MLGREEN))

# Plot 4: Complete evaluation code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
COMPLETE TRADING MODEL EVALUATION

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_trading_model(y_true, y_pred, returns):
    """Comprehensive trading model evaluation."""

    # Standard metrics
    print("=== CLASSIFICATION METRICS ===")
    print(classification_report(y_true, y_pred,
          target_names=['Down', 'Up']))

    # Directional accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"Directional Accuracy: {accuracy:.2%}")

    # Profit calculation
    positions = np.where(y_pred == 1, 1, -1)  # Long or short
    strategy_returns = positions * returns

    total_return = strategy_returns.sum()
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    max_dd = (strategy_returns.cumsum() -
              strategy_returns.cumsum().cummax()).min()

    print(f"\\n=== FINANCIAL METRICS ===")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")

    # Hit rate by position
    long_accuracy = y_true[y_pred == 1].mean()
    short_accuracy = (1 - y_true[y_pred == 0]).mean()
    print(f"\\nLong Hit Rate: {long_accuracy:.2%}")
    print(f"Short Hit Rate: {short_accuracy:.2%}")

# Usage
evaluate_trading_model(y_test, y_pred, test_returns)
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Evaluation Code', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
