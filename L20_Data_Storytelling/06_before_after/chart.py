"""Before/After - Transforming charts for clarity"""
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

fig = plt.figure(figsize=(16, 10))
fig.suptitle('Before/After: Chart Makeovers', fontsize=14, fontweight='bold', color=MLPURPLE)

# Row 1: Pie chart transformation
# BEFORE: Too many segments
ax1 = fig.add_subplot(2, 4, 1)
sectors = ['Tech', 'Finance', 'Health', 'Energy', 'Consumer', 'Utils', 'Materials', 'Industrial']
values = [25, 18, 15, 12, 10, 8, 7, 5]
colors_before = plt.cm.tab10(np.linspace(0, 1, 8))
ax1.pie(values, labels=sectors, colors=colors_before, autopct='%1.0f%%',
        textprops={'fontsize': 7})
ax1.set_title('BEFORE: Cluttered', fontsize=10, fontweight='bold', color=MLRED)

# AFTER: Grouped and simplified
ax2 = fig.add_subplot(2, 4, 2)
sectors_clean = ['Tech', 'Finance', 'Health', 'Other']
values_clean = [25, 18, 15, 42]
colors_clean = [MLBLUE, MLGREEN, MLORANGE, MLLAVENDER]
wedges, texts, autotexts = ax2.pie(values_clean, labels=sectors_clean, colors=colors_clean,
                                   autopct='%1.0f%%', startangle=90,
                                   wedgeprops=dict(edgecolor='white', linewidth=2),
                                   textprops={'fontsize': 10})
ax2.set_title('AFTER: Focused', fontsize=10, fontweight='bold', color=MLGREEN)

# Row 1: Line chart transformation
# BEFORE: No context
ax3 = fig.add_subplot(2, 4, 3)
days = np.arange(100)
price = 100 + np.cumsum(np.random.randn(100) * 2)
ax3.plot(days, price, color='blue')
ax3.set_title('BEFORE: No context', fontsize=10, fontweight='bold', color=MLRED)
ax3.set_xlabel('Days')
ax3.set_ylabel('Price')

# AFTER: Rich context
ax4 = fig.add_subplot(2, 4, 4)
ax4.plot(days, price, color=MLBLUE, linewidth=2, label='Stock Price')
ax4.axhline(100, color='gray', linestyle='--', linewidth=1.5, label='Starting Value')
ax4.fill_between(days, 100, price, where=price > 100, alpha=0.3, color=MLGREEN)
ax4.fill_between(days, 100, price, where=price < 100, alpha=0.3, color=MLRED)

# Mark peak
peak_idx = np.argmax(price)
ax4.scatter([peak_idx], [price[peak_idx]], color=MLGREEN, s=100, zorder=5, edgecolors='black')
ax4.annotate(f'Peak: ${price[peak_idx]:.0f}', xy=(peak_idx, price[peak_idx]),
             xytext=(peak_idx - 15, price[peak_idx] + 5), fontsize=9, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=MLGREEN))

ax4.set_title('AFTER: Story-driven', fontsize=10, fontweight='bold', color=MLGREEN)
ax4.set_xlabel('Trading Days', fontsize=10)
ax4.set_ylabel('Price ($)', fontsize=10)
ax4.legend(fontsize=8, loc='lower right')
ax4.grid(alpha=0.3)

# Row 2: Bar chart transformation
# BEFORE: Rainbow colors, no emphasis
ax5 = fig.add_subplot(2, 4, 5)
cats = ['A', 'B', 'C', 'D', 'E']
vals = [45, 32, 67, 28, 51]
colors_rainbow = ['red', 'orange', 'yellow', 'green', 'blue']
ax5.bar(cats, vals, color=colors_rainbow)
ax5.set_title('BEFORE: Rainbow chaos', fontsize=10, fontweight='bold', color=MLRED)

# AFTER: Strategic emphasis
ax6 = fig.add_subplot(2, 4, 6)
colors_emphasis = [MLLAVENDER, MLLAVENDER, MLGREEN, MLLAVENDER, MLLAVENDER]
bars = ax6.bar(cats, vals, color=colors_emphasis, edgecolor='black', linewidth=0.5)
ax6.bar(2, vals[2], color=MLGREEN, edgecolor='black', linewidth=1.5)  # Re-draw emphasized
ax6.annotate('Highest: 67', xy=(2, 67), xytext=(2, 75), ha='center',
             fontsize=10, fontweight='bold', color=MLGREEN,
             arrowprops=dict(arrowstyle='->', color=MLGREEN))
ax6.set_title('AFTER: Clear emphasis', fontsize=10, fontweight='bold', color=MLGREEN)
ax6.set_ylim(0, 85)
ax6.grid(alpha=0.3, axis='y')

# Row 2: Scatter transformation
# BEFORE: Just dots
ax7 = fig.add_subplot(2, 4, 7)
risk = np.random.uniform(5, 30, 30)
ret = 3 + 0.3 * risk + np.random.normal(0, 2, 30)
ax7.scatter(risk, ret, c='blue')
ax7.set_title('BEFORE: Just data', fontsize=10, fontweight='bold', color=MLRED)
ax7.set_xlabel('Risk')
ax7.set_ylabel('Return')

# AFTER: Analytical insight
ax8 = fig.add_subplot(2, 4, 8)
ax8.scatter(risk, ret, c=MLBLUE, s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

# Add regression
z = np.polyfit(risk, ret, 1)
x_line = np.linspace(5, 30, 100)
ax8.plot(x_line, np.poly1d(z)(x_line), color=MLRED, linewidth=2, linestyle='--',
         label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')

# Add quadrants
ax8.axhline(np.mean(ret), color='gray', linestyle=':', linewidth=1)
ax8.axvline(np.mean(risk), color='gray', linestyle=':', linewidth=1)

# Label quadrants
ax8.text(7, 14, 'Low Risk\nHigh Return', fontsize=8, color=MLGREEN, fontweight='bold')
ax8.text(25, 5, 'High Risk\nLow Return', fontsize=8, color=MLRED, fontweight='bold')

ax8.set_title('AFTER: With insight', fontsize=10, fontweight='bold', color=MLGREEN)
ax8.set_xlabel('Risk (%)', fontsize=10)
ax8.set_ylabel('Return (%)', fontsize=10)
ax8.legend(fontsize=8, loc='lower right')
ax8.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
