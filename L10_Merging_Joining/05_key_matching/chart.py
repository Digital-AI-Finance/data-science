"""Key Matching - Different key column scenarios"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

# Standard matplotlib configuration
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

# Course colors
MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scenario 1: Same column name
ax1 = axes[0, 0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

ax1.text(5, 9.5, 'Same Column Name', ha='center', fontsize=12,
         fontweight='bold', color=MLBLUE)

# df1
box1 = FancyBboxPatch((0.5, 5), 3.5, 3.5, boxstyle="round,pad=0.1",
                      edgecolor=MLBLUE, facecolor='#E6F0FF', linewidth=2)
ax1.add_patch(box1)
ax1.text(2.25, 8.2, 'df1', ha='center', fontsize=10, fontweight='bold', color=MLBLUE)
ax1.text(1.2, 7.5, 'Ticker', fontsize=9, fontweight='bold', color=MLGREEN)
ax1.text(2.8, 7.5, 'Price', fontsize=9, fontweight='bold')
for i, (t, p) in enumerate([('AAPL', 150), ('MSFT', 350)]):
    ax1.text(1.2, 7.0 - i*0.5, t, fontsize=8)
    ax1.text(2.8, 7.0 - i*0.5, f'${p}', fontsize=8)

# df2
box2 = FancyBboxPatch((5.5, 5), 3.5, 3.5, boxstyle="round,pad=0.1",
                      edgecolor=MLORANGE, facecolor='#FFF5E6', linewidth=2)
ax1.add_patch(box2)
ax1.text(7.25, 8.2, 'df2', ha='center', fontsize=10, fontweight='bold', color=MLORANGE)
ax1.text(6.2, 7.5, 'Ticker', fontsize=9, fontweight='bold', color=MLGREEN)
ax1.text(8, 7.5, 'Sector', fontsize=9, fontweight='bold')
for i, (t, s) in enumerate([('AAPL', 'Tech'), ('MSFT', 'Tech')]):
    ax1.text(6.2, 7.0 - i*0.5, t, fontsize=8)
    ax1.text(8, 7.0 - i*0.5, s, fontsize=8)

# Code
code_box = FancyBboxPatch((1, 1), 8, 2.5, boxstyle="round,pad=0.1",
                          edgecolor=MLGREEN, facecolor='#E6FFE6', linewidth=1.5)
ax1.add_patch(code_box)
ax1.text(5, 3.0, "pd.merge(df1, df2, on='Ticker')", ha='center',
         fontsize=10, family='monospace', fontweight='bold', color=MLGREEN)
ax1.text(5, 2.2, 'Use on= when column names match', ha='center',
         fontsize=9, style='italic', color='gray')

# Scenario 2: Different column names
ax2 = axes[0, 1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

ax2.text(5, 9.5, 'Different Column Names', ha='center', fontsize=12,
         fontweight='bold', color=MLORANGE)

# df1
box1b = FancyBboxPatch((0.5, 5), 3.5, 3.5, boxstyle="round,pad=0.1",
                       edgecolor=MLBLUE, facecolor='#E6F0FF', linewidth=2)
ax2.add_patch(box1b)
ax2.text(2.25, 8.2, 'df1', ha='center', fontsize=10, fontweight='bold', color=MLBLUE)
ax2.text(1.2, 7.5, 'Symbol', fontsize=9, fontweight='bold', color=MLGREEN)
ax2.text(2.8, 7.5, 'Price', fontsize=9, fontweight='bold')
for i, (t, p) in enumerate([('AAPL', 150), ('MSFT', 350)]):
    ax2.text(1.2, 7.0 - i*0.5, t, fontsize=8)
    ax2.text(2.8, 7.0 - i*0.5, f'${p}', fontsize=8)

# df2
box2b = FancyBboxPatch((5.5, 5), 3.5, 3.5, boxstyle="round,pad=0.1",
                       edgecolor=MLORANGE, facecolor='#FFF5E6', linewidth=2)
ax2.add_patch(box2b)
ax2.text(7.25, 8.2, 'df2', ha='center', fontsize=10, fontweight='bold', color=MLORANGE)
ax2.text(6.2, 7.5, 'Ticker', fontsize=9, fontweight='bold', color=MLRED)
ax2.text(8, 7.5, 'Sector', fontsize=9, fontweight='bold')
for i, (t, s) in enumerate([('AAPL', 'Tech'), ('MSFT', 'Tech')]):
    ax2.text(6.2, 7.0 - i*0.5, t, fontsize=8)
    ax2.text(8, 7.0 - i*0.5, s, fontsize=8)

# Code
code_box2 = FancyBboxPatch((0.5, 1), 9, 2.5, boxstyle="round,pad=0.1",
                           edgecolor=MLORANGE, facecolor='#FFF5E6', linewidth=1.5)
ax2.add_patch(code_box2)
ax2.text(5, 3.0, "pd.merge(df1, df2, left_on='Symbol', right_on='Ticker')", ha='center',
         fontsize=9, family='monospace', fontweight='bold', color=MLORANGE)
ax2.text(5, 2.2, 'Use left_on/right_on for different names', ha='center',
         fontsize=9, style='italic', color='gray')

# Scenario 3: Multiple keys
ax3 = axes[1, 0]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')

ax3.text(5, 9.5, 'Multiple Keys', ha='center', fontsize=12,
         fontweight='bold', color=MLGREEN)

# df1
box1c = FancyBboxPatch((0.3, 5), 4, 3.5, boxstyle="round,pad=0.1",
                       edgecolor=MLBLUE, facecolor='#E6F0FF', linewidth=2)
ax3.add_patch(box1c)
ax3.text(2.3, 8.2, 'df1', ha='center', fontsize=10, fontweight='bold', color=MLBLUE)
headers = ['Ticker', 'Date', 'Price']
for i, h in enumerate(headers):
    color = MLGREEN if i < 2 else 'black'
    ax3.text(0.8 + i*1.3, 7.5, h, fontsize=9, fontweight='bold', color=color)
data = [('AAPL', 'Jan', 150), ('AAPL', 'Feb', 155)]
for i, (t, d, p) in enumerate(data):
    ax3.text(0.8, 7.0 - i*0.5, t, fontsize=8)
    ax3.text(2.1, 7.0 - i*0.5, d, fontsize=8)
    ax3.text(3.4, 7.0 - i*0.5, f'${p}', fontsize=8)

# df2
box2c = FancyBboxPatch((5.2, 5), 4.3, 3.5, boxstyle="round,pad=0.1",
                       edgecolor=MLORANGE, facecolor='#FFF5E6', linewidth=2)
ax3.add_patch(box2c)
ax3.text(7.35, 8.2, 'df2', ha='center', fontsize=10, fontweight='bold', color=MLORANGE)
headers2 = ['Ticker', 'Date', 'Volume']
for i, h in enumerate(headers2):
    color = MLGREEN if i < 2 else 'black'
    ax3.text(5.7 + i*1.3, 7.5, h, fontsize=9, fontweight='bold', color=color)
data2 = [('AAPL', 'Jan', '10M'), ('AAPL', 'Feb', '12M')]
for i, (t, d, v) in enumerate(data2):
    ax3.text(5.7, 7.0 - i*0.5, t, fontsize=8)
    ax3.text(7.0, 7.0 - i*0.5, d, fontsize=8)
    ax3.text(8.3, 7.0 - i*0.5, v, fontsize=8)

# Code
code_box3 = FancyBboxPatch((0.5, 1), 9, 2.5, boxstyle="round,pad=0.1",
                           edgecolor=MLGREEN, facecolor='#E6FFE6', linewidth=1.5)
ax3.add_patch(code_box3)
ax3.text(5, 3.0, "pd.merge(df1, df2, on=['Ticker', 'Date'])", ha='center',
         fontsize=10, family='monospace', fontweight='bold', color=MLGREEN)
ax3.text(5, 2.2, 'Pass list of columns for composite key', ha='center',
         fontsize=9, style='italic', color='gray')

# Scenario 4: Index-based
ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

ax4.text(5, 9.5, 'Index-Based Merge', ha='center', fontsize=12,
         fontweight='bold', color=MLPURPLE)

# df1
box1d = FancyBboxPatch((0.5, 5), 3.5, 3.5, boxstyle="round,pad=0.1",
                       edgecolor=MLBLUE, facecolor='#E6F0FF', linewidth=2)
ax4.add_patch(box1d)
ax4.text(2.25, 8.2, 'df1 (index=Ticker)', ha='center', fontsize=9, fontweight='bold', color=MLBLUE)
ax4.text(1.0, 7.5, 'Index', fontsize=8, fontweight='bold', color=MLGREEN)
ax4.text(2.8, 7.5, 'Price', fontsize=9, fontweight='bold')
for i, (t, p) in enumerate([('AAPL', 150), ('MSFT', 350)]):
    ax4.text(1.0, 7.0 - i*0.5, t, fontsize=8, color=MLGREEN)
    ax4.text(2.8, 7.0 - i*0.5, f'${p}', fontsize=8)

# df2
box2d = FancyBboxPatch((5.5, 5), 3.5, 3.5, boxstyle="round,pad=0.1",
                       edgecolor=MLORANGE, facecolor='#FFF5E6', linewidth=2)
ax4.add_patch(box2d)
ax4.text(7.25, 8.2, 'df2 (index=Ticker)', ha='center', fontsize=9, fontweight='bold', color=MLORANGE)
ax4.text(6.0, 7.5, 'Index', fontsize=8, fontweight='bold', color=MLGREEN)
ax4.text(8, 7.5, 'Sector', fontsize=9, fontweight='bold')
for i, (t, s) in enumerate([('AAPL', 'Tech'), ('MSFT', 'Tech')]):
    ax4.text(6.0, 7.0 - i*0.5, t, fontsize=8, color=MLGREEN)
    ax4.text(8, 7.0 - i*0.5, s, fontsize=8)

# Code
code_box4 = FancyBboxPatch((0.5, 1), 9, 2.5, boxstyle="round,pad=0.1",
                           edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.3, linewidth=1.5)
ax4.add_patch(code_box4)
ax4.text(5, 3.0, "df1.join(df2)  # or merge with left_index=True, right_index=True", ha='center',
         fontsize=9, family='monospace', fontweight='bold', color=MLPURPLE)
ax4.text(5, 2.2, 'Use join() for index-based merging', ha='center',
         fontsize=9, style='italic', color='gray')

fig.suptitle('Key Matching Scenarios', fontsize=16, fontweight='bold', color=MLPURPLE, y=0.99)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
