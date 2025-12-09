"""Finance Story - Complete investment narrative"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3,
                       height_ratios=[0.3, 1.2, 1, 0.8])

# Title with story hook
fig.suptitle('Why Tech Outperformed: A Data-Driven Analysis', fontsize=16, fontweight='bold', color=MLPURPLE, y=0.98)

# Row 0: Story hook - headline insight
ax_hook = fig.add_subplot(gs[0, :])
ax_hook.axis('off')
ax_hook.text(0.5, 0.6, 'Tech stocks delivered 28% returns in 2024 - but why?',
             fontsize=14, ha='center', va='center', color=MLPURPLE, fontweight='bold')
ax_hook.text(0.5, 0.2, 'Three factors drove the outperformance: AI investment surge, strong earnings, and falling interest rates.',
             fontsize=11, ha='center', va='center', color='gray', style='italic')

# Row 1: Setup - Show the performance gap
ax_setup = fig.add_subplot(gs[1, :2])
months = pd.date_range('2024-01-01', periods=12, freq='ME')
tech = np.cumsum([2.5, 3.1, 1.8, -1.2, 4.5, 2.8, 3.2, 1.5, 2.1, 3.8, 2.2, 1.7])
sp500 = np.cumsum([1.5, 1.2, 0.8, -0.5, 2.1, 1.5, 1.8, 0.5, 1.2, 2.0, 1.0, 1.2])
bonds = np.cumsum([0.3, 0.2, 0.4, 0.3, 0.2, 0.3, 0.4, 0.2, 0.5, 0.3, 0.4, 0.3])

ax_setup.plot(months, tech, color=MLBLUE, linewidth=3, label='Tech Sector')
ax_setup.plot(months, sp500, color='gray', linewidth=2, linestyle='--', label='S&P 500')
ax_setup.plot(months, bonds, color=MLGREEN, linewidth=2, linestyle=':', label='Bonds')

ax_setup.fill_between(months, tech, sp500, alpha=0.3, color=MLBLUE)
ax_setup.annotate(f'Gap: {tech[-1] - sp500[-1]:.0f}%', xy=(months[-1], tech[-1]),
                  xytext=(months[-6], tech[-1] + 3),
                  fontsize=11, fontweight='bold', color=MLBLUE,
                  arrowprops=dict(arrowstyle='->', color=MLBLUE, lw=2))

ax_setup.set_title('Chapter 1: The Performance Gap', fontsize=12, fontweight='bold', color=MLPURPLE)
ax_setup.set_ylabel('Cumulative Return (%)', fontsize=10)
ax_setup.legend(fontsize=9, loc='upper left')
ax_setup.grid(alpha=0.3)
ax_setup.tick_params(axis='x', rotation=45)

# Row 1, Col 2: Contributing factors
ax_factors = fig.add_subplot(gs[1, 2])
factors = ['AI Investment', 'Earnings Growth', 'Rate Cuts', 'Margin Expansion']
contributions = [12, 8, 5, 3]
colors_factors = [MLBLUE, MLGREEN, MLORANGE, MLLAVENDER]

bars = ax_factors.barh(factors, contributions, color=colors_factors, edgecolor='black', linewidth=0.5)
ax_factors.set_title('Factor Contributions', fontsize=11, fontweight='bold', color=MLPURPLE)
ax_factors.set_xlabel('Contribution to Return (%)', fontsize=10)

for i, v in enumerate(contributions):
    ax_factors.text(v + 0.3, i, f'+{v}%', va='center', fontsize=10, fontweight='bold')

# Row 2: Evidence - Supporting data
# AI Investment surge
ax_ai = fig.add_subplot(gs[2, 0])
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
ai_capex = [45, 62, 78, 95]  # Billions
ax_ai.bar(quarters, ai_capex, color=MLBLUE, edgecolor='black', linewidth=0.5)
ax_ai.plot(quarters, ai_capex, color=MLRED, linewidth=2, marker='o', markersize=8)

# Annotate growth
ax_ai.annotate('+111% YoY', xy=(3, 95), xytext=(2, 100),
               fontsize=10, fontweight='bold', color=MLGREEN,
               arrowprops=dict(arrowstyle='->', color=MLGREEN))

ax_ai.set_title('AI CapEx Spending ($B)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax_ai.set_ylabel('Spending ($B)', fontsize=10)
ax_ai.grid(alpha=0.3, axis='y')

# Earnings growth
ax_earnings = fig.add_subplot(gs[2, 1])
companies = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'META']
earnings_growth = [125, 22, 11, 32, 68]
colors_earn = [MLGREEN if e > 20 else MLBLUE for e in earnings_growth]

ax_earnings.barh(companies, earnings_growth, color=colors_earn, edgecolor='black', linewidth=0.5)
ax_earnings.axvline(20, color=MLRED, linestyle='--', linewidth=1.5, label='S&P avg: 20%')

ax_earnings.set_title('Earnings Growth (%)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax_earnings.set_xlabel('YoY Growth (%)', fontsize=10)
ax_earnings.legend(fontsize=8)

# Rate environment
ax_rates = fig.add_subplot(gs[2, 2])
rate_months = ['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov']
fed_rate = [5.5, 5.5, 5.25, 5.0, 4.75, 4.5]
tech_pe = [28, 29, 31, 33, 35, 36]

ax_rates.plot(rate_months, fed_rate, color=MLRED, linewidth=2.5, marker='s', markersize=8, label='Fed Rate (%)')
ax_rates2 = ax_rates.twinx()
ax_rates2.plot(rate_months, tech_pe, color=MLBLUE, linewidth=2.5, marker='o', markersize=8, label='Tech P/E')

ax_rates.set_ylabel('Fed Rate (%)', fontsize=10, color=MLRED)
ax_rates2.set_ylabel('P/E Ratio', fontsize=10, color=MLBLUE)
ax_rates.set_title('Rates Fall, Valuations Rise', fontsize=11, fontweight='bold', color=MLPURPLE)

lines1, labels1 = ax_rates.get_legend_handles_labels()
lines2, labels2 = ax_rates2.get_legend_handles_labels()
ax_rates.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')

# Row 3: Conclusion and action
ax_conclusion = fig.add_subplot(gs[3, :])
ax_conclusion.axis('off')

conclusion = '''
CONCLUSION: Tech's outperformance was driven by fundamental factors - not speculation.
Strong earnings (+35% avg) justified expanding valuations as rates declined.

FORWARD OUTLOOK: AI investment cycle expected to continue through 2025.
However, valuations are stretched (P/E 36x vs historical 25x).

ACTION: Maintain tech allocation but consider adding hedges; trim positions on any
earnings disappointments. Monitor rate trajectory for valuation support.
'''

ax_conclusion.text(0.02, 0.85, conclusion, transform=ax_conclusion.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.7, edgecolor=MLPURPLE))
ax_conclusion.set_title('Chapter 4: Conclusion & Action', fontsize=12, fontweight='bold', color=MLPURPLE)

plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
