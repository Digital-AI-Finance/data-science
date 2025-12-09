"""A/B Testing - Comparing two versions"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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
fig.suptitle('A/B Testing in Finance', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Two trading strategies
ax1 = axes[0, 0]
n = 100
strategy_a = np.random.binomial(1, 0.52, n)  # 52% win rate
strategy_b = np.random.binomial(1, 0.58, n)  # 58% win rate

win_rate_a = np.mean(strategy_a)
win_rate_b = np.mean(strategy_b)

bars = ax1.bar(['Strategy A', 'Strategy B'], [win_rate_a, win_rate_b],
               color=[MLBLUE, MLGREEN], alpha=0.7, edgecolor='black')

# Add error bars (95% CI)
for i, (rate, n_obs) in enumerate([(win_rate_a, n), (win_rate_b, n)]):
    se = np.sqrt(rate * (1 - rate) / n_obs)
    ci = 1.96 * se
    ax1.errorbar([i], [rate], yerr=ci, color='black', capsize=5, capthick=2)

# Perform chi-squared test
contingency = [[sum(strategy_a), n - sum(strategy_a)],
               [sum(strategy_b), n - sum(strategy_b)]]
chi2, p_value = stats.chi2_contingency(contingency)[:2]

ax1.set_title(f'Win Rate Comparison (p = {p_value:.4f})', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('Win Rate', fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 1)

for bar, rate in zip(bars, [win_rate_a, win_rate_b]):
    ax1.text(bar.get_x() + bar.get_width()/2, rate + 0.05, f'{rate:.1%}',
             ha='center', fontsize=11, fontweight='bold')

# Plot 2: Conversion over time (cumulative)
ax2 = axes[0, 1]
days = 30
daily_a = np.random.binomial(100, 0.05, days)  # 5% conversion
daily_b = np.random.binomial(100, 0.065, days)  # 6.5% conversion

cumsum_a = np.cumsum(daily_a) / np.cumsum(np.full(days, 100))
cumsum_b = np.cumsum(daily_b) / np.cumsum(np.full(days, 100))

ax2.plot(range(1, days+1), cumsum_a * 100, color=MLBLUE, linewidth=2.5, label='Control (A)')
ax2.plot(range(1, days+1), cumsum_b * 100, color=MLGREEN, linewidth=2.5, label='Treatment (B)')
ax2.fill_between(range(1, days+1), cumsum_a * 100, cumsum_b * 100, alpha=0.2, color=MLGREEN)

ax2.set_title('Cumulative Conversion Rate', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Day', fontsize=10)
ax2.set_ylabel('Conversion Rate (%)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Sample size calculation
ax3 = axes[1, 0]
baseline_rates = [0.05, 0.10, 0.20]
lift_values = np.linspace(0.05, 0.50, 20)

for baseline, color, label in zip(baseline_rates, [MLBLUE, MLORANGE, MLGREEN],
                                   ['5% baseline', '10% baseline', '20% baseline']):
    sample_sizes = []
    for lift in lift_values:
        p1 = baseline
        p2 = baseline * (1 + lift)
        # Simplified sample size formula
        p_avg = (p1 + p2) / 2
        effect = abs(p2 - p1)
        n = (2 * (1.96 + 0.84)**2 * p_avg * (1 - p_avg)) / (effect**2)
        sample_sizes.append(n)
    ax3.plot(lift_values * 100, sample_sizes, color=color, linewidth=2.5, label=label)

ax3.set_title('Required Sample Size vs Expected Lift', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Minimum Detectable Effect (%)', fontsize=10)
ax3.set_ylabel('Sample Size per Group', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.set_yscale('log')
ax3.set_xlim(5, 50)

# Plot 4: A/B testing process
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'A/B Testing Process', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

steps = [
    ('1. Hypothesis', 'H0: No difference\nH1: B is better'),
    ('2. Sample Size', 'Calculate n for desired power\n(typically 80%)'),
    ('3. Randomize', 'Randomly assign to A or B\n(equal groups)'),
    ('4. Run Test', 'Collect data, monitor for\nissues (no peeking!)'),
    ('5. Analyze', 'Chi-squared or t-test\nat predetermined end'),
]

y = 0.78
for step, desc in steps:
    ax4.text(0.1, y, step, fontsize=11, fontweight='bold', color=MLBLUE, transform=ax4.transAxes)
    ax4.text(0.35, y, desc, fontsize=9, transform=ax4.transAxes)
    y -= 0.15

ax4.text(0.5, 0.05, 'Warning: Multiple testing (peeking) inflates Type I error!',
         ha='center', fontsize=10, style='italic', color=MLRED, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLRED))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
