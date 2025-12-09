"""Generate L03: Control Flow charts"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

COLOR_PRIMARY = '#9B7EBD'
COLOR_SECONDARY = '#6B5B95'
COLOR_ACCENT = '#4A90E2'
COLOR_LIGHT = '#ADADE0'
COLOR_GREEN = '#44A05B'
COLOR_ORANGE = '#FF7F0E'
COLOR_RED = '#D62728'

plt.style.use('seaborn-v0_8-whitegrid')
base_dir = Path('D:/Joerg/Research/slides/DataScience_3/L03_Control_Flow')

print('L03: Control Flow...')

# Chart 1: If-else flowchart
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'if-elif-else Decision Flow', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Start
start_box = FancyBboxPatch((3.5, 8.2), 3, 0.6, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(start_box)
ax.text(5, 8.5, 'Start: Get stock price', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_SECONDARY)

# First condition
cond1_box = FancyBboxPatch((3.2, 7.0), 3.6, 0.8, boxstyle="round,pad=0.05",
                           edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(cond1_box)
ax.text(5, 7.4, 'if price > 160?', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_ACCENT)

# True branch 1
true1_box = FancyBboxPatch((0.5, 5.8), 2.5, 0.8, boxstyle="round,pad=0.05",
                           edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(true1_box)
ax.text(1.75, 6.2, 'action = "SELL"', ha='center', va='center',
        fontsize=9, family='monospace', color=COLOR_GREEN, fontweight='bold')

# Second condition
cond2_box = FancyBboxPatch((5.5, 5.8), 3.6, 0.8, boxstyle="round,pad=0.05",
                           edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(cond2_box)
ax.text(7.3, 6.2, 'elif price < 140?', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_ACCENT)

# True branch 2
true2_box = FancyBboxPatch((5.5, 4.6), 2.5, 0.8, boxstyle="round,pad=0.05",
                           edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(true2_box)
ax.text(6.75, 5.0, 'action = "BUY"', ha='center', va='center',
        fontsize=9, family='monospace', color=COLOR_GREEN, fontweight='bold')

# Else branch
else_box = FancyBboxPatch((8.5, 4.6), 1.3, 0.8, boxstyle="round,pad=0.05",
                          edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
ax.add_patch(else_box)
ax.text(9.15, 5.0, 'HOLD', ha='center', va='center',
        fontsize=9, family='monospace', color=COLOR_ORANGE, fontweight='bold')

# End
end_box = FancyBboxPatch((3.5, 3.0), 3, 0.6, boxstyle="round,pad=0.1",
                         edgecolor=COLOR_SECONDARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(end_box)
ax.text(5, 3.3, 'End: Execute action', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_SECONDARY)

# Arrows with labels
ax.annotate('', xy=(5, 7.0), xytext=(5, 8.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=2))

ax.annotate('', xy=(1.75, 6.6), xytext=(3.5, 7.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))
ax.text(2.5, 7.0, 'True', ha='center', va='bottom',
        fontsize=9, color=COLOR_GREEN, fontweight='bold')

ax.annotate('', xy=(7.3, 5.8), xytext=(6.5, 7.0),
            arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=2))
ax.text(6.8, 6.5, 'False', ha='center', va='center',
        fontsize=9, color=COLOR_ORANGE, fontweight='bold')

ax.annotate('', xy=(6.75, 4.6), xytext=(7.3, 5.8),
            arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))
ax.text(7.5, 5.2, 'True', ha='center', va='center',
        fontsize=9, color=COLOR_GREEN, fontweight='bold')

ax.annotate('', xy=(9.15, 4.6), xytext=(8.9, 5.8),
            arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=2))
ax.text(9.5, 5.2, 'False', ha='center', va='center',
        fontsize=9, color=COLOR_ORANGE, fontweight='bold')

# All paths to end
ax.annotate('', xy=(5, 3.6), xytext=(1.75, 5.8),
            arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=1.5, alpha=0.5))
ax.annotate('', xy=(5, 3.6), xytext=(6.75, 4.6),
            arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=1.5, alpha=0.5))
ax.annotate('', xy=(5, 3.6), xytext=(9.15, 4.6),
            arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=1.5, alpha=0.5))

# Code example
code_box = FancyBboxPatch((1, 0.5), 8, 2, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_SECONDARY, facecolor='#F5F5F5', linewidth=2)
ax.add_patch(code_box)
ax.text(5, 2.3, 'Python Code', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY)

code_lines = [
    'price = 155.00',
    'if price > 160:',
    '    action = "SELL"',
    'elif price < 140:',
    '    action = "BUY"',
    'else:',
    '    action = "HOLD"',
]

y_pos = 1.9
for line in code_lines:
    ax.text(1.5, y_pos, line, ha='left', va='top',
            fontsize=9, family='monospace', color='black')
    y_pos -= 0.2

plt.tight_layout()
plt.savefig(base_dir / '01_if_else_flowchart' / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('  Chart 1/8: If-else flowchart')

# Chart 2: For loop visual
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'for Loop: Iteration Over Sequence', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# List to iterate over
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
prices = [150.50, 340.00, 125.75, 165.00]

# Original list
ax.text(1, 8.2, 'tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]', ha='left', va='top',
        fontsize=10, family='monospace', color='black', fontweight='bold')

# Loop visualization
box_width = 1.8
start_x = 1.5

for i, (ticker, price) in enumerate(zip(tickers, prices)):
    x = start_x + i * box_width

    # Iteration number
    iter_circle = Circle((x + 0.7, 7.2), 0.3, edgecolor=COLOR_PRIMARY,
                         facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(iter_circle)
    ax.text(x + 0.7, 7.2, str(i), ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_SECONDARY)

    # Item box
    item_box = FancyBboxPatch((x, 6.0), 1.4, 0.7, boxstyle="round,pad=0.05",
                              edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
    ax.add_patch(item_box)
    ax.text(x + 0.7, 6.35, f'"{ticker}"', ha='center', va='center',
            fontsize=9, family='monospace', color='black', fontweight='bold')

    # Arrow
    ax.annotate('', xy=(x + 0.7, 6.0), xytext=(x + 0.7, 6.9),
                arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=2))

    # Process box
    process_box = FancyBboxPatch((x, 4.8), 1.4, 0.9, boxstyle="round,pad=0.05",
                                 edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=1.5)
    ax.add_patch(process_box)
    ax.text(x + 0.7, 5.5, 'Process', ha='center', va='center',
            fontsize=8, fontweight='bold', color=COLOR_GREEN)
    ax.text(x + 0.7, 5.15, f'${price}', ha='center', va='center',
            fontsize=9, family='monospace', color='black')

# Loop label
ax.text(0.5, 6.35, 'for ticker in tickers:', ha='left', va='center',
        fontsize=10, family='monospace', color=COLOR_PRIMARY, fontweight='bold')

# Code example
code_box = FancyBboxPatch((0.5, 0.5), 9, 3.8, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
ax.add_patch(code_box)
ax.text(5, 4.1, 'Loop Example: Calculate Total Portfolio Value', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

code_lines = [
    'portfolio = {"AAPL": 150.50, "MSFT": 340.00, "GOOGL": 125.75}',
    'shares = {"AAPL": 100, "MSFT": 50, "GOOGL": 75}',
    '',
    'total_value = 0',
    'for ticker in portfolio:',
    '    price = portfolio[ticker]',
    '    num_shares = shares[ticker]',
    '    value = price * num_shares',
    '    total_value += value',
    '    print(f"{ticker}: ${value:,.2f}")',
    '',
    'print(f"Total: ${total_value:,.2f}")',
    '# Output: Total: $31,481.25',
]

y_pos = 3.7
for line in code_lines:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(1, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color)
    y_pos -= 0.26

plt.tight_layout()
plt.savefig(base_dir / '02_for_loop_visual' / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('  Chart 2/8: For loop visual')

# Chart 3: While loop
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'while Loop: Conditional Iteration', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Flowchart
start_box = FancyBboxPatch((3.5, 8.2), 3, 0.6, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(start_box)
ax.text(5, 8.5, 'Start: price = 100', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_SECONDARY)

# Condition
cond_box = FancyBboxPatch((3.2, 6.8), 3.6, 0.9, boxstyle="round,pad=0.05",
                          edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(cond_box)
ax.text(5, 7.25, 'while price < 150?', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_ACCENT)

# Loop body
loop_body = FancyBboxPatch((0.5, 5.0), 3, 1.2, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(loop_body)
ax.text(2, 5.9, 'Increase price:', ha='center', va='top',
        fontsize=10, fontweight='bold', color=COLOR_GREEN)
ax.text(2, 5.5, 'price *= 1.05', ha='center', va='top',
        fontsize=9, family='monospace', color='black')
ax.text(2, 5.15, '(5% increase)', ha='center', va='top',
        fontsize=8, color='#808080')

# Exit
exit_box = FancyBboxPatch((6.5, 5.0), 3, 1.2, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
ax.add_patch(exit_box)
ax.text(8, 5.9, 'Exit loop:', ha='center', va='top',
        fontsize=10, fontweight='bold', color=COLOR_ORANGE)
ax.text(8, 5.5, 'price >= 150', ha='center', va='top',
        fontsize=9, family='monospace', color='black')
ax.text(8, 5.15, 'Continue program', ha='center', va='top',
        fontsize=8, color='#808080')

# Arrows
ax.annotate('', xy=(5, 6.8), xytext=(5, 8.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=2))

ax.annotate('', xy=(2, 6.2), xytext=(3.5, 7.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))
ax.text(2.5, 6.8, 'True', ha='center', va='center',
        fontsize=9, color=COLOR_GREEN, fontweight='bold')

ax.annotate('', xy=(8, 6.2), xytext=(6.5, 7.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=2))
ax.text(7.5, 6.8, 'False', ha='center', va='center',
        fontsize=9, color=COLOR_ORANGE, fontweight='bold')

# Loop back arrow
ax.annotate('', xy=(3.2, 7.5), xytext=(0.5, 6.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_PRIMARY, lw=2,
                          connectionstyle='arc3,rad=0.3'))
ax.text(1.5, 7.5, 'Repeat', ha='center', va='center',
        fontsize=9, color=COLOR_PRIMARY, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLOR_PRIMARY))

# Code example
code_box = FancyBboxPatch((0.5, 0.3), 9, 4.2, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_SECONDARY, facecolor='#F5F5F5', linewidth=2)
ax.add_patch(code_box)
ax.text(5, 4.3, 'while Loop Example: Growth Simulation', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

code_lines = [
    '# Simulate stock price growth until target',
    'price = 100.00',
    'target = 150.00',
    'years = 0',
    'growth_rate = 0.05  # 5% per year',
    '',
    'while price < target:',
    '    price *= (1 + growth_rate)',
    '    years += 1',
    '    print(f"Year {years}: ${price:.2f}")',
    '',
    'print(f"Reached ${target} in {years} years")',
    '',
    '# Output: Reached $150.00 in 9 years',
]

y_pos = 3.9
for line in code_lines:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(1, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color)
    y_pos -= 0.26

plt.tight_layout()
plt.savefig(base_dir / '03_while_loop' / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('  Chart 3/8: While loop diagram')

# For brevity, I'll create simplified versions of remaining charts
# Chart 4: Nested loops
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Nested Loops: Loop Within a Loop', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Outer loop
outer_box = FancyBboxPatch((0.5, 6.0), 9, 2.8, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(outer_box)
ax.text(1, 8.6, 'Outer Loop: for ticker in ["AAPL", "MSFT", "GOOGL"]', ha='left', va='top',
        fontsize=10, family='monospace', fontweight='bold', color=COLOR_SECONDARY)

# Inner loop
inner_box = FancyBboxPatch((1.5, 6.5), 7, 1.8, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(inner_box)
ax.text(2, 8.1, 'Inner Loop: for day in range(5)', ha='left', va='top',
        fontsize=10, family='monospace', fontweight='bold', color=COLOR_ACCENT)

# Process
process_box = FancyBboxPatch((2.5, 6.8), 5, 1.2, boxstyle="round,pad=0.05",
                             edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=1.5)
ax.add_patch(process_box)
ax.text(5, 7.6, 'Process each ticker for each day', ha='center', va='center',
        fontsize=9, fontweight='bold', color=COLOR_GREEN)
ax.text(5, 7.2, 'Total iterations: 3 tickers × 5 days = 15', ha='center', va='center',
        fontsize=9, color='black')

# Example output
example_box = FancyBboxPatch((0.5, 0.5), 9, 5.0, boxstyle="round,pad=0.1",
                             edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
ax.add_patch(example_box)
ax.text(5, 5.3, 'Nested Loop Example: Price Matrix', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

code_lines = [
    'tickers = ["AAPL", "MSFT", "GOOGL"]',
    'days = ["Mon", "Tue", "Wed", "Thu", "Fri"]',
    '',
    'for ticker in tickers:                    # Outer loop (3 iterations)',
    '    print(f"\\n{ticker} prices:")',
    '    for day in days:                      # Inner loop (5 iterations)',
    '        price = get_price(ticker, day)    # Called 15 times total',
    '        print(f"  {day}: ${price:.2f}")',
    '',
    '# Output:',
    '# AAPL prices:',
    '#   Mon: $150.50',
    '#   Tue: $151.25',
    '#   ...',
]

y_pos = 4.9
for line in code_lines:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(1, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color)
    y_pos -= 0.3

plt.tight_layout()
plt.savefig(base_dir / '04_nested_loops' / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('  Chart 4/8: Nested loops')

# Chart 5: Break and continue
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'break vs continue: Loop Control', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Break section
break_box = FancyBboxPatch((0.3, 5.0), 4.5, 3.8, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_RED, facecolor='white', linewidth=2)
ax.add_patch(break_box)
ax.text(2.55, 8.6, 'break: Exit Loop Immediately', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_RED)

break_code = [
    'prices = [150, 165, 148, 175, 162]',
    '',
    'for price in prices:',
    '    if price < 150:',
    '        print(f"Stop! Low: ${price}")',
    '        break  # Exit loop',
    '    print(f"OK: ${price}")',
    '',
    '# Output:',
    '# OK: $150',
    '# OK: $165',
    '# Stop! Low: $148',
    '# (loop ends, 175 and 162 not processed)',
]

y_pos = 8.2
for line in break_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(0.6, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color)
    y_pos -= 0.26

# Continue section
continue_box = FancyBboxPatch((5.2, 5.0), 4.5, 3.8, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
ax.add_patch(continue_box)
ax.text(7.45, 8.6, 'continue: Skip to Next Iteration', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_ORANGE)

continue_code = [
    'prices = [150, 165, 148, 175, 162]',
    '',
    'for price in prices:',
    '    if price < 150:',
    '        print(f"Skip: ${price}")',
    '        continue  # Skip rest',
    '    print(f"Process: ${price}")',
    '',
    '# Output:',
    '# Process: $150',
    '# Process: $165',
    '# Skip: $148',
    '# Process: $175',
    '# Process: $162',
]

y_pos = 8.2
for line in continue_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(5.5, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color)
    y_pos -= 0.26

# Comparison
comp_box = FancyBboxPatch((1, 0.8), 8, 3.7, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_SECONDARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(comp_box)
ax.text(5, 4.3, 'Key Differences', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

diff_table = [
    ('break', 'Exits loop completely', 'Use when condition met', 3.6),
    ('continue', 'Skips current iteration', 'Use to filter items', 3.0),
    ('break', 'No more iterations', 'Loop terminates', 2.4),
    ('continue', 'Continues with next', 'Loop continues', 1.8),
]

for keyword, action, use_case, y in diff_table:
    if keyword == 'break':
        color = COLOR_RED
    else:
        color = COLOR_ORANGE

    ax.text(2, y, keyword, ha='center', va='center',
            fontsize=10, family='monospace', fontweight='bold', color=color)
    ax.text(4.5, y, action, ha='left', va='center',
            fontsize=9, color='black')
    ax.text(7, y, use_case, ha='left', va='center',
            fontsize=9, color='#808080', style='italic')

plt.tight_layout()
plt.savefig(base_dir / '05_break_continue' / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('  Chart 5/8: Break and continue')

# Chart 6: Trading rules decision tree
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Trading Rules: Decision Tree', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Root
root_box = FancyBboxPatch((3.5, 8.2), 3, 0.7, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(root_box)
ax.text(5, 8.55, 'Check Price & Volume', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_SECONDARY)

# Level 1: Price check
price_check = FancyBboxPatch((3.2, 6.8), 3.6, 0.8, boxstyle="round,pad=0.05",
                             edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(price_check)
ax.text(5, 7.2, 'price < buy_threshold?', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_ACCENT)

# Left branch: Check volume
vol_check = FancyBboxPatch((0.5, 5.3), 3.2, 0.8, boxstyle="round,pad=0.05",
                           edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(vol_check)
ax.text(2.1, 5.7, 'volume > min_vol?', ha='center', va='center',
        fontsize=9, fontweight='bold', color=COLOR_ACCENT)

# Buy action
buy_box = FancyBboxPatch((0.3, 3.8), 1.8, 0.9, boxstyle="round,pad=0.05",
                         edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(buy_box)
ax.text(1.2, 4.5, 'BUY', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_GREEN)
ax.text(1.2, 4.1, 'Good price\n& volume', ha='center', va='center',
        fontsize=7, color='black')

# Wait action 1
wait1_box = FancyBboxPatch((2.5, 3.8), 1.5, 0.9, boxstyle="round,pad=0.05",
                           edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
ax.add_patch(wait1_box)
ax.text(3.25, 4.5, 'WAIT', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_ORANGE)
ax.text(3.25, 4.1, 'Low\nvolume', ha='center', va='center',
        fontsize=7, color='black')

# Right branch: Check sell threshold
sell_check = FancyBboxPatch((6.2, 5.3), 3.2, 0.8, boxstyle="round,pad=0.05",
                            edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(sell_check)
ax.text(7.8, 5.7, 'price > sell_threshold?', ha='center', va='center',
        fontsize=9, fontweight='bold', color=COLOR_ACCENT)

# Sell action
sell_box = FancyBboxPatch((6.0, 3.8), 1.8, 0.9, boxstyle="round,pad=0.05",
                          edgecolor=COLOR_RED, facecolor='#FFEBEE', linewidth=2)
ax.add_patch(sell_box)
ax.text(6.9, 4.5, 'SELL', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_RED)
ax.text(6.9, 4.1, 'High\nprice', ha='center', va='center',
        fontsize=7, color='black')

# Hold action
hold_box = FancyBboxPatch((8.2, 3.8), 1.5, 0.9, boxstyle="round,pad=0.05",
                          edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
ax.add_patch(hold_box)
ax.text(8.95, 4.5, 'HOLD', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
ax.text(8.95, 4.1, 'Wait for\nbetter', ha='center', va='center',
        fontsize=7, color='black')

# Arrows
ax.annotate('', xy=(5, 6.8), xytext=(5, 8.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=2))

ax.annotate('', xy=(2.1, 6.1), xytext=(3.5, 6.9),
            arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))
ax.text(2.5, 6.5, 'True', ha='center', va='center',
        fontsize=8, color=COLOR_GREEN, fontweight='bold')

ax.annotate('', xy=(7.8, 6.1), xytext=(6.5, 6.9),
            arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=2))
ax.text(7.3, 6.5, 'False', ha='center', va='center',
        fontsize=8, color=COLOR_ORANGE, fontweight='bold')

ax.annotate('', xy=(1.2, 4.7), xytext=(1.5, 5.3),
            arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=1.5))
ax.annotate('', xy=(3.25, 4.7), xytext=(2.7, 5.3),
            arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=1.5))

ax.annotate('', xy=(6.9, 4.7), xytext=(7.2, 5.3),
            arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=1.5))
ax.annotate('', xy=(8.95, 4.7), xytext=(8.4, 5.3),
            arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=1.5))

# Code
code_box = FancyBboxPatch((0.5, 0.3), 9, 3.0, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_SECONDARY, facecolor='#F5F5F5', linewidth=2)
ax.add_patch(code_box)
ax.text(5, 3.1, 'Python Implementation', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY)

code_lines = [
    'price, volume = 145.00, 1200000',
    'buy_threshold, sell_threshold = 150.00, 170.00',
    'min_volume = 1000000',
    '',
    'if price < buy_threshold:',
    '    if volume > min_volume:',
    '        action = "BUY"',
    '    else:',
    '        action = "WAIT"',
    'elif price > sell_threshold:',
    '    action = "SELL"',
    'else:',
    '    action = "HOLD"',
]

y_pos = 2.7
for line in code_lines:
    ax.text(1, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color='black')
    y_pos -= 0.19

plt.tight_layout()
plt.savefig(base_dir / '06_trading_rules' / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('  Chart 6/8: Trading rules decision tree')

# Chart 7: Loop comparison table
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'for vs while: Loop Comparison', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Headers
header_y = 8.7
ax.text(2.5, header_y, 'for Loop', ha='center', va='center',
        fontsize=13, fontweight='bold', color=COLOR_PRIMARY)
ax.text(5.0, header_y, 'Aspect', ha='center', va='center',
        fontsize=13, fontweight='bold', color=COLOR_SECONDARY)
ax.text(7.5, header_y, 'while Loop', ha='center', va='center',
        fontsize=13, fontweight='bold', color=COLOR_ACCENT)

# Separator
ax.plot([0.5, 9.5], [8.4, 8.4], color=COLOR_SECONDARY, lw=2)

# Comparison rows
comparisons = [
    ('Iterate over sequence', 'Use Case', 'Repeat until condition', 7.8),
    ('Known iterations', 'Duration', 'Unknown iterations', 7.1),
    ('for x in sequence:', 'Syntax', 'while condition:', 6.4),
    ('Automatic', 'Increment', 'Manual', 5.7),
    ('List, range(), dict', 'Common With', 'Counters, flags', 5.0),
    ('More readable', 'Readability', 'More flexible', 4.3),
    ('Portfolio analysis', 'Finance Example', 'Price convergence', 3.6),
]

for for_val, aspect, while_val, y in comparisons:
    # for column
    for_box = FancyBboxPatch((0.5, y - 0.25), 3.5, 0.5, boxstyle="round,pad=0.05",
                             edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=1.5)
    ax.add_patch(for_box)
    ax.text(2.25, y, for_val, ha='center', va='center',
            fontsize=9, color=COLOR_PRIMARY, fontweight='bold')

    # Aspect column
    aspect_box = FancyBboxPatch((4.2, y - 0.25), 1.6, 0.5, boxstyle="round,pad=0.05",
                                edgecolor=COLOR_SECONDARY, facecolor=COLOR_LIGHT, linewidth=1.5)
    ax.add_patch(aspect_box)
    ax.text(5.0, y, aspect, ha='center', va='center',
            fontsize=9, color='black', fontweight='bold')

    # while column
    while_box = FancyBboxPatch((6.0, y - 0.25), 3.5, 0.5, boxstyle="round,pad=0.05",
                               edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
    ax.add_patch(while_box)
    ax.text(7.75, y, while_val, ha='center', va='center',
            fontsize=9, color=COLOR_ACCENT, fontweight='bold')

# Code examples
for_code_box = FancyBboxPatch((0.5, 0.3), 4.3, 2.8, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_PRIMARY, facecolor='#F5F5F5', linewidth=2)
ax.add_patch(for_code_box)
ax.text(2.65, 3.0, 'for Example', ha='center', va='top',
        fontsize=10, fontweight='bold', color=COLOR_PRIMARY)

for_lines = [
    'prices = [150, 165, 148]',
    'total = 0',
    'for price in prices:',
    '    total += price',
    'avg = total / len(prices)',
    '',
    '# 3 iterations (known)',
]

y_pos = 2.6
for line in for_lines:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(0.8, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color)
    y_pos -= 0.35

while_code_box = FancyBboxPatch((5.2, 0.3), 4.3, 2.8, boxstyle="round,pad=0.1",
                                edgecolor=COLOR_ACCENT, facecolor='#F5F5F5', linewidth=2)
ax.add_patch(while_code_box)
ax.text(7.35, 3.0, 'while Example', ha='center', va='top',
        fontsize=10, fontweight='bold', color=COLOR_ACCENT)

while_lines = [
    'price = 100',
    'target = 150',
    'while price < target:',
    '    price *= 1.05',
    '    years += 1',
    '',
    '# Unknown iterations',
]

y_pos = 2.6
for line in while_lines:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(5.5, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color)
    y_pos -= 0.35

plt.tight_layout()
plt.savefig(base_dir / '07_loop_comparison' / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('  Chart 7/8: Loop comparison')

# Chart 8: Control flow patterns
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Common Control Flow Patterns', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Pattern 1: Guard clause
pattern1_box = FancyBboxPatch((0.3, 6.8), 4.5, 2.3, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
ax.add_patch(pattern1_box)
ax.text(2.55, 8.9, 'Pattern 1: Guard Clause', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)

guard_code = [
    'def buy_stock(price, balance):',
    '    # Guard: Check preconditions',
    '    if price <= 0:',
    '        return "Invalid price"',
    '    if balance < price:',
    '        return "Insufficient funds"',
    '',
    '    # Main logic',
    '    execute_buy(price)',
]

y_pos = 8.5
for line in guard_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(0.6, y_pos, line, ha='left', va='top',
            fontsize=7, family='monospace', color=color)
    y_pos -= 0.22

# Pattern 2: Accumulator
pattern2_box = FancyBboxPatch((5.2, 6.8), 4.5, 2.3, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(pattern2_box)
ax.text(7.45, 8.9, 'Pattern 2: Accumulator', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_ACCENT)

accum_code = [
    'prices = [150, 165, 148, 172]',
    'total = 0  # Accumulator',
    '',
    'for price in prices:',
    '    total += price',
    '',
    'average = total / len(prices)',
    'print(f"Avg: ${average:.2f}")',
]

y_pos = 8.5
for line in accum_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(5.5, y_pos, line, ha='left', va='top',
            fontsize=7, family='monospace', color=color)
    y_pos -= 0.22

# Pattern 3: Search/Find
pattern3_box = FancyBboxPatch((0.3, 3.8), 4.5, 2.3, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_GREEN, facecolor='white', linewidth=2)
ax.add_patch(pattern3_box)
ax.text(2.55, 5.9, 'Pattern 3: Search & Break', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_GREEN)

search_code = [
    'prices = [150, 165, 148, 172]',
    'found = False',
    '',
    'for price in prices:',
    '    if price < 150:',
    '        print(f"Found: ${price}")',
    '        found = True',
    '        break  # Stop searching',
]

y_pos = 5.5
for line in search_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(0.6, y_pos, line, ha='left', va='top',
            fontsize=7, family='monospace', color=color)
    y_pos -= 0.22

# Pattern 4: Filter
pattern4_box = FancyBboxPatch((5.2, 3.8), 4.5, 2.3, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
ax.add_patch(pattern4_box)
ax.text(7.45, 5.9, 'Pattern 4: Filter Pattern', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_ORANGE)

filter_code = [
    'all_prices = [150, 165, 148, 172]',
    'high_prices = []  # Filtered list',
    '',
    'for price in all_prices:',
    '    if price > 160:',
    '        high_prices.append(price)',
    '',
    '# Result: [165, 172]',
]

y_pos = 5.5
for line in filter_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(5.5, y_pos, line, ha='left', va='top',
            fontsize=7, family='monospace', color=color)
    y_pos -= 0.22

# Pattern 5: Counter
pattern5_box = FancyBboxPatch((0.3, 0.8), 4.5, 2.3, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_RED, facecolor='white', linewidth=2)
ax.add_patch(pattern5_box)
ax.text(2.55, 2.9, 'Pattern 5: Counter', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_RED)

counter_code = [
    'prices = [150, 165, 148, 172, 145]',
    'count_low = 0  # Counter',
    '',
    'for price in prices:',
    '    if price < 150:',
    '        count_low += 1',
    '',
    'print(f"{count_low} low prices")',
]

y_pos = 2.5
for line in counter_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(0.6, y_pos, line, ha='left', va='top',
            fontsize=7, family='monospace', color=color)
    y_pos -= 0.22

# Pattern 6: Min/Max
pattern6_box = FancyBboxPatch((5.2, 0.8), 4.5, 2.3, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
ax.add_patch(pattern6_box)
ax.text(7.45, 2.9, 'Pattern 6: Find Min/Max', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)

minmax_code = [
    'prices = [150, 165, 148, 172]',
    'max_price = prices[0]  # Initialize',
    '',
    'for price in prices:',
    '    if price > max_price:',
    '        max_price = price',
    '',
    'print(f"Max: ${max_price}")',
]

y_pos = 2.5
for line in minmax_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(5.5, y_pos, line, ha='left', va='top',
            fontsize=7, family='monospace', color=color)
    y_pos -= 0.22

plt.tight_layout()
plt.savefig(base_dir / '08_control_patterns' / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('  Chart 8/8: Control flow patterns')

print('\nL03 COMPLETE: 8/8 charts generated')
print('='*60)
