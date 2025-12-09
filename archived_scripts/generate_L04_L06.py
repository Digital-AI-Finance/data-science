"""Generate charts for L04-L06: Functions, DataFrames, Selection/Filtering"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Course colors
MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D42728'

plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'

BASE_DIR = Path(__file__).parent

def save_chart(fig, lesson_folder, chart_name):
    """Save chart to appropriate folder"""
    chart_dir = BASE_DIR / lesson_folder / chart_name
    chart_dir.mkdir(parents=True, exist_ok=True)
    output_path = chart_dir / 'chart.pdf'
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")

# =============================================================================
# L04: Functions
# =============================================================================
def generate_L04_charts():
    print("\nL04: Functions...")

    # Chart 1: Function anatomy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Function box
    rect = patches.FancyBboxPatch((1, 2), 8, 4, boxstyle="round,pad=0.1",
                                   facecolor=MLLAVENDER, edgecolor=MLPURPLE, linewidth=2)
    ax.add_patch(rect)

    # Labels
    ax.text(5, 7, 'Function Anatomy', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)
    ax.text(1.5, 5.5, 'def calculate_return(price_old, price_new):', fontsize=11, fontfamily='monospace')
    ax.text(2, 4.5, '"""Calculate percentage return."""', fontsize=10, fontfamily='monospace', color='gray')
    ax.text(2, 3.5, 'return (price_new - price_old) / price_old * 100', fontsize=10, fontfamily='monospace')

    ax.annotate('keyword', xy=(1.5, 5.5), xytext=(0.5, 6.5), fontsize=9, color=MLORANGE,
                arrowprops=dict(arrowstyle='->', color=MLORANGE))
    ax.annotate('parameters', xy=(6, 5.5), xytext=(7, 6.5), fontsize=9, color=MLBLUE,
                arrowprops=dict(arrowstyle='->', color=MLBLUE))
    ax.annotate('docstring', xy=(3, 4.5), xytext=(6.5, 4.8), fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate('return value', xy=(2, 3.5), xytext=(0.5, 2.8), fontsize=9, color=MLGREEN,
                arrowprops=dict(arrowstyle='->', color=MLGREEN))

    ax.text(5, 1, 'Functions encapsulate reusable logic', fontsize=10, ha='center', style='italic')
    save_chart(fig, 'L04_Functions', '01_function_anatomy')
    print("  Chart 1/8: Function anatomy")

    # Chart 2: Parameter passing
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Parameter Passing', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Caller side
    rect1 = patches.FancyBboxPatch((0.5, 4), 3.5, 2.5, boxstyle="round,pad=0.1",
                                    facecolor='#E8E8FF', edgecolor=MLPURPLE, linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.25, 6, 'Caller', fontsize=11, fontweight='bold', ha='center', color=MLPURPLE)
    ax.text(2.25, 5.2, 'price = 100', fontsize=10, fontfamily='monospace', ha='center')
    ax.text(2.25, 4.5, 'ret = calc(price)', fontsize=10, fontfamily='monospace', ha='center')

    # Arrow
    ax.annotate('', xy=(6, 5.25), xytext=(4, 5.25),
                arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))
    ax.text(5, 5.8, 'value copied', fontsize=9, ha='center', color=MLORANGE)

    # Function side
    rect2 = patches.FancyBboxPatch((6, 4), 3.5, 2.5, boxstyle="round,pad=0.1",
                                    facecolor='#E8FFE8', edgecolor=MLGREEN, linewidth=2)
    ax.add_patch(rect2)
    ax.text(7.75, 6, 'Function', fontsize=11, fontweight='bold', ha='center', color=MLGREEN)
    ax.text(7.75, 5.2, 'def calc(p):', fontsize=10, fontfamily='monospace', ha='center')
    ax.text(7.75, 4.5, '  return p * 0.05', fontsize=10, fontfamily='monospace', ha='center')

    # Types
    ax.text(2.25, 2.5, 'Positional: func(a, b)', fontsize=10, fontfamily='monospace', ha='center')
    ax.text(2.25, 1.8, 'Keyword: func(x=1, y=2)', fontsize=10, fontfamily='monospace', ha='center')
    ax.text(7.75, 2.5, 'Default: def f(x=10)', fontsize=10, fontfamily='monospace', ha='center')
    ax.text(7.75, 1.8, '*args, **kwargs', fontsize=10, fontfamily='monospace', ha='center')

    save_chart(fig, 'L04_Functions', '02_parameter_passing')
    print("  Chart 2/8: Parameter passing")

    # Chart 3: Return value flowchart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Return Value Flow', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Boxes
    boxes = [
        (1, 5, 'Function Call', MLLAVENDER),
        (4, 5, 'Execute Body', '#FFE0B2'),
        (7, 5, 'Return Value', '#C8E6C9')
    ]
    for x, y, label, color in boxes:
        rect = patches.FancyBboxPatch((x, y), 2, 1.2, boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor=MLPURPLE, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+1, y+0.6, label, fontsize=10, ha='center', va='center')

    # Arrows
    ax.annotate('', xy=(4, 5.6), xytext=(3, 5.6), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
    ax.annotate('', xy=(7, 5.6), xytext=(6, 5.6), arrowprops=dict(arrowstyle='->', color=MLPURPLE))

    # Examples
    ax.text(1, 3.5, 'Single return:', fontsize=10, fontweight='bold')
    ax.text(1, 2.8, 'return price * 1.05', fontsize=10, fontfamily='monospace')

    ax.text(5, 3.5, 'Multiple returns:', fontsize=10, fontweight='bold')
    ax.text(5, 2.8, 'return mean, std', fontsize=10, fontfamily='monospace')

    ax.text(1, 1.5, 'No return (None):', fontsize=10, fontweight='bold')
    ax.text(1, 0.8, 'print("Hello")', fontsize=10, fontfamily='monospace')

    ax.text(5, 1.5, 'Early return:', fontsize=10, fontweight='bold')
    ax.text(5, 0.8, 'if x < 0: return 0', fontsize=10, fontfamily='monospace')

    save_chart(fig, 'L04_Functions', '03_return_flowchart')
    print("  Chart 3/8: Return flowchart")

    # Chart 4: Scope diagram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Variable Scope: Local vs Global', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Global scope
    rect_global = patches.FancyBboxPatch((0.5, 1), 9, 5.5, boxstyle="round,pad=0.1",
                                          facecolor='#FFF3E0', edgecolor=MLORANGE, linewidth=2)
    ax.add_patch(rect_global)
    ax.text(5, 6, 'Global Scope', fontsize=12, fontweight='bold', ha='center', color=MLORANGE)
    ax.text(1.5, 5.2, 'tax_rate = 0.15', fontsize=10, fontfamily='monospace')

    # Local scope
    rect_local = patches.FancyBboxPatch((2, 2), 6, 2.5, boxstyle="round,pad=0.1",
                                         facecolor='#E3F2FD', edgecolor=MLBLUE, linewidth=2)
    ax.add_patch(rect_local)
    ax.text(5, 4, 'Local Scope (inside function)', fontsize=11, fontweight='bold', ha='center', color=MLBLUE)
    ax.text(3, 3.2, 'def calc_tax(income):', fontsize=10, fontfamily='monospace')
    ax.text(3.5, 2.5, 'tax = income * tax_rate', fontsize=10, fontfamily='monospace')

    ax.text(5, 0.5, 'Local variables exist only during function execution', fontsize=10, ha='center', style='italic')

    save_chart(fig, 'L04_Functions', '04_scope_diagram')
    print("  Chart 4/8: Scope diagram")

    # Chart 5: Docstring format
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Docstring Best Practices', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    code = '''def calculate_sharpe(returns, rf_rate=0.02):
    """
    Calculate the Sharpe ratio for a series of returns.

    Parameters:
        returns (array): Daily return values
        rf_rate (float): Risk-free rate (default: 0.02)

    Returns:
        float: Annualized Sharpe ratio
    """
    excess = returns.mean() - rf_rate/252
    return excess / returns.std() * np.sqrt(252)'''

    rect = patches.FancyBboxPatch((0.5, 0.5), 9, 6.5, boxstyle="round,pad=0.1",
                                   facecolor='#F5F5F5', edgecolor=MLPURPLE, linewidth=1.5)
    ax.add_patch(rect)

    y_pos = 6.5
    for line in code.split('\n'):
        ax.text(1, y_pos, line, fontsize=9, fontfamily='monospace')
        y_pos -= 0.45

    save_chart(fig, 'L04_Functions', '05_docstring_format')
    print("  Chart 5/8: Docstring format")

    # Chart 6: Function call stack
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Function Call Stack', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Stack frames
    frames = [
        (3, 'main()', '#E8E8FF'),
        (2, 'calculate_portfolio()', '#D0D0FF'),
        (1, 'get_returns()', '#B8B8FF'),
        (0, 'fetch_price()', '#A0A0FF')
    ]

    for i, (level, name, color) in enumerate(frames):
        rect = patches.FancyBboxPatch((2, level * 1.5 + 1), 4, 1.2,
                                       boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor=MLPURPLE, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(4, level * 1.5 + 1.6, name, fontsize=10, ha='center', fontfamily='monospace')

    ax.annotate('', xy=(6.5, 5.5), xytext=(6.5, 1.5),
                arrowprops=dict(arrowstyle='<->', color=MLGREEN, lw=2))
    ax.text(7, 3.5, 'Call\nStack', fontsize=10, ha='left', color=MLGREEN)

    ax.text(8.5, 5.5, 'First In', fontsize=9, color='gray')
    ax.text(8.5, 1.5, 'Last Out', fontsize=9, color='gray')

    save_chart(fig, 'L04_Functions', '06_call_stack')
    print("  Chart 6/8: Call stack")

    # Chart 7: Pure vs impure functions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Pure vs Impure Functions', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Pure function
    rect1 = patches.FancyBboxPatch((0.5, 3.5), 4, 3.5, boxstyle="round,pad=0.1",
                                    facecolor='#C8E6C9', edgecolor=MLGREEN, linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.5, 6.5, 'Pure Function', fontsize=12, fontweight='bold', ha='center', color=MLGREEN)
    ax.text(2.5, 5.8, 'Same input -> Same output', fontsize=9, ha='center')
    ax.text(2.5, 5.1, 'No side effects', fontsize=9, ha='center')
    ax.text(2.5, 4.2, 'def add(a, b):', fontsize=10, fontfamily='monospace', ha='center')
    ax.text(2.5, 3.7, '  return a + b', fontsize=10, fontfamily='monospace', ha='center')

    # Impure function
    rect2 = patches.FancyBboxPatch((5.5, 3.5), 4, 3.5, boxstyle="round,pad=0.1",
                                    facecolor='#FFCDD2', edgecolor=MLRED, linewidth=2)
    ax.add_patch(rect2)
    ax.text(7.5, 6.5, 'Impure Function', fontsize=12, fontweight='bold', ha='center', color=MLRED)
    ax.text(7.5, 5.8, 'Modifies external state', fontsize=9, ha='center')
    ax.text(7.5, 5.1, 'May have side effects', fontsize=9, ha='center')
    ax.text(7.5, 4.2, 'def update(lst, x):', fontsize=10, fontfamily='monospace', ha='center')
    ax.text(7.5, 3.7, '  lst.append(x)', fontsize=10, fontfamily='monospace', ha='center')

    ax.text(5, 2.5, 'Prefer pure functions for predictable, testable code', fontsize=10, ha='center', style='italic')

    save_chart(fig, 'L04_Functions', '07_pure_vs_impure')
    print("  Chart 7/8: Pure vs impure")

    # Chart 8: Finance functions library
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Essential Finance Functions', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    functions = [
        ('calculate_return(p1, p2)', 'Price change %'),
        ('annualize_return(daily_ret)', 'Convert to yearly'),
        ('calculate_volatility(returns)', 'Standard deviation'),
        ('sharpe_ratio(ret, rf)', 'Risk-adjusted return'),
        ('max_drawdown(prices)', 'Largest peak-to-trough'),
        ('beta(stock, market)', 'Market sensitivity')
    ]

    for i, (func, desc) in enumerate(functions):
        y = 6.5 - i * 1
        rect = patches.FancyBboxPatch((0.5, y - 0.3), 5, 0.7, boxstyle="round,pad=0.02",
                                       facecolor=MLLAVENDER, edgecolor=MLPURPLE, linewidth=1)
        ax.add_patch(rect)
        ax.text(0.7, y, func, fontsize=9, fontfamily='monospace', va='center')
        ax.text(6, y, desc, fontsize=9, va='center', color='gray')

    save_chart(fig, 'L04_Functions', '08_finance_functions')
    print("  Chart 8/8: Finance functions")

    print("\nL04 COMPLETE: 8/8 charts generated")

# =============================================================================
# L05: DataFrames Introduction
# =============================================================================
def generate_L05_charts():
    print("\nL05: DataFrames Introduction...")

    # Load stock data
    stock_data = pd.read_csv(BASE_DIR / 'datasets' / 'stock_prices.csv')

    # Chart 1: DataFrame structure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'DataFrame Structure', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Column headers
    cols = ['Index', 'Date', 'AAPL', 'MSFT', 'GOOGL']
    for i, col in enumerate(cols):
        rect = patches.FancyBboxPatch((i*1.8 + 0.5, 5.5), 1.6, 0.8, boxstyle="round,pad=0.02",
                                       facecolor=MLPURPLE, edgecolor=MLPURPLE, linewidth=1)
        ax.add_patch(rect)
        ax.text(i*1.8 + 1.3, 5.9, col, fontsize=9, ha='center', color='white', fontweight='bold')

    # Data rows
    data_rows = [
        ['0', '2024-01-02', '185.2', '376.1', '140.9'],
        ['1', '2024-01-03', '184.8', '374.2', '139.5'],
        ['2', '2024-01-04', '186.1', '378.5', '141.2'],
    ]

    for row_i, row in enumerate(data_rows):
        for col_i, val in enumerate(row):
            color = '#E8E8FF' if row_i % 2 == 0 else '#F5F5FF'
            rect = patches.FancyBboxPatch((col_i*1.8 + 0.5, 4.5 - row_i*0.9), 1.6, 0.7,
                                           boxstyle="round,pad=0.02", facecolor=color,
                                           edgecolor=MLLAVENDER, linewidth=1)
            ax.add_patch(rect)
            ax.text(col_i*1.8 + 1.3, 4.85 - row_i*0.9, val, fontsize=8, ha='center')

    ax.annotate('Columns\n(features)', xy=(5, 6.3), xytext=(7.5, 6.8), fontsize=9,
                arrowprops=dict(arrowstyle='->', color=MLBLUE), color=MLBLUE)
    ax.annotate('Rows\n(observations)', xy=(0.3, 4), xytext=(-0.5, 3), fontsize=9,
                arrowprops=dict(arrowstyle='->', color=MLORANGE), color=MLORANGE)

    ax.text(5, 1.5, '2D labeled data structure with rows and columns', fontsize=10, ha='center', style='italic')

    save_chart(fig, 'L05_DataFrames_Introduction', '01_dataframe_structure')
    print("  Chart 1/8: DataFrame structure")

    # Chart 2: Series vs DataFrame
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Series vs DataFrame', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Series
    rect1 = patches.FancyBboxPatch((0.5, 2), 3.5, 4.5, boxstyle="round,pad=0.1",
                                    facecolor='#E3F2FD', edgecolor=MLBLUE, linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.25, 6, 'Series (1D)', fontsize=12, fontweight='bold', ha='center', color=MLBLUE)
    ax.text(2.25, 5.2, 'Single column', fontsize=10, ha='center')

    series_data = [('0', '185.2'), ('1', '184.8'), ('2', '186.1')]
    for i, (idx, val) in enumerate(series_data):
        ax.text(1.5, 4.3 - i*0.6, idx, fontsize=9, fontfamily='monospace')
        ax.text(2.8, 4.3 - i*0.6, val, fontsize=9, fontfamily='monospace')

    # DataFrame
    rect2 = patches.FancyBboxPatch((5.5, 2), 4, 4.5, boxstyle="round,pad=0.1",
                                    facecolor='#E8F5E9', edgecolor=MLGREEN, linewidth=2)
    ax.add_patch(rect2)
    ax.text(7.5, 6, 'DataFrame (2D)', fontsize=12, fontweight='bold', ha='center', color=MLGREEN)
    ax.text(7.5, 5.2, 'Multiple columns', fontsize=10, ha='center')

    ax.text(6.2, 4.5, 'AAPL', fontsize=8, fontweight='bold')
    ax.text(7.5, 4.5, 'MSFT', fontsize=8, fontweight='bold')
    ax.text(8.5, 4.5, 'VOL', fontsize=8, fontweight='bold')

    df_data = [('185.2', '376.1', '1.2M'), ('184.8', '374.2', '1.1M'), ('186.1', '378.5', '1.3M')]
    for i, (a, m, v) in enumerate(df_data):
        ax.text(6.2, 4 - i*0.5, a, fontsize=8, fontfamily='monospace')
        ax.text(7.5, 4 - i*0.5, m, fontsize=8, fontfamily='monospace')
        ax.text(8.5, 4 - i*0.5, v, fontsize=8, fontfamily='monospace')

    ax.text(5, 1, 'DataFrame = Collection of Series sharing an index', fontsize=10, ha='center', style='italic')

    save_chart(fig, 'L05_DataFrames_Introduction', '02_series_vs_dataframe')
    print("  Chart 2/8: Series vs DataFrame")

    # Chart 3: CSV loading flowchart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Loading CSV Data', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Flow boxes
    boxes = [
        (1, 5, 'CSV File\nstock_prices.csv', '#FFE0B2'),
        (4, 5, 'pd.read_csv()', MLLAVENDER),
        (7, 5, 'DataFrame', '#C8E6C9')
    ]

    for x, y, text, color in boxes:
        rect = patches.FancyBboxPatch((x, y), 2, 1.5, boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor=MLPURPLE, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+1, y+0.75, text, fontsize=9, ha='center', va='center')

    ax.annotate('', xy=(4, 5.75), xytext=(3, 5.75), arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))
    ax.annotate('', xy=(7, 5.75), xytext=(6, 5.75), arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))

    # Options
    ax.text(1, 3.5, 'Common Parameters:', fontsize=10, fontweight='bold')
    options = [
        'filepath: "data/prices.csv"',
        'index_col: "Date"',
        'parse_dates: True',
        'usecols: ["AAPL", "MSFT"]'
    ]
    for i, opt in enumerate(options):
        ax.text(1.5, 3 - i*0.5, opt, fontsize=9, fontfamily='monospace')

    save_chart(fig, 'L05_DataFrames_Introduction', '03_csv_loading')
    print("  Chart 3/8: CSV loading")

    # Chart 4: head/tail output
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Viewing Data: head() and tail()', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # head()
    rect1 = patches.FancyBboxPatch((0.5, 3.5), 4, 3.5, boxstyle="round,pad=0.1",
                                    facecolor='#E3F2FD', edgecolor=MLBLUE, linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.5, 6.5, 'df.head(3)', fontsize=11, fontfamily='monospace', ha='center', color=MLBLUE)
    ax.text(2.5, 5.8, 'First 3 rows', fontsize=9, ha='center', style='italic')

    head_data = ['2024-01-02  185.2', '2024-01-03  184.8', '2024-01-04  186.1']
    for i, row in enumerate(head_data):
        ax.text(2.5, 5 - i*0.5, row, fontsize=9, fontfamily='monospace', ha='center')

    # tail()
    rect2 = patches.FancyBboxPatch((5.5, 3.5), 4, 3.5, boxstyle="round,pad=0.1",
                                    facecolor='#FFF3E0', edgecolor=MLORANGE, linewidth=2)
    ax.add_patch(rect2)
    ax.text(7.5, 6.5, 'df.tail(3)', fontsize=11, fontfamily='monospace', ha='center', color=MLORANGE)
    ax.text(7.5, 5.8, 'Last 3 rows', fontsize=9, ha='center', style='italic')

    tail_data = ['2024-12-27  195.8', '2024-12-30  196.2', '2024-12-31  197.1']
    for i, row in enumerate(tail_data):
        ax.text(7.5, 5 - i*0.5, row, fontsize=9, fontfamily='monospace', ha='center')

    ax.text(5, 2.5, 'Default: 5 rows | Customize: head(10), tail(20)', fontsize=10, ha='center')

    save_chart(fig, 'L05_DataFrames_Introduction', '04_head_tail')
    print("  Chart 4/8: head/tail")

    # Chart 5: info() breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'DataFrame Info: df.info()', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    info_lines = [
        '<class pandas.DataFrame>',
        'RangeIndex: 252 entries, 0 to 251',
        'Data columns (5 columns):',
        '  Date    252 non-null datetime64',
        '  AAPL    252 non-null float64',
        '  MSFT    252 non-null float64',
        '  GOOGL   250 non-null float64  (2 missing)',
        'memory usage: 10.0 KB'
    ]

    rect = patches.FancyBboxPatch((1, 1), 8, 5.5, boxstyle="round,pad=0.1",
                                   facecolor='#F5F5F5', edgecolor=MLPURPLE, linewidth=1.5)
    ax.add_patch(rect)

    for i, line in enumerate(info_lines):
        color = MLRED if 'missing' in line else 'black'
        ax.text(1.5, 6 - i*0.6, line, fontsize=9, fontfamily='monospace', color=color)

    save_chart(fig, 'L05_DataFrames_Introduction', '05_info_breakdown')
    print("  Chart 5/8: info() breakdown")

    # Chart 6: describe() statistics
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Summary Statistics: df.describe()', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Stats table
    stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    values_aapl = ['252', '189.5', '8.2', '175.1', '183.4', '188.9', '195.2', '210.3']
    values_msft = ['252', '385.2', '12.5', '355.8', '375.6', '384.1', '394.8', '420.5']

    # Headers
    ax.text(2, 6.5, 'Stat', fontsize=10, fontweight='bold', ha='center')
    ax.text(4.5, 6.5, 'AAPL', fontsize=10, fontweight='bold', ha='center', color=MLBLUE)
    ax.text(7, 6.5, 'MSFT', fontsize=10, fontweight='bold', ha='center', color=MLGREEN)

    for i, (stat, aapl, msft) in enumerate(zip(stats, values_aapl, values_msft)):
        y = 6 - i*0.6
        ax.text(2, y, stat, fontsize=9, ha='center')
        ax.text(4.5, y, aapl, fontsize=9, ha='center', fontfamily='monospace')
        ax.text(7, y, msft, fontsize=9, ha='center', fontfamily='monospace')

    ax.axhline(y=6.3, xmin=0.1, xmax=0.9, color=MLLAVENDER, linewidth=1)

    save_chart(fig, 'L05_DataFrames_Introduction', '06_describe_stats')
    print("  Chart 6/8: describe() stats")

    # Chart 7: Index and columns
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Index and Columns', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # DataFrame representation
    rect = patches.FancyBboxPatch((2, 2), 6, 4, boxstyle="round,pad=0.1",
                                   facecolor='white', edgecolor=MLPURPLE, linewidth=2)
    ax.add_patch(rect)

    # Columns bar
    rect_cols = patches.FancyBboxPatch((2, 5.5), 6, 0.5, boxstyle="round,pad=0.02",
                                        facecolor=MLBLUE, edgecolor=MLBLUE, linewidth=1)
    ax.add_patch(rect_cols)
    ax.text(5, 5.75, 'df.columns: ["Date", "AAPL", "MSFT", "GOOGL"]', fontsize=9,
            ha='center', color='white', fontfamily='monospace')

    # Index bar
    rect_idx = patches.FancyBboxPatch((2, 2), 0.8, 3.5, boxstyle="round,pad=0.02",
                                       facecolor=MLORANGE, edgecolor=MLORANGE, linewidth=1)
    ax.add_patch(rect_idx)
    ax.text(2.4, 3.75, 'df.index', fontsize=8, rotation=90, ha='center', va='center', color='white')

    ax.text(5, 3.5, 'Data', fontsize=12, ha='center', va='center', color='gray')

    ax.text(5, 1.2, 'df.shape: (252, 4)  |  df.dtypes: column data types', fontsize=10, ha='center')

    save_chart(fig, 'L05_DataFrames_Introduction', '07_index_columns')
    print("  Chart 7/8: Index and columns")

    # Chart 8: Stock data example
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use actual stock data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    np.random.seed(42)
    prices = 185 + np.cumsum(np.random.randn(50) * 2)

    ax.plot(dates, prices, color=MLPURPLE, linewidth=2, label='AAPL')
    ax.fill_between(dates, prices, alpha=0.3, color=MLLAVENDER)

    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Price ($)', fontsize=10)
    ax.set_title('Stock Price DataFrame Visualization', fontsize=14, fontweight='bold', color=MLPURPLE)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    save_chart(fig, 'L05_DataFrames_Introduction', '08_stock_example')
    print("  Chart 8/8: Stock data example")

    print("\nL05 COMPLETE: 8/8 charts generated")

# =============================================================================
# L06: Selection and Filtering
# =============================================================================
def generate_L06_charts():
    print("\nL06: Selection and Filtering...")

    # Chart 1: Column selection syntax
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Column Selection Methods', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    methods = [
        ("df['AAPL']", 'Single column (Series)', MLBLUE),
        ("df[['AAPL', 'MSFT']]", 'Multiple columns (DataFrame)', MLGREEN),
        ("df.AAPL", 'Attribute access (simple names)', MLORANGE),
        ("df.loc[:, 'AAPL':'GOOGL']", 'Range of columns', MLPURPLE)
    ]

    for i, (code, desc, color) in enumerate(methods):
        y = 6 - i * 1.3
        rect = patches.FancyBboxPatch((1, y - 0.3), 4, 0.8, boxstyle="round,pad=0.05",
                                       facecolor='#F5F5F5', edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(3, y + 0.1, code, fontsize=10, ha='center', fontfamily='monospace')
        ax.text(6, y + 0.1, desc, fontsize=10, ha='left', color='gray')

    save_chart(fig, 'L06_Selection_Filtering', '01_column_selection')
    print("  Chart 1/8: Column selection")

    # Chart 2: iloc vs loc comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'iloc vs loc', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # iloc
    rect1 = patches.FancyBboxPatch((0.5, 3), 4, 4, boxstyle="round,pad=0.1",
                                    facecolor='#E3F2FD', edgecolor=MLBLUE, linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.5, 6.5, 'iloc (Integer Location)', fontsize=11, fontweight='bold', ha='center', color=MLBLUE)
    ax.text(2.5, 5.7, 'Position-based indexing', fontsize=9, ha='center')
    ax.text(2.5, 4.8, 'df.iloc[0]', fontsize=10, fontfamily='monospace', ha='center')
    ax.text(2.5, 4.2, 'df.iloc[0:5, 1:3]', fontsize=10, fontfamily='monospace', ha='center')
    ax.text(2.5, 3.5, 'Uses: 0, 1, 2, ...', fontsize=9, ha='center', color='gray')

    # loc
    rect2 = patches.FancyBboxPatch((5.5, 3), 4, 4, boxstyle="round,pad=0.1",
                                    facecolor='#E8F5E9', edgecolor=MLGREEN, linewidth=2)
    ax.add_patch(rect2)
    ax.text(7.5, 6.5, 'loc (Label Location)', fontsize=11, fontweight='bold', ha='center', color=MLGREEN)
    ax.text(7.5, 5.7, 'Label-based indexing', fontsize=9, ha='center')
    ax.text(7.5, 4.8, "df.loc['2024-01-02']", fontsize=10, fontfamily='monospace', ha='center')
    ax.text(7.5, 4.2, "df.loc[:, 'AAPL']", fontsize=10, fontfamily='monospace', ha='center')
    ax.text(7.5, 3.5, 'Uses: dates, names', fontsize=9, ha='center', color='gray')

    ax.text(5, 2, 'iloc: exclusive end | loc: inclusive end', fontsize=10, ha='center', style='italic')

    save_chart(fig, 'L06_Selection_Filtering', '02_iloc_vs_loc')
    print("  Chart 2/8: iloc vs loc")

    # Chart 3: Boolean mask visual
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Boolean Masking', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Original data
    ax.text(1.5, 6.5, 'AAPL', fontsize=10, fontweight='bold', ha='center')
    prices = [185, 190, 188, 195, 182]
    for i, p in enumerate(prices):
        ax.text(1.5, 5.8 - i*0.6, str(p), fontsize=10, ha='center', fontfamily='monospace')

    # Condition
    ax.text(4, 6.5, 'df["AAPL"] > 188', fontsize=10, fontfamily='monospace', ha='center', color=MLBLUE)

    # Boolean mask
    ax.text(6.5, 6.5, 'Mask', fontsize=10, fontweight='bold', ha='center')
    masks = ['False', 'True', 'False', 'True', 'False']
    colors = [MLRED, MLGREEN, MLRED, MLGREEN, MLRED]
    for i, (m, c) in enumerate(zip(masks, colors)):
        ax.text(6.5, 5.8 - i*0.6, m, fontsize=10, ha='center', fontfamily='monospace', color=c)

    # Result
    ax.text(8.5, 6.5, 'Result', fontsize=10, fontweight='bold', ha='center')
    ax.text(8.5, 5.8, '190', fontsize=10, ha='center', fontfamily='monospace', color=MLGREEN)
    ax.text(8.5, 5.2, '195', fontsize=10, ha='center', fontfamily='monospace', color=MLGREEN)

    # Arrows
    ax.annotate('', xy=(3.5, 5), xytext=(2.5, 5), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
    ax.annotate('', xy=(6, 5), xytext=(5, 5), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
    ax.annotate('', xy=(8, 5), xytext=(7, 5), arrowprops=dict(arrowstyle='->', color=MLPURPLE))

    ax.text(5, 2.5, 'Boolean mask filters rows where condition is True', fontsize=10, ha='center', style='italic')

    save_chart(fig, 'L06_Selection_Filtering', '03_boolean_mask')
    print("  Chart 3/8: Boolean mask")

    # Chart 4: Conditional filtering flowchart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Conditional Filtering Flow', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Flow
    boxes = [
        (1, 4.5, 'DataFrame\n(252 rows)', '#E8E8FF'),
        (4, 4.5, 'Condition\nprice > 190', '#FFE0B2'),
        (7, 4.5, 'Filtered\n(45 rows)', '#C8E6C9')
    ]

    for x, y, text, color in boxes:
        rect = patches.FancyBboxPatch((x, y), 2, 1.8, boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor=MLPURPLE, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+1, y+0.9, text, fontsize=10, ha='center', va='center')

    ax.annotate('', xy=(4, 5.4), xytext=(3, 5.4), arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))
    ax.annotate('', xy=(7, 5.4), xytext=(6, 5.4), arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))

    ax.text(5, 3, 'df_filtered = df[df["AAPL"] > 190]', fontsize=11, fontfamily='monospace', ha='center')

    save_chart(fig, 'L06_Selection_Filtering', '04_conditional_filtering')
    print("  Chart 4/8: Conditional filtering")

    # Chart 5: Multiple conditions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Multiple Conditions', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # AND condition
    rect1 = patches.FancyBboxPatch((0.5, 4), 4, 2.5, boxstyle="round,pad=0.1",
                                    facecolor='#E3F2FD', edgecolor=MLBLUE, linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.5, 6, 'AND: &', fontsize=12, fontweight='bold', ha='center', color=MLBLUE)
    ax.text(2.5, 5.2, '(df["AAPL"] > 185) &', fontsize=9, fontfamily='monospace', ha='center')
    ax.text(2.5, 4.6, '(df["MSFT"] > 380)', fontsize=9, fontfamily='monospace', ha='center')

    # OR condition
    rect2 = patches.FancyBboxPatch((5.5, 4), 4, 2.5, boxstyle="round,pad=0.1",
                                    facecolor='#FFF3E0', edgecolor=MLORANGE, linewidth=2)
    ax.add_patch(rect2)
    ax.text(7.5, 6, 'OR: |', fontsize=12, fontweight='bold', ha='center', color=MLORANGE)
    ax.text(7.5, 5.2, '(df["AAPL"] > 200) |', fontsize=9, fontfamily='monospace', ha='center')
    ax.text(7.5, 4.6, '(df["MSFT"] > 400)', fontsize=9, fontfamily='monospace', ha='center')

    # NOT
    rect3 = patches.FancyBboxPatch((2.5, 1), 5, 2, boxstyle="round,pad=0.1",
                                    facecolor='#FFEBEE', edgecolor=MLRED, linewidth=2)
    ax.add_patch(rect3)
    ax.text(5, 2.5, 'NOT: ~', fontsize=12, fontweight='bold', ha='center', color=MLRED)
    ax.text(5, 1.7, '~(df["AAPL"] > 190)', fontsize=9, fontfamily='monospace', ha='center')

    ax.text(5, 0.3, 'Always use parentheses around each condition!', fontsize=10, ha='center', style='italic')

    save_chart(fig, 'L06_Selection_Filtering', '05_multiple_conditions')
    print("  Chart 5/8: Multiple conditions")

    # Chart 6: Chained filtering
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Chained Filtering with query()', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Traditional
    rect1 = patches.FancyBboxPatch((0.5, 4.5), 4.5, 2.5, boxstyle="round,pad=0.1",
                                    facecolor='#F5F5F5', edgecolor='gray', linewidth=1.5)
    ax.add_patch(rect1)
    ax.text(2.75, 6.5, 'Traditional', fontsize=11, fontweight='bold', ha='center', color='gray')
    ax.text(2.75, 5.5, 'df[(df["AAPL"] > 185) &', fontsize=9, fontfamily='monospace', ha='center')
    ax.text(2.75, 5, '   (df["Volume"] > 1e6)]', fontsize=9, fontfamily='monospace', ha='center')

    # Query method
    rect2 = patches.FancyBboxPatch((5.5, 4.5), 4, 2.5, boxstyle="round,pad=0.1",
                                    facecolor='#E8F5E9', edgecolor=MLGREEN, linewidth=2)
    ax.add_patch(rect2)
    ax.text(7.5, 6.5, 'query() Method', fontsize=11, fontweight='bold', ha='center', color=MLGREEN)
    ax.text(7.5, 5.3, 'df.query("AAPL > 185 and', fontsize=9, fontfamily='monospace', ha='center')
    ax.text(7.5, 4.8, '         Volume > 1e6")', fontsize=9, fontfamily='monospace', ha='center')

    ax.text(5, 3.5, 'query() is more readable for complex filters', fontsize=10, ha='center', style='italic')

    # isin example
    rect3 = patches.FancyBboxPatch((2, 1), 6, 1.8, boxstyle="round,pad=0.1",
                                    facecolor=MLLAVENDER, edgecolor=MLPURPLE, linewidth=1.5)
    ax.add_patch(rect3)
    ax.text(5, 2.3, 'Membership: isin()', fontsize=11, fontweight='bold', ha='center', color=MLPURPLE)
    ax.text(5, 1.5, 'df[df["Symbol"].isin(["AAPL", "MSFT", "GOOGL"])]', fontsize=9, fontfamily='monospace', ha='center')

    save_chart(fig, 'L06_Selection_Filtering', '06_chained_filtering')
    print("  Chart 6/8: Chained filtering")

    # Chart 7: Selection methods comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Selection Methods Comparison', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Table
    headers = ['Method', 'Use Case', 'Returns']
    data = [
        ['df["col"]', 'Single column', 'Series'],
        ['df[["col1","col2"]]', 'Multiple columns', 'DataFrame'],
        ['df.iloc[0]', 'Row by position', 'Series'],
        ['df.loc["date"]', 'Row by label', 'Series'],
        ['df[df.col > x]', 'Filter rows', 'DataFrame']
    ]

    # Draw headers
    for i, h in enumerate(headers):
        ax.text(1.5 + i*3, 6.5, h, fontsize=10, fontweight='bold', ha='center', color=MLPURPLE)

    ax.axhline(y=6.2, xmin=0.1, xmax=0.9, color=MLLAVENDER, linewidth=2)

    # Draw data
    for row_i, row in enumerate(data):
        y = 5.7 - row_i * 0.8
        for col_i, val in enumerate(row):
            font = 'monospace' if col_i == 0 else 'sans-serif'
            ax.text(1.5 + col_i*3, y, val, fontsize=9, ha='center', fontfamily=font)

    save_chart(fig, 'L06_Selection_Filtering', '07_selection_comparison')
    print("  Chart 7/8: Selection comparison")

    # Chart 8: Stock screening workflow
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Stock Screening Workflow', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

    # Workflow steps
    steps = [
        (1, 5.5, '1. Load Data', 'pd.read_csv("stocks.csv")', '#E3F2FD'),
        (1, 4, '2. Price Filter', 'df[df["price"] > 100]', '#E8F5E9'),
        (1, 2.5, '3. Volume Filter', 'df[df["volume"] > 1e6]', '#FFF3E0'),
        (5.5, 5.5, '4. Select Columns', 'df[["symbol","price"]]', '#F3E5F5'),
        (5.5, 4, '5. Sort', 'df.sort_values("price")', '#FFEBEE'),
        (5.5, 2.5, '6. Export', 'df.to_csv("screened.csv")', '#E0F7FA')
    ]

    for x, y, title, code, color in steps:
        rect = patches.FancyBboxPatch((x, y), 3.8, 1.2, boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor=MLPURPLE, linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 1.9, y + 0.85, title, fontsize=9, fontweight='bold', ha='center')
        ax.text(x + 1.9, y + 0.35, code, fontsize=8, fontfamily='monospace', ha='center')

    # Arrows
    ax.annotate('', xy=(1.9, 4), xytext=(1.9, 5.5), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
    ax.annotate('', xy=(1.9, 2.5), xytext=(1.9, 4), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
    ax.annotate('', xy=(5.5, 5.5), xytext=(4.8, 5.5), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
    ax.annotate('', xy=(7.4, 4), xytext=(7.4, 5.5), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
    ax.annotate('', xy=(7.4, 2.5), xytext=(7.4, 4), arrowprops=dict(arrowstyle='->', color=MLPURPLE))

    ax.text(5, 1, 'Combine filters to build powerful stock screeners', fontsize=10, ha='center', style='italic')

    save_chart(fig, 'L06_Selection_Filtering', '08_stock_screening')
    print("  Chart 8/8: Stock screening")

    print("\nL06 COMPLETE: 8/8 charts generated")

# =============================================================================
# Main execution
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("GENERATING CHARTS FOR L04-L06")
    print("=" * 60)

    generate_L04_charts()
    generate_L05_charts()
    generate_L06_charts()

    print("\n" + "=" * 60)
    print("ALL CHARTS GENERATED: L04-L06 (24 charts)")
    print("=" * 60)
