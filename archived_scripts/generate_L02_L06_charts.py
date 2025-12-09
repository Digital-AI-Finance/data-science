"""
Generate charts for L02-L06
Runs autonomously without user intervention
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Wedge
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Educational color scheme
COLOR_PRIMARY = '#9B7EBD'
COLOR_SECONDARY = '#6B5B95'
COLOR_ACCENT = '#4A90E2'
COLOR_LIGHT = '#ADADE0'
COLOR_GREEN = '#44A05B'
COLOR_ORANGE = '#FF7F0E'
COLOR_RED = '#D62728'

plt.style.use('seaborn-v0_8-whitegrid')
base_dir = Path('D:/Joerg/Research/slides/DataScience_3')

# Load stock data for charts that need it
stock_data = pd.read_csv(base_dir / 'datasets' / 'stock_prices.csv')

print('='*60)
print('GENERATING CHARTS FOR L02-L06')
print('='*60)

# ========== L02: DATA STRUCTURES ==========
print('\nL02: Data Structures...')

def l02_chart1():
    """List indexing visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'List Indexing in Python', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    # List representation
    prices = [150.00, 165.50, 148.25, 172.00, 169.75]
    n = len(prices)
    box_width = 1.3
    start_x = 2.0

    # Positive indices (top)
    ax.text(1.0, 7.5, 'Positive Indexing:', ha='left', va='center',
            fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

    for i, price in enumerate(prices):
        x = start_x + i * box_width
        # Box
        box = FancyBboxPatch((x, 6.5), box_width-0.1, 1, boxstyle="round,pad=0.05",
                             edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
        ax.add_patch(box)
        ax.text(x + (box_width-0.1)/2, 7.0, f'{price}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=COLOR_SECONDARY)

        # Index label (top)
        ax.text(x + (box_width-0.1)/2, 7.8, f'[{i}]', ha='center', va='center',
                fontsize=11, fontweight='bold', color=COLOR_ACCENT)

    # Variable name
    ax.text(start_x - 0.5, 7.0, 'prices =', ha='right', va='center',
            fontsize=11, family='monospace', color='black')

    # Negative indices (bottom)
    ax.text(1.0, 5.5, 'Negative Indexing:', ha='left', va='center',
            fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

    for i, price in enumerate(prices):
        x = start_x + i * box_width
        # Box
        box = FancyBboxPatch((x, 4.5), box_width-0.1, 1, boxstyle="round,pad=0.05",
                             edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
        ax.add_patch(box)
        ax.text(x + (box_width-0.1)/2, 5.0, f'{price}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='black')

        # Index label (bottom)
        neg_idx = i - n
        ax.text(x + (box_width-0.1)/2, 4.2, f'[{neg_idx}]', ha='center', va='center',
                fontsize=11, fontweight='bold', color=COLOR_ORANGE)

    # Examples
    examples_box = FancyBboxPatch((1.0, 1.0), 8, 2.8, boxstyle="round,pad=0.1",
                                  edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
    ax.add_patch(examples_box)
    ax.text(5, 3.6, 'Access Examples', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

    examples = [
        ('prices[0]', '150.0', 'First element'),
        ('prices[2]', '148.25', 'Third element'),
        ('prices[-1]', '169.75', 'Last element'),
        ('prices[-2]', '172.0', 'Second from end'),
    ]

    y_pos = 3.0
    for code, result, desc in examples:
        ax.text(1.5, y_pos, code, ha='left', va='top',
                fontsize=10, family='monospace', color='black', fontweight='bold')
        ax.text(3.5, y_pos, '→', ha='center', va='top',
                fontsize=12, color=COLOR_ACCENT)
        ax.text(4.0, y_pos, result, ha='left', va='top',
                fontsize=10, family='monospace', color=COLOR_GREEN, fontweight='bold')
        ax.text(5.5, y_pos, f'# {desc}', ha='left', va='top',
                fontsize=9, family='monospace', color='#808080')
        y_pos -= 0.5

    plt.tight_layout()
    plt.savefig(base_dir / 'L02_Data_Structures' / '01_list_indexing' / 'chart.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    print('  Chart 1/8: List indexing')

def l02_chart2():
    """Slicing notation"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'List Slicing: [start:stop:step]', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    # Original list
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'TSLA', 'NVDA']
    n = len(tickers)
    box_width = 1.0
    start_x = 1.5

    ax.text(0.5, 8.2, 'tickers =', ha='left', va='center',
            fontsize=11, family='monospace', fontweight='bold', color='black')

    for i, ticker in enumerate(tickers):
        x = start_x + i * box_width
        box = FancyBboxPatch((x, 7.5), box_width-0.1, 0.8, boxstyle="round,pad=0.05",
                             edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
        ax.add_patch(box)
        ax.text(x + (box_width-0.1)/2, 7.9, ticker, ha='center', va='center',
                fontsize=9, fontweight='bold', color=COLOR_SECONDARY)
        # Index
        ax.text(x + (box_width-0.1)/2, 7.2, f'[{i}]', ha='center', va='center',
                fontsize=8, color=COLOR_ACCENT)

    # Slicing examples
    slicing_examples = [
        ('tickers[0:3]', [0, 1, 2], ['AAPL', 'MSFT', 'GOOGL'], 'First 3 items', 6.0),
        ('tickers[2:5]', [2, 3, 4], ['GOOGL', 'AMZN', 'SPY'], 'Items 2 to 5', 4.5),
        ('tickers[::2]', [0, 2, 4, 6], ['AAPL', 'GOOGL', 'SPY', 'NVDA'], 'Every 2nd item', 3.0),
        ('tickers[-3:]', [4, 5, 6], ['SPY', 'TSLA', 'NVDA'], 'Last 3 items', 1.5),
    ]

    for slice_code, indices, result_list, desc, y in slicing_examples:
        # Slice code
        ax.text(0.5, y + 0.5, slice_code, ha='left', va='center',
                fontsize=10, family='monospace', fontweight='bold', color='black')

        # Result visualization
        for j, (idx, val) in enumerate(zip(indices, result_list)):
            x = start_x + j * box_width
            box = FancyBboxPatch((x, y), box_width-0.1, 0.6, boxstyle="round,pad=0.05",
                                 edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=1.5)
            ax.add_patch(box)
            ax.text(x + (box_width-0.1)/2, y + 0.3, val, ha='center', va='center',
                    fontsize=8, fontweight='bold', color=COLOR_GREEN)

        # Description
        ax.text(9.0, y + 0.3, desc, ha='right', va='center',
                fontsize=9, color='#808080', style='italic')

    plt.tight_layout()
    plt.savefig(base_dir / 'L02_Data_Structures' / '02_slicing_notation' / 'chart.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    print('  Chart 2/8: Slicing notation')

def l02_chart3():
    """Dictionary structure"""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Dictionary: Key-Value Pairs', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    # Dictionary representation
    portfolio = {
        'AAPL': 150.50,
        'MSFT': 340.00,
        'GOOGL': 125.75,
        'AMZN': 165.00
    }

    # Visual representation
    ax.text(5, 8.7, 'portfolio = { }', ha='center', va='top',
            fontsize=12, family='monospace', fontweight='bold', color='black')

    y_pos = 7.8
    for ticker, price in portfolio.items():
        # Key box
        key_box = FancyBboxPatch((2.0, y_pos - 0.4), 2, 0.7, boxstyle="round,pad=0.05",
                                 edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
        ax.add_patch(key_box)
        ax.text(3.0, y_pos, f'"{ticker}"', ha='center', va='center',
                fontsize=10, family='monospace', fontweight='bold', color=COLOR_SECONDARY)

        # Arrow
        ax.text(4.3, y_pos, ':', ha='center', va='center',
                fontsize=14, fontweight='bold', color=COLOR_ACCENT)

        # Value box
        value_box = FancyBboxPatch((4.8, y_pos - 0.4), 2, 0.7, boxstyle="round,pad=0.05",
                                   edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
        ax.add_patch(value_box)
        ax.text(5.8, y_pos, f'{price}', ha='center', va='center',
                fontsize=10, family='monospace', fontweight='bold', color=COLOR_GREEN)

        # Labels (only for first row)
        if y_pos == 7.8:
            ax.text(3.0, y_pos + 0.6, 'KEY', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=COLOR_PRIMARY)
            ax.text(5.8, y_pos + 0.6, 'VALUE', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=COLOR_GREEN)

        y_pos -= 1.0

    # Access examples
    examples_box = FancyBboxPatch((1.0, 1.0), 8, 2.8, boxstyle="round,pad=0.1",
                                  edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
    ax.add_patch(examples_box)
    ax.text(5, 3.6, 'Dictionary Operations', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

    operations = [
        ('portfolio["AAPL"]', '150.5', 'Access value by key'),
        ('portfolio["AAPL"] = 155.00', 'None', 'Update value'),
        ('portfolio["TSLA"] = 250.00', 'None', 'Add new key-value pair'),
        ('"MSFT" in portfolio', 'True', 'Check if key exists'),
    ]

    y_pos = 3.0
    for code, result, desc in operations:
        ax.text(1.5, y_pos, code, ha='left', va='top',
                fontsize=9, family='monospace', color='black', fontweight='bold')
        if result != 'None':
            ax.text(5.5, y_pos, '→', ha='center', va='top',
                    fontsize=11, color=COLOR_ACCENT)
            ax.text(6.0, y_pos, result, ha='left', va='top',
                    fontsize=9, family='monospace', color=COLOR_GREEN, fontweight='bold')
        ax.text(7.2, y_pos, f'# {desc}', ha='left', va='top',
                fontsize=8, family='monospace', color='#808080')
        y_pos -= 0.6

    plt.tight_layout()
    plt.savefig(base_dir / 'L02_Data_Structures' / '03_dictionary_structure' / 'chart.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    print('  Chart 3/8: Dictionary structure')

def l02_chart4():
    """Nested data structures"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Nested Data Structures', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    # Portfolio with nested structure
    ax.text(5, 8.8, 'portfolio = {', ha='center', va='top',
            fontsize=11, family='monospace', fontweight='bold', color='black')

    stocks = [
        ('AAPL', 150.50, 100, 15050.00),
        ('MSFT', 340.00, 50, 17000.00),
        ('GOOGL', 125.75, 75, 9431.25),
    ]

    y_pos = 8.0
    for ticker, price, shares, value in stocks:
        # Outer key
        outer_box = FancyBboxPatch((1.5, y_pos - 0.3), 1.5, 0.6, boxstyle="round,pad=0.05",
                                   edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
        ax.add_patch(outer_box)
        ax.text(2.25, y_pos, f'"{ticker}"', ha='center', va='center',
                fontsize=9, family='monospace', fontweight='bold', color=COLOR_SECONDARY)

        ax.text(3.2, y_pos, ':', ha='center', va='center',
                fontsize=12, fontweight='bold', color='black')

        # Nested dictionary
        nested_box = FancyBboxPatch((3.5, y_pos - 0.5), 5.5, 0.9, boxstyle="round,pad=0.05",
                                    edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
        ax.add_patch(nested_box)

        nested_text = f'{{"price": {price}, "shares": {shares}, "value": {value}}}'
        ax.text(6.25, y_pos, nested_text, ha='center', va='center',
                fontsize=8, family='monospace', color='black')

        y_pos -= 1.2

    ax.text(5, y_pos, '}', ha='center', va='top',
            fontsize=11, family='monospace', fontweight='bold', color='black')

    # Access examples
    examples_box = FancyBboxPatch((0.5, 1.0), 9, 3.2, boxstyle="round,pad=0.1",
                                  edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
    ax.add_patch(examples_box)
    ax.text(5, 4.0, 'Accessing Nested Data', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

    access_examples = [
        ('portfolio["AAPL"]', '{"price": 150.5, "shares": 100, ...}', 'Full nested dict'),
        ('portfolio["AAPL"]["price"]', '150.5', 'Specific value'),
        ('portfolio["MSFT"]["shares"]', '50', 'Shares for MSFT'),
        ('portfolio["GOOGL"]["value"]', '9431.25', 'Total value'),
    ]

    y_pos = 3.4
    for code, result, desc in access_examples:
        ax.text(1.0, y_pos, code, ha='left', va='top',
                fontsize=9, family='monospace', color='black', fontweight='bold')
        ax.text(4.8, y_pos, '→', ha='center', va='top',
                fontsize=11, color=COLOR_ACCENT)
        ax.text(5.2, y_pos, result, ha='left', va='top',
                fontsize=8, family='monospace', color=COLOR_GREEN, fontweight='bold')
        ax.text(7.5, y_pos, f'# {desc}', ha='left', va='top',
                fontsize=8, family='monospace', color='#808080')
        y_pos -= 0.6

    plt.tight_layout()
    plt.savefig(base_dir / 'L02_Data_Structures' / '04_nested_structures' / 'chart.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    print('  Chart 4/8: Nested structures')

def l02_chart5():
    """List methods comparison"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Common List Methods', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    methods = [
        ('append()', 'Add item to end', 'prices.append(175.00)', '[150, 165, 175]', COLOR_GREEN, 8.2),
        ('insert()', 'Add item at position', 'prices.insert(1, 160)', '[150, 160, 165]', COLOR_GREEN, 7.0),
        ('remove()', 'Remove first occurrence', 'prices.remove(165)', '[150]', COLOR_ORANGE, 5.8),
        ('pop()', 'Remove and return item', 'prices.pop()', 'Returns: 165', COLOR_ORANGE, 4.6),
        ('sort()', 'Sort list in place', 'prices.sort()', '[150, 165, 175]', COLOR_ACCENT, 3.4),
        ('reverse()', 'Reverse list order', 'prices.reverse()', '[165, 150]', COLOR_ACCENT, 2.2),
        ('count()', 'Count occurrences', 'prices.count(150)', '1', COLOR_PRIMARY, 1.0),
    ]

    for method, desc, example, result, color, y in methods:
        # Method name
        method_box = FancyBboxPatch((0.5, y - 0.35), 1.5, 0.6, boxstyle="round,pad=0.05",
                                    edgecolor=color, facecolor='white', linewidth=2)
        ax.add_patch(method_box)
        ax.text(1.25, y, method, ha='center', va='center',
                fontsize=10, family='monospace', fontweight='bold', color=color)

        # Description
        ax.text(2.3, y, desc, ha='left', va='center',
                fontsize=9, color='black')

        # Example code
        example_box = FancyBboxPatch((4.5, y - 0.35), 3, 0.6, boxstyle="round,pad=0.05",
                                     edgecolor=COLOR_SECONDARY, facecolor='#F5F5F5', linewidth=1.5)
        ax.add_patch(example_box)
        ax.text(6.0, y, example, ha='center', va='center',
                fontsize=8, family='monospace', color='black')

        # Result
        result_box = FancyBboxPatch((7.8, y - 0.35), 2, 0.6, boxstyle="round,pad=0.05",
                                    edgecolor=color, facecolor='white', linewidth=1.5)
        ax.add_patch(result_box)
        ax.text(8.8, y, result, ha='center', va='center',
                fontsize=8, family='monospace', color=color, fontweight='bold')

    # Headers
    ax.text(1.25, 9.0, 'Method', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
    ax.text(3.3, 9.0, 'Description', ha='left', va='center',
            fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
    ax.text(6.0, 9.0, 'Example', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
    ax.text(8.8, 9.0, 'Result', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_SECONDARY)

    plt.tight_layout()
    plt.savefig(base_dir / 'L02_Data_Structures' / '05_list_methods' / 'chart.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    print('  Chart 5/8: List methods')

def l02_chart6():
    """Portfolio as dictionary"""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Portfolio Representation: Dictionary', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    # Portfolio data
    portfolio = {
        'AAPL': {'shares': 100, 'buy_price': 145.00, 'current_price': 150.50},
        'MSFT': {'shares': 50, 'buy_price': 320.00, 'current_price': 340.00},
        'GOOGL': {'shares': 75, 'buy_price': 120.00, 'current_price': 125.75}
    }

    # Visual table
    ax.text(5, 8.5, 'portfolio = {', ha='center', va='top',
            fontsize=11, family='monospace', fontweight='bold', color='black')

    y_pos = 7.8
    for ticker, data in portfolio.items():
        # Stock ticker
        ticker_box = FancyBboxPatch((1.0, y_pos - 0.3), 1.2, 0.6, boxstyle="round,pad=0.05",
                                    edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
        ax.add_patch(ticker_box)
        ax.text(1.6, y_pos, f'"{ticker}"', ha='center', va='center',
                fontsize=9, family='monospace', fontweight='bold', color=COLOR_SECONDARY)

        # Details
        shares = data['shares']
        buy_price = data['buy_price']
        current_price = data['current_price']
        gain_loss = (current_price - buy_price) * shares
        gain_pct = ((current_price - buy_price) / buy_price) * 100

        details = f'shares: {shares}, buy: ${buy_price}, current: ${current_price}'
        details_box = FancyBboxPatch((2.5, y_pos - 0.3), 6.5, 0.6, boxstyle="round,pad=0.05",
                                     edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
        ax.add_patch(details_box)
        ax.text(5.75, y_pos, details, ha='center', va='center',
                fontsize=8, family='monospace', color='black')

        y_pos -= 0.9

    ax.text(5, y_pos + 0.3, '}', ha='center', va='top',
            fontsize=11, family='monospace', fontweight='bold', color='black')

    # Calculations
    calc_box = FancyBboxPatch((0.5, 1.0), 9, 3.8, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(calc_box)
    ax.text(5, 4.6, 'Portfolio Calculations', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_GREEN)

    calculations = [
        '# Calculate total value',
        'total_value = 0',
        'for ticker in portfolio:',
        '    shares = portfolio[ticker]["shares"]',
        '    price = portfolio[ticker]["current_price"]',
        '    total_value += shares * price',
        '',
        'print(f"Total portfolio value: ${total_value:,.2f}")',
        '# Output: Total portfolio value: $31,481.25',
    ]

    y_pos = 4.2
    for line in calculations:
        if line.startswith('#') or 'Output' in line:
            color = '#808080'
            weight = 'normal'
        else:
            color = 'black'
            weight = 'normal'

        ax.text(1.0, y_pos, line, ha='left', va='top',
                fontsize=8, family='monospace', color=color, fontweight=weight)
        y_pos -= 0.35

    plt.tight_layout()
    plt.savefig(base_dir / 'L02_Data_Structures' / '06_portfolio_dict' / 'chart.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    print('  Chart 6/8: Portfolio dictionary')

def l02_chart7():
    """List comprehension"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'List Comprehension: Concise List Creation', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    # Traditional approach
    trad_box = FancyBboxPatch((0.5, 6.5), 4.3, 2.5, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
    ax.add_patch(trad_box)
    ax.text(2.65, 8.8, 'Traditional Loop', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_ORANGE)

    trad_code = [
        'prices = [150, 165, 148, 172]',
        'doubled = []',
        'for price in prices:',
        '    doubled.append(price * 2)',
        '',
        '# Result: [300, 330, 296, 344]',
    ]

    y_pos = 8.3
    for line in trad_code:
        if line.startswith('#'):
            color = '#808080'
        else:
            color = 'black'
        ax.text(0.8, y_pos, line, ha='left', va='top',
                fontsize=9, family='monospace', color=color)
        y_pos -= 0.35

    # Arrow
    ax.annotate('', xy=(5.0, 7.7), xytext=(4.9, 7.7),
                arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=3))
    ax.text(4.95, 8.2, 'More Concise', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=COLOR_ACCENT)

    # List comprehension
    comp_box = FancyBboxPatch((5.2, 6.5), 4.3, 2.5, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(comp_box)
    ax.text(7.35, 8.8, 'List Comprehension', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_GREEN)

    comp_code = [
        'prices = [150, 165, 148, 172]',
        'doubled = [price * 2',
        '           for price in prices]',
        '',
        '',
        '# Result: [300, 330, 296, 344]',
    ]

    y_pos = 8.3
    for line in comp_code:
        if line.startswith('#'):
            color = '#808080'
        else:
            color = 'black'
        ax.text(5.5, y_pos, line, ha='left', va='top',
                fontsize=9, family='monospace', color=color, fontweight='bold')
        y_pos -= 0.35

    # More examples
    examples_box = FancyBboxPatch((0.5, 1.0), 9, 4.8, boxstyle="round,pad=0.1",
                                  edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
    ax.add_patch(examples_box)
    ax.text(5, 5.6, 'List Comprehension Examples', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

    examples = [
        ('Basic transformation', '[x * 2 for x in prices]', 'Double all prices'),
        ('With condition (filter)', '[x for x in prices if x > 150]', 'Only prices > 150'),
        ('String manipulation', '[t.lower() for t in tickers]', 'Lowercase all tickers'),
        ('Math operations', '[x**2 for x in [1,2,3,4]]', 'Squares: [1,4,9,16]'),
        ('With if-else', '[x if x > 150 else 0 for x in prices]', 'Set low prices to 0'),
    ]

    y_pos = 5.0
    for title, code, desc in examples:
        ax.text(1.0, y_pos, f'{title}:', ha='left', va='top',
                fontsize=9, fontweight='bold', color=COLOR_PRIMARY)
        ax.text(1.5, y_pos - 0.35, code, ha='left', va='top',
                fontsize=9, family='monospace', color='black')
        ax.text(1.5, y_pos - 0.65, f'# {desc}', ha='left', va='top',
                fontsize=8, family='monospace', color='#808080')
        y_pos -= 0.95

    plt.tight_layout()
    plt.savefig(base_dir / 'L02_Data_Structures' / '07_list_comprehension' / 'chart.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    print('  Chart 7/8: List comprehension')

def l02_chart8():
    """Data structure selection guide"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Choosing the Right Data Structure', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    # Decision tree
    ax.text(5, 8.8, 'Need to store data?', ha='center', va='top',
            fontsize=12, fontweight='bold', color='black')

    # Branch 1: Ordered sequence
    branch1_box = FancyBboxPatch((0.3, 6.5), 3.5, 1.8, boxstyle="round,pad=0.1",
                                 edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(branch1_box)
    ax.text(2.05, 8.0, 'Ordered sequence?', ha='center', va='top',
            fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
    ax.text(0.6, 7.5, 'Use LIST', ha='left', va='top',
            fontsize=10, fontweight='bold', color=COLOR_GREEN)
    ax.text(0.6, 7.1, 'Examples:', ha='left', va='top',
            fontsize=9, fontweight='bold', color='black')
    ax.text(0.8, 6.7, '- Stock prices over time', ha='left', va='top',
            fontsize=8, color='black')
    ax.text(0.8, 6.4, '- List of tickers', ha='left', va='top',
            fontsize=8, color='black')

    # Branch 2: Key-value mapping
    branch2_box = FancyBboxPatch((6.2, 6.5), 3.5, 1.8, boxstyle="round,pad=0.1",
                                 edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
    ax.add_patch(branch2_box)
    ax.text(7.95, 8.0, 'Key-value mapping?', ha='center', va='top',
            fontsize=11, fontweight='bold', color=COLOR_ACCENT)
    ax.text(6.5, 7.5, 'Use DICTIONARY', ha='left', va='top',
            fontsize=10, fontweight='bold', color=COLOR_GREEN)
    ax.text(6.5, 7.1, 'Examples:', ha='left', va='top',
            fontsize=9, fontweight='bold', color='black')
    ax.text(6.7, 6.7, '- Stock ticker → price', ha='left', va='top',
            fontsize=8, color='black')
    ax.text(6.7, 6.4, '- Portfolio holdings', ha='left', va='top',
            fontsize=8, color='black')

    # Arrows
    ax.annotate('', xy=(2, 6.5), xytext=(4.5, 8.3),
                arrowprops=dict(arrowstyle='->', color=COLOR_PRIMARY, lw=2))
    ax.annotate('', xy=(8, 6.5), xytext=(5.5, 8.3),
                arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))

    # Comparison table
    table_box = FancyBboxPatch((0.5, 0.5), 9, 5.5, boxstyle="round,pad=0.1",
                               edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
    ax.add_patch(table_box)
    ax.text(5, 5.8, 'Feature Comparison', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

    # Table headers
    ax.text(2.5, 5.2, 'List', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
    ax.text(5.0, 5.2, 'Feature', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
    ax.text(7.5, 5.2, 'Dictionary', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_ACCENT)

    # Separator
    ax.plot([0.7, 9.3], [5.0, 5.0], color=COLOR_SECONDARY, lw=1.5)

    # Comparison rows
    comparisons = [
        ('Ordered', 'Order', 'Unordered', 4.6),
        ('prices[0]', 'Access', 'portfolio["AAPL"]', 4.0),
        ('O(n) search', 'Speed', 'O(1) lookup', 3.4),
        ('Duplicates OK', 'Duplicates', 'Unique keys', 2.8),
        ('Integer indices', 'Keys', 'Any immutable', 2.2),
        ('append(), sort()', 'Methods', 'get(), keys()', 1.6),
        ('[1,2,3]', 'Syntax', '{"a": 1, "b": 2}', 1.0),
    ]

    for list_val, feature, dict_val, y in comparisons:
        ax.text(2.5, y, list_val, ha='center', va='center',
                fontsize=9, family='monospace', color=COLOR_PRIMARY, fontweight='bold')
        ax.text(5.0, y, feature, ha='center', va='center',
                fontsize=9, color='black')
        ax.text(7.5, y, dict_val, ha='center', va='center',
                fontsize=9, family='monospace', color=COLOR_ACCENT, fontweight='bold')

    plt.tight_layout()
    plt.savefig(base_dir / 'L02_Data_Structures' / '08_structure_selection' / 'chart.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    print('  Chart 8/8: Structure selection guide')

# Execute L02
l02_chart1()
l02_chart2()
l02_chart3()
l02_chart4()
l02_chart5()
l02_chart6()
l02_chart7()
l02_chart8()

print('\nL02 COMPLETE: 8/8 charts generated')
print('='*60)
