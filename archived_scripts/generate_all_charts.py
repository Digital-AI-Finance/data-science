"""
Autonomous chart generation script for DataScience_3 course
Generates all 48 charts for lessons L01-L06
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Educational color scheme
COLOR_PRIMARY = '#9B7EBD'    # mllavender
COLOR_SECONDARY = '#6B5B95'  # mlpurple
COLOR_ACCENT = '#4A90E2'     # mlblue
COLOR_LIGHT = '#ADADE0'      # mllavender2
COLOR_GREEN = '#44A05B'
COLOR_ORANGE = '#FF7F0E'
COLOR_RED = '#D62728'

plt.style.use('seaborn-v0_8-whitegrid')

# Base directory
base_dir = Path('D:/Joerg/Research/slides/DataScience_3')

# ========== L01: PYTHON SETUP ==========

def generate_l01_chart1():
    """Jupyter interface mockup"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Jupyter Notebook Interface', ha='center', va='top',
            fontsize=20, fontweight='bold', color=COLOR_SECONDARY)

    # Menu bar
    menu_bar = FancyBboxPatch((0.5, 8.5), 9, 0.6, boxstyle="round,pad=0.05",
                              edgecolor=COLOR_SECONDARY, facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(menu_bar)
    ax.text(1, 8.8, 'File  Edit  View  Insert  Cell  Kernel  Help',
            va='center', fontsize=11, color=COLOR_SECONDARY, fontweight='bold')

    # Code cell
    code_cell = FancyBboxPatch((0.5, 5.5), 9, 2.5, boxstyle="round,pad=0.05",
                               edgecolor=COLOR_PRIMARY, facecolor='#F5F5F5', linewidth=2)
    ax.add_patch(code_cell)
    ax.text(0.7, 7.8, 'In [1]:', va='top', fontsize=11, color=COLOR_SECONDARY,
            fontweight='bold', family='monospace')
    ax.text(1.5, 7.8, '# Calculate stock price change', va='top', fontsize=10,
            color='#808080', family='monospace')
    ax.text(1.5, 7.4, 'initial_price = 150.00', va='top', fontsize=10,
            color='black', family='monospace')
    ax.text(1.5, 7.0, 'final_price = 165.50', va='top', fontsize=10,
            color='black', family='monospace')
    ax.text(1.5, 6.6, 'change = final_price - initial_price', va='top', fontsize=10,
            color='black', family='monospace')
    ax.text(1.5, 6.2, 'print(f"Change: ${change:.2f}")', va='top', fontsize=10,
            color='black', family='monospace')

    # Output cell
    output_cell = FancyBboxPatch((0.5, 4.0), 9, 1.2, boxstyle="round,pad=0.05",
                                 edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
    ax.add_patch(output_cell)
    ax.text(0.7, 5.0, 'Out[1]:', va='top', fontsize=11, color=COLOR_ACCENT,
            fontweight='bold', family='monospace')
    ax.text(1.5, 5.0, 'Change: $15.50', va='top', fontsize=10,
            color='black', family='monospace')

    # Annotations
    ax.annotate('Input Code', xy=(0.5, 6.5), xytext=(0.2, 3.0),
                fontsize=10, color=COLOR_PRIMARY, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_PRIMARY, lw=2))
    ax.annotate('Output Result', xy=(0.5, 4.5), xytext=(0.2, 2.0),
                fontsize=10, color=COLOR_ACCENT, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))

    plt.tight_layout()
    output_path = base_dir / 'L01_Python_Setup' / '01_jupyter_interface' / 'chart.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('L01 Chart 1/8: Jupyter interface')

def generate_l01_chart2():
    """Data types hierarchy"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Python Data Types Hierarchy', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    # Root
    root = FancyBboxPatch((3.5, 7.5), 3, 0.8, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_SECONDARY, facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(root)
    ax.text(5, 7.9, 'Data Types', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLOR_SECONDARY)

    # Numeric types
    numeric = FancyBboxPatch((0.5, 5.5), 2, 0.8, boxstyle="round,pad=0.1",
                             edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
    ax.add_patch(numeric)
    ax.text(1.5, 5.9, 'Numeric', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

    # Text
    text_box = FancyBboxPatch((3, 5.5), 2, 0.8, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
    ax.add_patch(text_box)
    ax.text(4, 5.9, 'Text', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

    # Boolean
    bool_box = FancyBboxPatch((5.5, 5.5), 2, 0.8, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
    ax.add_patch(bool_box)
    ax.text(6.5, 5.9, 'Boolean', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

    # Sequence
    seq_box = FancyBboxPatch((8, 5.5), 2, 0.8, boxstyle="round,pad=0.1",
                             edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
    ax.add_patch(seq_box)
    ax.text(9, 5.9, 'Sequence', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

    # Subtypes
    subtypes = [
        (0.2, 3.5, 'int\n150'),
        (1.3, 3.5, 'float\n150.50'),
        (2.4, 3.5, 'complex\n1+2j'),
        (3, 3.5, 'str\n"AAPL"'),
        (5.5, 3.5, 'bool\nTrue/False'),
        (7.5, 3.5, 'list\n[1,2,3]'),
        (8.6, 3.5, 'tuple\n(1,2)'),
    ]

    for x, y, text in subtypes:
        subtype_box = FancyBboxPatch((x, y), 1, 0.9, boxstyle="round,pad=0.05",
                                     edgecolor=COLOR_ACCENT, facecolor='#F0F0F0', linewidth=1.5)
        ax.add_patch(subtype_box)
        ax.text(x + 0.5, y + 0.45, text, ha='center', va='center',
                fontsize=9, family='monospace', color='black')

    # Arrows
    arrow_props = dict(arrowstyle='->', color=COLOR_SECONDARY, lw=1.5)
    ax.annotate('', xy=(1.5, 5.5), xytext=(5, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(4, 5.5), xytext=(5, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 5.5), xytext=(5, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(9, 5.5), xytext=(5, 7.5), arrowprops=arrow_props)

    # Finance note
    note_box = FancyBboxPatch((2, 1.0), 6, 1.2, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(note_box)
    ax.text(5, 1.9, 'Finance Application', ha='center', va='top',
            fontsize=11, fontweight='bold', color=COLOR_GREEN)
    ax.text(5, 1.5, 'price = 150.50  # float for stock price', ha='center', va='top',
            fontsize=9, family='monospace', color='black')
    ax.text(5, 1.2, 'ticker = "AAPL"  # string for stock symbol', ha='center', va='top',
            fontsize=9, family='monospace', color='black')

    plt.tight_layout()
    output_path = base_dir / 'L01_Python_Setup' / '02_data_types_hierarchy' / 'chart.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('L01 Chart 2/8: Data types hierarchy')

def generate_l01_chart3():
    """Variable assignment flowchart"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Variable Assignment Process', ha='center', va='top',
            fontsize=16, fontweight='bold', color=COLOR_SECONDARY)

    # Step boxes
    steps = [
        (8.5, 'Write variable name', COLOR_PRIMARY),
        (7.3, 'Use = operator', COLOR_PRIMARY),
        (6.1, 'Provide value', COLOR_PRIMARY),
        (4.9, 'Python stores in memory', COLOR_ACCENT),
        (3.7, 'Variable ready to use', COLOR_GREEN),
    ]

    for y, text, color in steps:
        box = FancyBboxPatch((2, y - 0.4), 6, 0.8, boxstyle="round,pad=0.1",
                             edgecolor=color, facecolor='white', linewidth=2)
        ax.add_patch(box)
        ax.text(5, y, text, ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)

        # Arrow to next step
        if y > 4:
            ax.annotate('', xy=(5, y - 0.5), xytext=(5, y - 1.1),
                       arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=2))

    # Example
    example_box = FancyBboxPatch((1.5, 1.5), 7, 1.8, boxstyle="round,pad=0.1",
                                 edgecolor=COLOR_SECONDARY, facecolor='#F5F5F5', linewidth=2)
    ax.add_patch(example_box)
    ax.text(5, 3.1, 'Example: Stock Price', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_SECONDARY)
    ax.text(2, 2.7, 'stock_price = 150.50', ha='left', va='top',
            fontsize=11, family='monospace', color='black')
    ax.text(2, 2.3, 'print(stock_price)  # Output: 150.50', ha='left', va='top',
            fontsize=11, family='monospace', color='#808080')
    ax.text(2, 1.9, 'new_price = stock_price * 1.05', ha='left', va='top',
            fontsize=11, family='monospace', color='black')

    plt.tight_layout()
    output_path = base_dir / 'L01_Python_Setup' / '03_variable_assignment' / 'chart.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('L01 Chart 3/8: Variable assignment flowchart')

def generate_l01_chart4():
    """Integer vs float comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Integer vs Float: Key Differences', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    # Left column: Integer
    int_box = FancyBboxPatch((0.5, 6.5), 4, 2.5, boxstyle="round,pad=0.1",
                             edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(int_box)
    ax.text(2.5, 8.7, 'Integer (int)', ha='center', va='top',
            fontsize=14, fontweight='bold', color=COLOR_SECONDARY)
    ax.text(0.8, 8.2, '- Whole numbers only', ha='left', va='top',
            fontsize=10, color='black')
    ax.text(0.8, 7.8, '- No decimal point', ha='left', va='top',
            fontsize=10, color='black')
    ax.text(0.8, 7.4, '- Exact counting', ha='left', va='top',
            fontsize=10, color='black')
    ax.text(0.8, 7.0, '- Examples:', ha='left', va='top',
            fontsize=10, fontweight='bold', color='black')

    int_examples = FancyBboxPatch((1, 6.2), 2.8, 0.6, boxstyle="round,pad=0.05",
                                  edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
    ax.add_patch(int_examples)
    ax.text(2.4, 6.5, 'shares = 100', ha='center', va='center',
            fontsize=9, family='monospace', color='black')

    # Right column: Float
    float_box = FancyBboxPatch((5.5, 6.5), 4, 2.5, boxstyle="round,pad=0.1",
                               edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(float_box)
    ax.text(7.5, 8.7, 'Float (float)', ha='center', va='top',
            fontsize=14, fontweight='bold', color=COLOR_SECONDARY)
    ax.text(5.8, 8.2, '- Decimal numbers', ha='left', va='top',
            fontsize=10, color='black')
    ax.text(5.8, 7.8, '- Has decimal point', ha='left', va='top',
            fontsize=10, color='black')
    ax.text(5.8, 7.4, '- Precise measurements', ha='left', va='top',
            fontsize=10, color='black')
    ax.text(5.8, 7.0, '- Examples:', ha='left', va='top',
            fontsize=10, fontweight='bold', color='black')

    float_examples = FancyBboxPatch((6, 6.2), 3, 0.6, boxstyle="round,pad=0.05",
                                    edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
    ax.add_patch(float_examples)
    ax.text(7.5, 6.5, 'price = 150.50', ha='center', va='center',
            fontsize=9, family='monospace', color='black')

    # Comparison table
    table_box = FancyBboxPatch((1, 2.5), 8, 3.5, boxstyle="round,pad=0.1",
                               edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
    ax.add_patch(table_box)
    ax.text(5, 5.8, 'Finance Use Cases', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

    # Table headers
    ax.text(2.5, 5.3, 'Integer', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
    ax.text(7.5, 5.3, 'Float', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_PRIMARY)

    # Draw separator line
    ax.plot([1.5, 8.5], [5.1, 5.1], color=COLOR_SECONDARY, lw=1.5)
    ax.plot([5, 5], [5.1, 2.7], color=COLOR_SECONDARY, lw=1.5)

    # Table content
    int_uses = ['Number of shares', 'Days in period', 'Number of trades']
    float_uses = ['Stock price', 'Portfolio value', 'Return percentage']

    y_pos = 4.7
    for int_use, float_use in zip(int_uses, float_uses):
        ax.text(2.5, y_pos, int_use, ha='center', va='center',
                fontsize=10, color='black')
        ax.text(7.5, y_pos, float_use, ha='center', va='center',
                fontsize=10, color='black')
        y_pos -= 0.6

    # Code example
    code_box = FancyBboxPatch((1.5, 1.0), 7, 1.2, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_GREEN, facecolor='#F5F5F5', linewidth=2)
    ax.add_patch(code_box)
    ax.text(5, 2.0, 'value = shares * price  # int * float = float', ha='center', va='top',
            fontsize=10, family='monospace', color='black')
    ax.text(5, 1.6, 'value = 100 * 150.50  # = 15050.0', ha='center', va='top',
            fontsize=10, family='monospace', color='black')
    ax.text(5, 1.2, 'type(value)  # <class \'float\'>', ha='center', va='top',
            fontsize=10, family='monospace', color='#808080')

    plt.tight_layout()
    output_path = base_dir / 'L01_Python_Setup' / '04_int_vs_float' / 'chart.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('L01 Chart 4/8: Integer vs float comparison')

def generate_l01_chart5():
    """String operations visual"""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'String Operations in Python', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    operations = [
        ('Concatenation', 'ticker1 + ticker2', '"AAPL" + "MSFT"', '"AAPLMSFT"', 8.0),
        ('Repetition', 'ticker * 3', '"XYZ" * 3', '"XYZXYZXYZ"', 6.8),
        ('Upper/Lower', 'ticker.upper()', '"aapl".upper()', '"AAPL"', 5.6),
        ('Slicing', 'ticker[0:2]', '"APPLE"[0:2]', '"AP"', 4.4),
        ('Length', 'len(ticker)', 'len("AAPL")', '4', 3.2),
        ('Format', 'f-string', 'f"Price: ${price}"', '"Price: $150.50"', 2.0),
    ]

    for op_name, syntax, example, result, y in operations:
        # Operation box
        op_box = FancyBboxPatch((0.5, y - 0.4), 2, 0.7, boxstyle="round,pad=0.05",
                                edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
        ax.add_patch(op_box)
        ax.text(1.5, y, op_name, ha='center', va='center',
                fontsize=10, fontweight='bold', color=COLOR_SECONDARY)

        # Syntax box
        syntax_box = FancyBboxPatch((2.8, y - 0.4), 2.2, 0.7, boxstyle="round,pad=0.05",
                                    edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
        ax.add_patch(syntax_box)
        ax.text(3.9, y, syntax, ha='center', va='center',
                fontsize=9, family='monospace', color='black')

        # Example box
        example_box = FancyBboxPatch((5.3, y - 0.4), 2, 0.7, boxstyle="round,pad=0.05",
                                     edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
        ax.add_patch(example_box)
        ax.text(6.3, y, example, ha='center', va='center',
                fontsize=8, family='monospace', color='#808080')

        # Result box
        result_box = FancyBboxPatch((7.5, y - 0.4), 2, 0.7, boxstyle="round,pad=0.05",
                                    edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=1.5)
        ax.add_patch(result_box)
        ax.text(8.5, y, result, ha='center', va='center',
                fontsize=9, family='monospace', color=COLOR_GREEN)

    # Headers
    ax.text(1.5, 9.0, 'Operation', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
    ax.text(3.9, 9.0, 'Syntax', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
    ax.text(6.3, 9.0, 'Example', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
    ax.text(8.5, 9.0, 'Result', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_SECONDARY)

    plt.tight_layout()
    output_path = base_dir / 'L01_Python_Setup' / '05_string_operations' / 'chart.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('L01 Chart 5/8: String operations visual')

def generate_l01_chart6():
    """Boolean logic truth table"""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Boolean Logic & Truth Tables', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    # AND operator
    and_box = FancyBboxPatch((0.5, 6.5), 4, 2.5, boxstyle="round,pad=0.1",
                             edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
    ax.add_patch(and_box)
    ax.text(2.5, 8.7, 'AND Operator', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

    # AND truth table
    and_table_data = [
        ('A', 'B', 'A and B'),
        ('True', 'True', 'True'),
        ('True', 'False', 'False'),
        ('False', 'True', 'False'),
        ('False', 'False', 'False'),
    ]

    y_pos = 8.2
    for i, (a, b, result) in enumerate(and_table_data):
        if i == 0:
            color = COLOR_SECONDARY
            weight = 'bold'
        else:
            color = 'black'
            weight = 'normal'

        ax.text(1.0, y_pos, a, ha='center', va='center',
                fontsize=9, fontweight=weight, color=color)
        ax.text(2.0, y_pos, b, ha='center', va='center',
                fontsize=9, fontweight=weight, color=color)
        ax.text(3.5, y_pos, result, ha='center', va='center',
                fontsize=9, fontweight=weight,
                color=COLOR_GREEN if result == 'True' and i > 0 else color)
        y_pos -= 0.4

    # OR operator
    or_box = FancyBboxPatch((5.5, 6.5), 4, 2.5, boxstyle="round,pad=0.1",
                            edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
    ax.add_patch(or_box)
    ax.text(7.5, 8.7, 'OR Operator', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

    # OR truth table
    or_table_data = [
        ('A', 'B', 'A or B'),
        ('True', 'True', 'True'),
        ('True', 'False', 'True'),
        ('False', 'True', 'True'),
        ('False', 'False', 'False'),
    ]

    y_pos = 8.2
    for i, (a, b, result) in enumerate(or_table_data):
        if i == 0:
            color = COLOR_SECONDARY
            weight = 'bold'
        else:
            color = 'black'
            weight = 'normal'

        ax.text(6.0, y_pos, a, ha='center', va='center',
                fontsize=9, fontweight=weight, color=color)
        ax.text(7.0, y_pos, b, ha='center', va='center',
                fontsize=9, fontweight=weight, color=color)
        ax.text(8.5, y_pos, result, ha='center', va='center',
                fontsize=9, fontweight=weight,
                color=COLOR_GREEN if result == 'True' and i > 0 else color)
        y_pos -= 0.4

    # NOT operator
    not_box = FancyBboxPatch((3, 3.5), 4, 2.0, boxstyle="round,pad=0.1",
                             edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
    ax.add_patch(not_box)
    ax.text(5, 5.2, 'NOT Operator', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

    # NOT truth table
    not_table_data = [
        ('A', 'not A'),
        ('True', 'False'),
        ('False', 'True'),
    ]

    y_pos = 4.7
    for i, (a, result) in enumerate(not_table_data):
        if i == 0:
            color = COLOR_SECONDARY
            weight = 'bold'
        else:
            color = 'black'
            weight = 'normal'

        ax.text(4.0, y_pos, a, ha='center', va='center',
                fontsize=9, fontweight=weight, color=color)
        ax.text(6.0, y_pos, result, ha='center', va='center',
                fontsize=9, fontweight=weight, color=color)
        y_pos -= 0.4

    # Finance example
    finance_box = FancyBboxPatch((1, 1.0), 8, 2.0, boxstyle="round,pad=0.1",
                                 edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(finance_box)
    ax.text(5, 2.8, 'Finance Example: Buy Signal', ha='center', va='top',
            fontsize=11, fontweight='bold', color=COLOR_GREEN)
    ax.text(1.5, 2.3, 'price = 145.00', ha='left', va='top',
            fontsize=9, family='monospace', color='black')
    ax.text(1.5, 1.9, 'volume = 1000000', ha='left', va='top',
            fontsize=9, family='monospace', color='black')
    ax.text(1.5, 1.5, 'buy = (price < 150) and (volume > 500000)  # True', ha='left', va='top',
            fontsize=9, family='monospace', color='black')
    ax.text(1.5, 1.1, 'print(f"Buy signal: {buy}")  # Buy signal: True', ha='left', va='top',
            fontsize=9, family='monospace', color='#808080')

    plt.tight_layout()
    output_path = base_dir / 'L01_Python_Setup' / '06_boolean_logic' / 'chart.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('L01 Chart 6/8: Boolean logic truth table')

def generate_l01_chart7():
    """Type conversion diagram"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Type Conversion (Casting)', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    # Type boxes in circle
    types = [
        ('int', 5, 7, 0),
        ('float', 7.5, 5, 1),
        ('str', 5, 3, 2),
        ('bool', 2.5, 5, 3),
    ]

    type_colors = [COLOR_PRIMARY, COLOR_ACCENT, COLOR_ORANGE, COLOR_GREEN]

    for type_name, x, y, idx in types:
        type_box = FancyBboxPatch((x - 0.6, y - 0.4), 1.2, 0.8,
                                  boxstyle="round,pad=0.1",
                                  edgecolor=type_colors[idx],
                                  facecolor='white', linewidth=3)
        ax.add_patch(type_box)
        ax.text(x, y, type_name, ha='center', va='center',
                fontsize=14, fontweight='bold', color=type_colors[idx])

    # Conversion arrows and labels
    conversions = [
        # (from_idx, to_idx, label, curve)
        (0, 1, 'float()', 0.3),  # int to float
        (1, 0, 'int()', -0.3),   # float to int
        (0, 2, 'str()', 0.2),    # int to str
        (2, 0, 'int()', -0.2),   # str to int
        (1, 2, 'str()', 0.2),    # float to str
        (2, 1, 'float()', -0.2), # str to float
        (3, 0, 'int()', 0.2),    # bool to int
        (0, 3, 'bool()', -0.2),  # int to bool
    ]

    for from_idx, to_idx, label, _ in conversions:
        from_type = types[from_idx]
        to_type = types[to_idx]

        # Draw curved arrow
        ax.annotate('', xy=(to_type[1], to_type[2]),
                   xytext=(from_type[1], from_type[2]),
                   arrowprops=dict(arrowstyle='->', color=type_colors[from_idx],
                                 lw=1.5, alpha=0.6,
                                 connectionstyle='arc3,rad=0.3'))

    # Examples table
    examples_box = FancyBboxPatch((0.5, 0.2), 9, 2.0, boxstyle="round,pad=0.1",
                                  edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
    ax.add_patch(examples_box)
    ax.text(5, 2.0, 'Conversion Examples', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

    examples = [
        'int("150")      # "150" -> 150',
        'float("150.50") # "150.50" -> 150.5',
        'str(150)        # 150 -> "150"',
        'int(150.99)     # 150.99 -> 150 (truncates!)',
        'bool(0)         # 0 -> False',
        'bool(150)       # 150 -> True',
    ]

    y_pos = 1.6
    x_left = 1.0
    x_right = 5.5

    for i, example in enumerate(examples):
        x = x_left if i < 3 else x_right
        y_offset = i if i < 3 else i - 3
        ax.text(x, y_pos - y_offset * 0.35, example, ha='left', va='top',
                fontsize=9, family='monospace', color='black')

    plt.tight_layout()
    output_path = base_dir / 'L01_Python_Setup' / '07_type_conversion' / 'chart.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('L01 Chart 7/8: Type conversion diagram')

def generate_l01_chart8():
    """Python vs Excel comparison"""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Python vs Excel for Finance', ha='center', va='top',
            fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

    # Comparison table
    categories = [
        ('Feature', 'Excel', 'Python', 8.5),
        ('Data Size', 'Limited (1M rows)', 'Unlimited', 7.8),
        ('Automation', 'Manual/Macros', 'Full Scripts', 7.1),
        ('Reproducibility', 'Low', 'High', 6.4),
        ('Version Control', 'Difficult', 'Git Integration', 5.7),
        ('Visualization', 'Built-in Charts', 'Custom Libraries', 5.0),
        ('Speed', 'Slow (large data)', 'Fast', 4.3),
        ('Learning Curve', 'Easy', 'Moderate', 3.6),
    ]

    # Draw table header
    header_box = FancyBboxPatch((1, 8.2), 8, 0.6, boxstyle="round,pad=0.05",
                                edgecolor=COLOR_SECONDARY, facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(header_box)

    for category, excel, python, y in categories:
        if y == 8.5:  # Header
            ax.text(2, y, category, ha='center', va='center',
                   fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
            ax.text(5, y, excel, ha='center', va='center',
                   fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
            ax.text(8, y, python, ha='center', va='center',
                   fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
        else:
            # Category
            cat_box = FancyBboxPatch((1, y - 0.25), 2, 0.5, boxstyle="round,pad=0.05",
                                     edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=1.5)
            ax.add_patch(cat_box)
            ax.text(2, y, category, ha='center', va='center',
                   fontsize=9, fontweight='bold', color=COLOR_PRIMARY)

            # Excel
            excel_box = FancyBboxPatch((3.5, y - 0.25), 2.5, 0.5, boxstyle="round,pad=0.05",
                                       edgecolor=COLOR_ACCENT, facecolor='#F5F5F5', linewidth=1.5)
            ax.add_patch(excel_box)
            ax.text(4.75, y, excel, ha='center', va='center',
                   fontsize=9, color='black')

            # Python
            python_box = FancyBboxPatch((6.5, y - 0.25), 2.5, 0.5, boxstyle="round,pad=0.05",
                                        edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=1.5)
            ax.add_patch(python_box)
            ax.text(7.75, y, python, ha='center', va='center',
                   fontsize=9, color=COLOR_GREEN, fontweight='bold')

    # Best use cases
    excel_use = FancyBboxPatch((0.5, 0.8), 4.2, 2.2, boxstyle="round,pad=0.1",
                               edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
    ax.add_patch(excel_use)
    ax.text(2.6, 2.8, 'Best for Excel:', ha='center', va='top',
            fontsize=11, fontweight='bold', color=COLOR_ACCENT)
    excel_uses = [
        '- Quick calculations',
        '- Small datasets (<10K rows)',
        '- Visual exploration',
        '- Ad-hoc analysis',
    ]
    y_pos = 2.4
    for use in excel_uses:
        ax.text(0.8, y_pos, use, ha='left', va='top',
                fontsize=9, color='black')
        y_pos -= 0.35

    python_use = FancyBboxPatch((5.3, 0.8), 4.2, 2.2, boxstyle="round,pad=0.1",
                                edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(python_use)
    ax.text(7.4, 2.8, 'Best for Python:', ha='center', va='top',
            fontsize=11, fontweight='bold', color=COLOR_GREEN)
    python_uses = [
        '- Large datasets (>100K rows)',
        '- Automated workflows',
        '- Machine learning',
        '- Production systems',
    ]
    y_pos = 2.4
    for use in python_uses:
        ax.text(5.6, y_pos, use, ha='left', va='top',
                fontsize=9, color='black')
        y_pos -= 0.35

    plt.tight_layout()
    output_path = base_dir / 'L01_Python_Setup' / '08_python_vs_excel' / 'chart.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('L01 Chart 8/8: Python vs Excel comparison')

# ========== MAIN EXECUTION ==========

if __name__ == '__main__':
    print('=' * 60)
    print('AUTONOMOUS CHART GENERATION - DataScience_3 Course')
    print('=' * 60)
    print('\nGenerating charts for L01: Python Setup...\n')

    generate_l01_chart1()
    generate_l01_chart2()
    generate_l01_chart3()
    generate_l01_chart4()
    generate_l01_chart5()
    generate_l01_chart6()
    generate_l01_chart7()
    generate_l01_chart8()

    print('\n' + '=' * 60)
    print('L01 COMPLETE: 8/8 charts generated')
    print('=' * 60)
