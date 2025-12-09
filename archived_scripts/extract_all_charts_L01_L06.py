"""
Comprehensive script to extract ALL chart code from generation scripts
and create individual chart.py files with course colors for L01-L06.
"""
import re
from pathlib import Path
import subprocess

# Standard imports and rc Params for all charts
CHART_HEADER = '''"""Generated chart using course color palette"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Wedge
import numpy as np
import pandas as pd
import seaborn as sns
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

# Map old colors to new course colors
COLOR_PRIMARY = MLPURPLE
COLOR_SECONDARY = MLPURPLE
COLOR_ACCENT = MLBLUE
COLOR_LIGHT = MLLAVENDER
COLOR_GREEN = MLGREEN
COLOR_ORANGE = MLORANGE
COLOR_RED = MLRED

base_dir = Path(r'D:/Joerg/Research/slides/DataScience_3')
'''

def color_replace(code):
    """Replace color hex codes with variable names and old variables"""
    replacements = [
        ("'#9B7EBD'", "COLOR_PRIMARY"),
        ("'#6B5B95'", "COLOR_SECONDARY"),
        ("'#4A90E2'", "COLOR_ACCENT"),
        ("'#ADADE0'", "COLOR_LIGHT"),
        ("'#44A05B'", "COLOR_GREEN"),
        ("'#FF7F0E'", "COLOR_ORANGE"),
        ("'#D62728'", "COLOR_RED"),
        ("'#808080'", "'#808080'"),  # Keep gray
        ("'white'", "'white'"),
        ("'black'", "'black'"),
    ]

    for old, new in replacements:
        code = code.replace(old, new)

    return code

def process_L01_from_generate_all():
    """Extract L01 charts from generate_all_charts.py"""
    script_path = Path(r"D:\Joerg\Research\slides\DataScience_3\generate_all_charts.py")
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # L01 chart folder names
    chart_folders = [
        '01_jupyter_interface',
        '02_data_types_hierarchy',
        '03_variable_assignment',
        '04_int_vs_float',
        '05_string_operations',
        '06_boolean_logic',
        '07_type_conversion',
        '08_python_vs_excel'
    ]

    # Extract functions
    pattern = r'def generate_l01_chart\d+\(\):.*?(?=\ndef |# ========== |$)'
    functions = re.findall(pattern, content, re.DOTALL)

    print(f"Found {len(functions)} L01 functions")

    for i, func_code in enumerate(functions, 1):
        if i > len(chart_folders):
            break

        folder_name = chart_folders[i-1]
        chart_dir = Path(r"D:\Joerg\Research\slides\DataScience_3\L01_Python_Setup") / folder_name
        chart_py = chart_dir / 'chart.py'

        # Extract function body
        lines = func_code.split('\n')
        # Find start of actual code (after docstring)
        start_idx = 0
        for idx, line in enumerate(lines):
            if 'fig, ax' in line or 'fig =' in line or 'plt.' in line:
                start_idx = idx
                break

        body_lines = lines[start_idx:]

        # Remove one level of indentation
        deindented = []
        for line in body_lines:
            if line.startswith('    '):
                deindented.append(line[4:])
            else:
                deindented.append(line)

        body_code = '\n'.join(deindented)
        body_code = color_replace(body_code)

        # Write file
        full_code = CHART_HEADER + '\n' + body_code
        with open(chart_py, 'w', encoding='utf-8') as f:
            f.write(full_code)

        print(f"  L01 {i}/8: {folder_name}")

def process_L02():
    """Extract L02 charts from generate_L02_L06_charts.py"""
    script_path = Path(r"D:\Joerg\Research\slides\DataScience_3\generate_L02_L06_charts.py")
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chart_folders = [
        '01_list_indexing',
        '02_slicing_notation',
        '03_dictionary_structure',
        '04_nested_structures',
        '05_list_methods',
        '06_portfolio_dict',
        '07_list_comprehension',
        '08_structure_selection'
    ]

    pattern = r'def l02_chart\d+\(\):.*?(?=\ndef |print\(.*L02 COMPLETE|$)'
    functions = re.findall(pattern, content, re.DOTALL)

    print(f"Found {len(functions)} L02 functions")

    for i, func_code in enumerate(functions, 1):
        if i > len(chart_folders):
            break

        folder_name = chart_folders[i-1]
        chart_dir = Path(r"D:\Joerg\Research\slides\DataScience_3\L02_Data_Structures") / folder_name
        chart_py = chart_dir / 'chart.py'

        # Extract function body
        lines = func_code.split('\n')
        start_idx = 0
        for idx, line in enumerate(lines):
            if 'fig, ax' in line or 'fig =' in line:
                start_idx = idx
                break

        body_lines = lines[start_idx:]
        deindented = []
        for line in body_lines:
            if line.startswith('    '):
                deindented.append(line[4:])
            else:
                deindented.append(line)

        body_code = '\n'.join(deindented)
        body_code = color_replace(body_code)

        full_code = CHART_HEADER + '\n' + body_code
        with open(chart_py, 'w', encoding='utf-8') as f:
            f.write(full_code)

        print(f"  L02 {i}/8: {folder_name}")

def process_L03():
    """Extract L03 charts from generate_L03_charts.py"""
    script_path = Path(r"D:\Joerg\Research\slides\DataScience_3\generate_L03_charts.py")
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chart_folders = [
        '01_if_else_flowchart',
        '02_for_loop_visual',
        '03_while_loop',
        '04_nested_loops',
        '05_break_continue',
        '06_trading_rules',
        '07_loop_comparison',
        '08_control_patterns'
    ]

    # L03 script has direct code, not functions - need to split by chart
    # Look for chart boundaries by finding "# Chart X:" comments and plt.savefig
    parts = re.split(r'# Chart \d+:[^\n]*\n', content)

    print(f"Found {len(parts)-1} L03 chart sections")

    for i in range(1, min(len(parts), 9)):
        chart_code = parts[i]

        folder_name = chart_folders[i-1]
        chart_dir = Path(r"D:\Joerg\Research\slides\DataScience_3\L03_Control_Flow") / folder_name
        chart_py = chart_dir / 'chart.py'

        # Clean up the code - remove print statements at end
        chart_code = re.sub(r'\nprint\(.*?\)\s*$', '', chart_code, flags=re.MULTILINE)
        chart_code = color_replace(chart_code)

        full_code = CHART_HEADER + '\n' + chart_code
        with open(chart_py, 'w', encoding='utf-8') as f:
            f.write(full_code)

        print(f"  L03 {i}/8: {folder_name}")

def process_L04_L05_L06():
    """Extract L04-L06 charts from generate_L04_L06.py"""
    script_path = Path(r"D:\Joerg\Research\slides\DataScience_3\generate_L04_L06.py")
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # L04
    l04_folders = [
        '01_function_anatomy',
        '02_parameter_passing',
        '03_return_flowchart',
        '04_scope_diagram',
        '05_docstring_format',
        '06_call_stack',
        '07_pure_vs_impure',
        '08_finance_functions'
    ]

    # L05
    l05_folders = [
        '01_dataframe_structure',
        '02_series_vs_dataframe',
        '03_csv_loading',
        '04_head_tail',
        '05_info_breakdown',
        '06_describe_stats',
        '07_index_columns',
        '08_stock_example'
    ]

    # L06
    l06_folders = [
        '01_column_selection',
        '02_iloc_vs_loc',
        '03_boolean_mask',
        '04_conditional_filtering',
        '05_multiple_conditions',
        '06_chained_filtering',
        '07_selection_comparison',
        '08_stock_screening'
    ]

    # Extract function bodies
    l04_match = re.search(r'def generate_L04_charts\(\):.*?(?=\ndef generate_L05|$)', content, re.DOTALL)
    l05_match = re.search(r'def generate_L05_charts\(\):.*?(?=\ndef generate_L06|$)', content, re.DOTALL)
    l06_match = re.search(r'def generate_L06_charts\(\):.*?(?=\ndef |if __name__|$)', content, re.DOTALL)

    # Process L04
    if l04_match:
        l04_content = l04_match.group(0)
        parts = re.split(r'# Chart \d+:[^\n]*\n', l04_content)
        print(f"Found {len(parts)-1} L04 chart sections")

        for i in range(1, min(len(parts), 9)):
            chart_code = parts[i]
            folder_name = l04_folders[i-1]
            chart_dir = Path(r"D:\Joerg\Research\slides\DataScience_3\L04_Functions") / folder_name
            chart_py = chart_dir / 'chart.py'

            chart_code = re.sub(r'\nprint\(.*?\)\s*$', '', chart_code, flags=re.MULTILINE)
            chart_code = color_replace(chart_code)
            full_code = CHART_HEADER + '\n' + chart_code
            with open(chart_py, 'w', encoding='utf-8') as f:
                f.write(full_code)

            print(f"  L04 {i}/8: {folder_name}")

    # Process L05
    if l05_match:
        l05_content = l05_match.group(0)
        parts = re.split(r'# Chart \d+:[^\n]*\n', l05_content)
        print(f"Found {len(parts)-1} L05 chart sections")

        for i in range(1, min(len(parts), 9)):
            chart_code = parts[i]
            folder_name = l05_folders[i-1]
            chart_dir = Path(r"D:\Joerg\Research\slides\DataScience_3\L05_DataFrames_Introduction") / folder_name
            chart_py = chart_dir / 'chart.py'

            chart_code = re.sub(r'\nprint\(.*?\)\s*$', '', chart_code, flags=re.MULTILINE)
            chart_code = color_replace(chart_code)
            full_code = CHART_HEADER + '\n' + chart_code
            with open(chart_py, 'w', encoding='utf-8') as f:
                f.write(full_code)

            print(f"  L05 {i}/8: {folder_name}")

    # Process L06
    if l06_match:
        l06_content = l06_match.group(0)
        parts = re.split(r'# Chart \d+:[^\n]*\n', l06_content)
        print(f"Found {len(parts)-1} L06 chart sections")

        for i in range(1, min(len(parts), 9)):
            chart_code = parts[i]
            folder_name = l06_folders[i-1]
            chart_dir = Path(r"D:\Joerg\Research\slides\DataScience_3\L06_Selection_Filtering") / folder_name
            chart_py = chart_dir / 'chart.py'

            chart_code = re.sub(r'\nprint\(.*?\)\s*$', '', chart_code, flags=re.MULTILINE)
            chart_code = color_replace(chart_code)
            full_code = CHART_HEADER + '\n' + chart_code
            with open(chart_py, 'w', encoding='utf-8') as f:
                f.write(full_code)

            print(f"  L06 {i}/8: {folder_name}")

def main():
    print("="*60)
    print("EXTRACTING ALL CHARTS FOR L01-L06")
    print("="*60)

    print("\nProcessing L01...")
    process_L01_from_generate_all()

    print("\nProcessing L02...")
    process_L02()

    print("\nProcessing L03...")
    process_L03()

    print("\nProcessing L04-L06...")
    process_L04_L05_L06()

    print("\n" + "="*60)
    print("ALL CHART.PY FILES CREATED!")
    print("="*60)

if __name__ == "__main__":
    main()
