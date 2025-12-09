"""
Extract chart generation code from batch scripts and create individual chart.py files
with proper course colors and styling.
"""
import re
from pathlib import Path
import subprocess
import sys

# Course colors - MANDATORY
MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

# Standard imports and rcParams for all charts
CHART_HEADER = '''"""
Generated chart using course color palette
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Wedge
import numpy as np
import pandas as pd
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

# Old color mapping (for reference)
COLOR_PRIMARY = '#9B7EBD'  # -> MLPURPLE
COLOR_SECONDARY = '#6B5B95'  # -> MLPURPLE (darker)
COLOR_ACCENT = '#4A90E2'  # -> MLBLUE
COLOR_LIGHT = '#ADADE0'  # -> MLLAVENDER (same)
COLOR_GREEN = '#44A05B'  # -> MLGREEN
COLOR_ORANGE = '#FF7F0E'  # -> MLORANGE (same)
COLOR_RED = '#D62728'  # -> MLRED (same)

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
    """Replace old colors with course colors in code"""
    # Don't replace colors in the header definition area
    replacements = {
        "'#9B7EBD'": "COLOR_PRIMARY",
        "'#6B5B95'": "COLOR_SECONDARY",
        "'#4A90E2'": "COLOR_ACCENT",
        "'#ADADE0'": "COLOR_LIGHT",
        "'#44A05B'": "COLOR_GREEN",
        "'#FF7F0E'": "COLOR_ORANGE",
        "'#D62728'": "COLOR_RED",
        # Also handle without quotes
        "#9B7EBD": "COLOR_PRIMARY",
        "#6B5B95": "COLOR_SECONDARY",
        "#4A90E2": "COLOR_ACCENT",
        "#ADADE0": "COLOR_LIGHT",
        "#44A05B": "COLOR_GREEN",
    }

    result = code
    for old, new in replacements.items():
        if old in result:
            result = result.replace(old, new)

    return result

def extract_chart_functions(script_path):
    """Extract chart functions from generation script"""
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all function definitions
    pattern = r'def (l\d+_chart\d+)\(\):.*?(?=\ndef |$)'
    matches = re.findall(pattern, content, re.DOTALL)

    functions = {}
    for match in re.finditer(pattern, content, re.DOTALL):
        func_name = match.group(1)
        func_body = match.group(0)
        functions[func_name] = func_body

    return functions

def create_chart_py_file(lesson_folder, chart_folder, function_code):
    """Create individual chart.py file"""
    chart_dir = Path(lesson_folder) / chart_folder
    if not chart_dir.exists():
        print(f"  Warning: {chart_dir} does not exist, skipping")
        return False

    chart_py = chart_dir / 'chart.py'

    # Extract just the function body (remove def line and indentation)
    lines = function_code.split('\n')
    # Find where the actual code starts (after the docstring)
    start_idx = 0
    for i, line in enumerate(lines):
        if 'fig, ax' in line or 'plt.figure' in line:
            start_idx = i
            break

    # Get the function body without def line
    body_lines = lines[start_idx:]

    # Remove one level of indentation
    deindented = []
    for line in body_lines:
        if line.startswith('    '):
            deindented.append(line[4:])
        elif line.strip() == '':
            deindented.append('')
        else:
            deindented.append(line)

    # Color replacement
    body_code = '\n'.join(deindented)
    body_code = color_replace(body_code)

    # Create complete chart.py
    full_code = CHART_HEADER + '\n' + body_code

    # Write to file
    with open(chart_py, 'w', encoding='utf-8') as f:
        f.write(full_code)

    print(f"  Created: {chart_py}")
    return True

def main():
    base_dir = Path(r"D:\Joerg\Research\slides\DataScience_3")

    # Mapping of generation scripts to lessons and chart folders
    # Format: (script_file, [(lesson, chart_folder, function_name), ...])

    mappings = []

    # We need to manually map functions to folders by reading the scripts
    # For now, let's create a targeted script that processes L02-L06 from the generation script

    # Read L02_L06 script and extract functions
    print("\nProcessing L02-L06...")
    script = base_dir / 'generate_L02_L06_charts.py'
    functions = extract_chart_functions(script)

    # L02 mappings
    l02_folders = [
        '01_list_indexing',
        '02_slicing_notation',
        '03_dictionary_structure',
        '04_nested_structures',
        '05_list_methods',
        '06_portfolio_dict',
        '07_list_comprehension',
        '08_structure_selection'
    ]

    for i, folder in enumerate(l02_folders, 1):
        func_name = f'l02_chart{i}'
        if func_name in functions:
            create_chart_py_file('L02_Data_Structures', folder, functions[func_name])

    print(f"\nL02: Processed {len(l02_folders)} charts")

    # Now we need to do the same for L03-L06
    # Let me check which scripts have which lessons

    print("\n" + "="*60)
    print("Individual chart.py files created successfully!")
    print("Next step: Run each chart.py to generate PDFs")

if __name__ == "__main__":
    main()
