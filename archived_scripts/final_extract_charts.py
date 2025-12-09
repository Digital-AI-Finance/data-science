"""
Final chart extraction that creates working individual chart.py files
by properly handling the path references and chart code structure.
"""
import re
from pathlib import Path

# Standard header with course colors
CHART_HEADER = '''"""Generated chart with course color palette"""
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

# Map old colors to course colors
COLOR_PRIMARY = MLPURPLE
COLOR_SECONDARY = MLPURPLE
COLOR_ACCENT = MLBLUE
COLOR_LIGHT = MLLAVENDER
COLOR_GREEN = MLGREEN
COLOR_ORANGE = MLORANGE
COLOR_RED = MLRED

'''

def fix_savefig_path(code, lesson_folder, chart_folder):
    """Fix the plt.savefig path to use current directory"""
    # Pattern 1a: plt.savefig(base_dir / 'folder' / 'chart.pdf', ...) - 2 levels
    code = re.sub(
        r"plt\.savefig\(base_dir\s*/\s*['\"][^'\"]*?['\"]\s*/\s*['\"]chart\.pdf['\"][^)]*?\)",
        "plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')",
        code,
        flags=re.DOTALL
    )

    # Pattern 1b: plt.savefig(base_dir / 'lesson' / 'folder' / 'chart.pdf', ...) - 3 levels - multiline
    code = re.sub(
        r"plt\.savefig\(base_dir\s*/\s*['\"][^'\"]*?['\"]\s*/\s*['\"][^'\"]*?['\"]\s*/\s*['\"]chart\.pdf['\"][^)]*?\)",
        "plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')",
        code,
        flags=re.DOTALL
    )

    # Pattern 2: output_path = base_dir / ...
    code = re.sub(
        r"output_path\s*=\s*base_dir\s*/\s*['\"][^'\"]*?['\"]\s*/\s*['\"][^'\"]*?['\"]\s*/\s*['\"]chart\.pdf['\"]",
        "output_path = 'chart.pdf'",
        code
    )

    # Pattern 3: save_chart(fig, ...) function calls
    code = re.sub(
        r"save_chart\(fig,\s*['\"][^'\"]*?['\"]\s*,\s*['\"][^'\"]*?['\"]\)",
        "plt.savefig('chart.pdf', format='pdf', bbox_inches='tight', dpi=150); plt.close(fig)",
        code
    )

    # Pattern 4: Just replace any remaining base_dir with Path('.')
    if 'base_dir' in code:
        code = "base_dir = Path('.')\n" + code

    return code

def color_replace(code):
    """Replace old color hex values with course color variables"""
    replacements = [
        ("'#9B7EBD'", "COLOR_PRIMARY"),
        ("'#6B5B95'", "COLOR_SECONDARY"),
        ("'#4A90E2'", "COLOR_ACCENT"),
        ("'#ADADE0'", "COLOR_LIGHT"),
        ("'#44A05B'", "COLOR_GREEN"),
    ]

    for old, new in replacements:
        code = code.replace(old, new)

    return code

def extract_function_body(func_code):
    """Extract the body of a function, removing def line and dedenting"""
    lines = func_code.split('\n')

    # Find where actual code starts
    start_idx = 0
    for i, line in enumerate(lines):
        if 'fig' in line and ('plt.' in line or '=' in line):
            start_idx = i
            break

    body_lines = lines[start_idx:]

    # Dedent
    deindented = []
    for line in body_lines:
        if line.startswith('    '):
            deindented.append(line[4:])
        else:
            deindented.append(line)

    return '\n'.join(deindented)

def process_lesson_from_functions(script_path, lesson_folder, chart_folders, pattern):
    """Extract charts from scripts that use function definitions"""
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    functions = re.findall(pattern, content, re.DOTALL)

    print(f"Found {len(functions)} functions for {lesson_folder}")

    for i, func_code in enumerate(functions):
        if i >= len(chart_folders):
            break

        folder_name = chart_folders[i]
        chart_dir = Path(r"D:\Joerg\Research\slides\DataScience_3") / lesson_folder / folder_name
        chart_py = chart_dir / 'chart.py'

        body_code = extract_function_body(func_code)
        body_code = color_replace(body_code)
        body_code = fix_savefig_path(body_code, lesson_folder, folder_name)

        # Remove print statements
        body_code = re.sub(r"\nprint\([^)]*\)\s*", "\n", body_code)

        full_code = CHART_HEADER + '\n' + body_code

        with open(chart_py, 'w', encoding='utf-8') as f:
            f.write(full_code)

        print(f"  {i+1}/8: {folder_name}")

def process_L03_direct_code():
    """L03 has direct code, not functions"""
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

    # Split by Chart comments
    parts = re.split(r'# Chart \d+:[^\n]*\n', content)

    print(f"Found {len(parts)-1} chart sections for L03")

    for i in range(1, min(len(parts), 9)):
        chart_code = parts[i]
        folder_name = chart_folders[i-1]
        chart_dir = Path(r"D:\Joerg\Research\slides\DataScience_3\L03_Control_Flow") / folder_name
        chart_py = chart_dir / 'chart.py'

        chart_code = color_replace(chart_code)
        chart_code = fix_savefig_path(chart_code, 'L03_Control_Flow', folder_name)
        chart_code = re.sub(r"\nprint\([^)]*\)\s*", "\n", chart_code)

        full_code = CHART_HEADER + '\n' + chart_code

        with open(chart_py, 'w', encoding='utf-8') as f:
            f.write(full_code)

        print(f"  {i}/8: {folder_name}")

def process_L04_L05_L06_from_functions():
    """L04-L06 already use course colors, just need to extract to individual files"""
    script_path = Path(r"D:\Joerg\Research\slides\DataScience_3\generate_L04_L06.py")
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lessons = [
        ('L04_Functions', [
            '01_function_anatomy', '02_parameter_passing', '03_return_flowchart',
            '04_scope_diagram', '05_docstring_format', '06_call_stack',
            '07_pure_vs_impure', '08_finance_functions'
        ]),
        ('L05_DataFrames_Introduction', [
            '01_dataframe_structure', '02_series_vs_dataframe', '03_csv_loading',
            '04_head_tail', '05_info_breakdown', '06_describe_stats',
            '07_index_columns', '08_stock_example'
        ]),
        ('L06_Selection_Filtering', [
            '01_column_selection', '02_iloc_vs_loc', '03_boolean_mask',
            '04_conditional_filtering', '05_multiple_conditions', '06_chained_filtering',
            '07_selection_comparison', '08_stock_screening'
        ])
    ]

    for lesson_name, chart_folders in lessons:
        # Extract function
        pattern = rf'def generate_{lesson_name.split("_")[0]}_charts\(\):.*?(?=\ndef |if __name__|$)'
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            print(f"Could not find function for {lesson_name}")
            continue

        func_content = match.group(0)

        # Split by # Chart X: comments
        parts = re.split(r'# Chart \d+:[^\n]*\n', func_content)

        print(f"Found {len(parts)-1} chart sections for {lesson_name}")

        for i in range(1, min(len(parts), 9)):
            chart_code = parts[i]
            folder_name = chart_folders[i-1]
            chart_dir = Path(r"D:\Joerg\Research\slides\DataScience_3") / lesson_name / folder_name
            chart_py = chart_dir / 'chart.py'

            # Dedent the code (it's inside a function so has 4-space indent)
            lines = chart_code.split('\n')
            deindented = []
            for line in lines:
                if line.startswith('    '):
                    deindented.append(line[4:])
                else:
                    deindented.append(line)
            chart_code = '\n'.join(deindented)

            # L04-L06 already use course colors, just fix paths
            chart_code = fix_savefig_path(chart_code, lesson_name, folder_name)
            chart_code = re.sub(r"\nprint\([^)]*\)\s*", "\n", chart_code, flags=re.DOTALL)
            # Also remove any orphaned closing parens and quotes from multiline prints
            chart_code = re.sub(r'^\s*["\'][^"\']*["\']\s*\)\s*$', '', chart_code, flags=re.MULTILINE)

            # Add course color definitions at top
            header = CHART_HEADER.replace("# Map old colors to course colors", "# Course colors already used in this lesson")
            full_code = header + '\n' + chart_code

            with open(chart_py, 'w', encoding='utf-8') as f:
                f.write(full_code)

            print(f"  {i}/8: {folder_name}")

def main():
    print("="*60)
    print("FINAL CHART EXTRACTION - CREATING INDIVIDUAL FILES")
    print("="*60)

    # L01
    print("\nProcessing L01...")
    l01_folders = [
        '01_jupyter_interface', '02_data_types_hierarchy', '03_variable_assignment',
        '04_int_vs_float', '05_string_operations', '06_boolean_logic',
        '07_type_conversion', '08_python_vs_excel'
    ]
    process_lesson_from_functions(
        Path(r"D:\Joerg\Research\slides\DataScience_3\generate_all_charts.py"),
        'L01_Python_Setup',
        l01_folders,
        r'def generate_l01_chart\d+\(\):.*?(?=\ndef |# =====|$)'
    )

    # L02
    print("\nProcessing L02...")
    l02_folders = [
        '01_list_indexing', '02_slicing_notation', '03_dictionary_structure',
        '04_nested_structures', '05_list_methods', '06_portfolio_dict',
        '07_list_comprehension', '08_structure_selection'
    ]
    process_lesson_from_functions(
        Path(r"D:\Joerg\Research\slides\DataScience_3\generate_L02_L06_charts.py"),
        'L02_Data_Structures',
        l02_folders,
        r'def l02_chart\d+\(\):.*?(?=\ndef l02_chart|# Execute L02|$)'
    )

    # L03
    print("\nProcessing L03...")
    process_L03_direct_code()

    # L04-L06
    print("\nProcessing L04-L06...")
    process_L04_L05_L06_from_functions()

    print("\n" + "="*60)
    print("ALL INDIVIDUAL CHART.PY FILES CREATED!")
    print("="*60)

if __name__ == "__main__":
    main()
