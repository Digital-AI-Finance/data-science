"""
Run all chart.py files to generate PDFs with improved colors
"""
import subprocess
from pathlib import Path
import sys

def run_chart(chart_py_path):
    """Run a chart.py file and return success status"""
    try:
        result = subprocess.run(
            [sys.executable, str(chart_py_path)],
            cwd=chart_py_path.parent,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

def main():
    base_dir = Path(r"D:\Joerg\Research\slides\DataScience_3")

    # Define all lessons and folders
    lessons = {
        'L01_Python_Setup': [
            '01_jupyter_interface',
            '02_data_types_hierarchy',
            '03_variable_assignment',
            '04_int_vs_float',
            '05_string_operations',
            '06_boolean_logic',
            '07_type_conversion',
            '08_python_vs_excel'
        ],
        'L02_Data_Structures': [
            '01_list_indexing',
            '02_slicing_notation',
            '03_dictionary_structure',
            '04_nested_structures',
            '05_list_methods',
            '06_portfolio_dict',
            '07_list_comprehension',
            '08_structure_selection'
        ],
        'L03_Control_Flow': [
            '01_if_else_flowchart',
            '02_for_loop_visual',
            '03_while_loop',
            '04_nested_loops',
            '05_break_continue',
            '06_trading_rules',
            '07_loop_comparison',
            '08_control_patterns'
        ],
        'L04_Functions': [
            '01_function_anatomy',
            '02_parameter_passing',
            '03_return_flowchart',
            '04_scope_diagram',
            '05_docstring_format',
            '06_call_stack',
            '07_pure_vs_impure',
            '08_finance_functions'
        ],
        'L05_DataFrames_Introduction': [
            '01_dataframe_structure',
            '02_series_vs_dataframe',
            '03_csv_loading',
            '04_head_tail',
            '05_info_breakdown',
            '06_describe_stats',
            '07_index_columns',
            '08_stock_example'
        ],
        'L06_Selection_Filtering': [
            '01_column_selection',
            '02_iloc_vs_loc',
            '03_boolean_mask',
            '04_conditional_filtering',
            '05_multiple_conditions',
            '06_chained_filtering',
            '07_selection_comparison',
            '08_stock_screening'
        ]
    }

    print("="*60)
    print("GENERATING ALL CHART PDFs FOR L01-L06")
    print("="*60)

    total_charts = 0
    successful = 0
    failed = []

    for lesson, folders in lessons.items():
        print(f"\n{lesson}...")
        lesson_path = base_dir / lesson

        for i, folder in enumerate(folders, 1):
            chart_py = lesson_path / folder / 'chart.py'

            if not chart_py.exists():
                print(f"  {i}/8: {folder} - MISSING chart.py")
                failed.append(f"{lesson}/{folder}")
                continue

            total_charts += 1
            success, output = run_chart(chart_py)

            if success:
                successful += 1
                print(f"  {i}/8: {folder} - OK")
            else:
                print(f"  {i}/8: {folder} - FAILED")
                print(f"       Error: {output[:100]}")
                failed.append(f"{lesson}/{folder}")

    print("\n" + "="*60)
    print(f"SUMMARY")
    print("="*60)
    print(f"Total charts: {total_charts}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed charts:")
        for f in failed:
            print(f"  - {f}")

if __name__ == "__main__":
    main()
