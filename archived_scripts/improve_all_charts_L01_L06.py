"""
Script to improve all charts in L01-L06 by reading existing chart.py files
and creating improved versions with proper styling and course colors.
"""
import os
from pathlib import Path
import subprocess

# Course colors - MANDATORY
MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

# Standard rcParams for all charts
STANDARD_RC_PARAMS = """import matplotlib.pyplot as plt
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
"""

def find_chart_folders(base_dir, lessons):
    """Find all chart folders in specified lessons"""
    chart_folders = []

    for lesson in lessons:
        lesson_path = Path(base_dir) / lesson
        if not lesson_path.exists():
            print(f"Warning: {lesson} not found")
            continue

        # Find all subfolders that might contain charts
        for subfolder in sorted(lesson_path.iterdir()):
            if subfolder.is_dir() and not subfolder.name.startswith('.'):
                # Skip empty chart folders and data folders
                if 'chart' in subfolder.name.lower() and len(list(subfolder.iterdir())) == 0:
                    continue
                if subfolder.name == 'data':
                    continue

                # Check if it has a chart.py or chart.pdf
                has_py = (subfolder / 'chart.py').exists()
                has_pdf = (subfolder / 'chart.pdf').exists()

                if has_py or has_pdf:
                    chart_folders.append({
                        'path': subfolder,
                        'lesson': lesson,
                        'name': subfolder.name,
                        'has_py': has_py,
                        'has_pdf': has_pdf
                    })

    return chart_folders

def main():
    base_dir = Path(r"D:\Joerg\Research\slides\DataScience_3")
    lessons = ['L01_Python_Setup', 'L02_Data_Structures', 'L03_Control_Flow',
               'L04_Functions', 'L05_DataFrames_Introduction', 'L06_Selection_Filtering']

    chart_folders = find_chart_folders(base_dir, lessons)

    print(f"\nFound {len(chart_folders)} chart folders:")
    print("=" * 80)

    for cf in chart_folders:
        status = []
        if cf['has_py']:
            status.append('chart.py')
        if cf['has_pdf']:
            status.append('chart.pdf')
        print(f"{cf['lesson']:30s} {cf['name']:30s} [{', '.join(status)}]")

    print("\n" + "=" * 80)
    print(f"Total: {len(chart_folders)} charts")
    print(f"With chart.py: {sum(1 for cf in chart_folders if cf['has_py'])}")
    print(f"PDF only: {sum(1 for cf in chart_folders if cf['has_pdf'] and not cf['has_py'])}")

if __name__ == "__main__":
    main()
