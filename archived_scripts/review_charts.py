"""Review all chart.py files in L01-L06"""
import os
import re
from pathlib import Path

# Working directory
base_dir = Path(r"D:\Joerg\Research\slides\DataScience_3")

# Lessons to review
lessons = ['L01_Python_Setup', 'L02_Data_Structures', 'L03_Control_Flow',
           'L04_Functions', 'L05_DataFrames_Introduction', 'L06_Selection_Filtering']

# Course colors
COURSE_COLORS = {
    'MLPURPLE': '#3333B2',
    'MLLAVENDER': '#ADADE0',
    'MLBLUE': '#0066CC',
    'MLORANGE': '#FF7F0E',
    'MLGREEN': '#2CA02C',
    'MLRED': '#D62728'
}

issues = []
charts_reviewed = 0

for lesson in lessons:
    lesson_path = base_dir / lesson
    if not lesson_path.exists():
        continue

    # Find all chart.py files
    chart_files = list(lesson_path.glob("*/chart.py"))

    for chart_file in chart_files:
        charts_reviewed += 1
        chart_name = f"{lesson}/{chart_file.parent.name}"

        try:
            content = chart_file.read_text()

            # Check 1: Font size >= 8pt
            font_size_match = re.search(r"'font\.size':\s*(\d+)", content)
            if font_size_match:
                font_size = int(font_size_match.group(1))
                if font_size < 8:
                    issues.append(f"{chart_name}: Font size {font_size}pt < 8pt minimum")

            # Check 2: Uses course colors
            uses_course_colors = all(color in content for color in ['MLPURPLE', 'MLBLUE'])
            if not uses_course_colors:
                issues.append(f"{chart_name}: Missing course color definitions")

            # Check 3: Figure size
            if 'figsize' not in content:
                issues.append(f"{chart_name}: No figure size specified")

            # Check 4: savefig issues
            if 'plt.savefig' in content:
                # Check for incorrect format parameter
                if "format='pdf'" in content:
                    issues.append(f"{chart_name}: Uses format='pdf' parameter (should omit)")
                # Check for semicolon inline close
                if '; plt.close(fig)' in content:
                    issues.append(f"{chart_name}: Uses semicolon inline close (should be separate line)")
            elif 'output_path' in content:
                # Good - using output_path variable
                pass
            else:
                issues.append(f"{chart_name}: No savefig call found")

            # Check 5: Title present
            if 'ax.text' not in content and 'plt.title' not in content and 'set_title' not in content:
                issues.append(f"{chart_name}: No title found")

        except Exception as e:
            issues.append(f"{chart_name}: ERROR reading file - {str(e)}")

print(f"Charts Reviewed: {charts_reviewed}")
print(f"Issues Found: {len(issues)}\n")

if issues:
    print("ISSUES FOUND:")
    print("=" * 80)
    for issue in issues:
        print(f"  - {issue}")
else:
    print("All charts passed quality checks!")
