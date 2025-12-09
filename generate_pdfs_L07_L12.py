"""Generate all chart PDFs for L07-L12"""
import subprocess
import sys
from pathlib import Path

base_dir = Path(__file__).parent

# L07-L12 lessons
lessons = [
    'L07_Missing_Data',
    'L08_Basic_Operations',
    'L09_GroupBy_Operations',
    'L10_Merging_Joining',
    'L11_NumPy_Basics',
    'L12_Time_Series'
]

success_count = 0
fail_count = 0
failed_charts = []

for lesson in lessons:
    lesson_dir = base_dir / lesson
    if not lesson_dir.exists():
        print(f"SKIP: {lesson} folder not found")
        continue

    # Find all chart folders
    chart_folders = sorted([d for d in lesson_dir.iterdir()
                           if d.is_dir() and d.name[:2].isdigit()])

    print(f"\n{'='*60}")
    print(f"Processing {lesson}: {len(chart_folders)} charts")
    print('='*60)

    for chart_folder in chart_folders:
        chart_py = chart_folder / 'chart.py'
        if not chart_py.exists():
            print(f"  SKIP: {chart_folder.name} - no chart.py")
            continue

        try:
            result = subprocess.run(
                [sys.executable, str(chart_py)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(chart_folder)
            )

            if result.returncode == 0:
                chart_pdf = chart_folder / 'chart.pdf'
                if chart_pdf.exists():
                    print(f"  OK: {chart_folder.name}")
                    success_count += 1
                else:
                    print(f"  WARN: {chart_folder.name} - no PDF created")
                    fail_count += 1
                    failed_charts.append(f"{lesson}/{chart_folder.name}")
            else:
                print(f"  FAIL: {chart_folder.name}")
                print(f"        {result.stderr[:200]}")
                fail_count += 1
                failed_charts.append(f"{lesson}/{chart_folder.name}")

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {chart_folder.name}")
            fail_count += 1
            failed_charts.append(f"{lesson}/{chart_folder.name}")
        except Exception as e:
            print(f"  ERROR: {chart_folder.name} - {e}")
            fail_count += 1
            failed_charts.append(f"{lesson}/{chart_folder.name}")

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"Successful: {success_count}")
print(f"Failed: {fail_count}")
if failed_charts:
    print("\nFailed charts:")
    for fc in failed_charts:
        print(f"  - {fc}")
