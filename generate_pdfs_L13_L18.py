"""Generate PDFs for L13-L18 charts"""
import subprocess
import sys
from pathlib import Path

base_dir = Path(__file__).parent
lessons = ['L13_Descriptive_Statistics', 'L14_Distributions', 'L15_Hypothesis_Testing',
           'L16_Correlation', 'L17_Matplotlib_Basics', 'L18_Seaborn_Plots']

successful = 0
failed = 0
errors = []

for lesson in lessons:
    lesson_path = base_dir / lesson
    if not lesson_path.exists():
        print(f"[SKIP] {lesson} - folder not found")
        continue

    for chart_folder in sorted(lesson_path.iterdir()):
        if not chart_folder.is_dir():
            continue

        chart_py = chart_folder / 'chart.py'
        if not chart_py.exists():
            continue

        print(f"[RUN] {lesson}/{chart_folder.name}/chart.py", end=' ... ')

        try:
            result = subprocess.run(
                [sys.executable, str(chart_py)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(chart_folder)
            )

            pdf_path = chart_folder / 'chart.pdf'
            if pdf_path.exists() and result.returncode == 0:
                print("OK")
                successful += 1
            else:
                print("FAILED")
                failed += 1
                errors.append((f"{lesson}/{chart_folder.name}", result.stderr[:200] if result.stderr else "No PDF generated"))
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            failed += 1
            errors.append((f"{lesson}/{chart_folder.name}", "Timeout after 60s"))
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
            errors.append((f"{lesson}/{chart_folder.name}", str(e)))

print(f"\n{'='*50}")
print(f"SUMMARY: {successful} successful, {failed} failed")
print(f"{'='*50}")

if errors:
    print("\nFailed charts:")
    for path, err in errors:
        print(f"  - {path}: {err[:100]}")
