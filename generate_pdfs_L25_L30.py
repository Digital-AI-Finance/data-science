"""Generate PDFs for L25-L30 charts"""
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent

lessons = [
    "L25_Logistic_Regression",
    "L26_Decision_Trees",
    "L27_Classification_Metrics",
    "L28_Class_Imbalance",
    "L29_KMeans_Clustering",
    "L30_Hierarchical_Clustering"
]

total = 0
success = 0
failed = []

for lesson in lessons:
    lesson_dir = BASE_DIR / lesson
    if not lesson_dir.exists():
        print(f"Lesson folder not found: {lesson}")
        continue

    chart_folders = sorted([d for d in lesson_dir.iterdir() if d.is_dir() and d.name[:2].isdigit()])

    for chart_folder in chart_folders:
        chart_py = chart_folder / "chart.py"
        if not chart_py.exists():
            continue

        total += 1
        print(f"Generating: {lesson}/{chart_folder.name}")

        try:
            result = subprocess.run(
                [sys.executable, str(chart_py)],
                cwd=str(chart_folder),
                capture_output=True,
                text=True,
                timeout=60
            )

            pdf_path = chart_folder / "chart.pdf"
            if pdf_path.exists():
                success += 1
                print(f"  OK: {pdf_path.name}")
            else:
                failed.append(f"{lesson}/{chart_folder.name}")
                print(f"  FAILED: No PDF generated")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            failed.append(f"{lesson}/{chart_folder.name} (timeout)")
            print(f"  FAILED: Timeout")
        except Exception as e:
            failed.append(f"{lesson}/{chart_folder.name} ({e})")
            print(f"  FAILED: {e}")

print(f"\n{'='*50}")
print(f"SUMMARY: {success}/{total} PDFs generated successfully")
if failed:
    print(f"Failed ({len(failed)}):")
    for f in failed:
        print(f"  - {f}")
