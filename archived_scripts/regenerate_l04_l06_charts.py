"""
Regenerate all chart PDFs for L04-L06 after fixing savefig patterns.
"""

from pathlib import Path
import subprocess
import sys

# Working directory
BASE_DIR = Path(__file__).parent

# Lessons to regenerate
LESSONS = ['L04_Functions', 'L05_DataFrames_Introduction', 'L06_Selection_Filtering']

def regenerate_chart(chart_path):
    """Regenerate a single chart PDF."""
    chart_dir = chart_path.parent
    pdf_path = chart_dir / 'chart.pdf'

    # Delete old PDF if exists
    if pdf_path.exists():
        pdf_path.unlink()

    # Run chart.py
    result = subprocess.run(
        [sys.executable, str(chart_path)],
        cwd=str(chart_dir),
        capture_output=True,
        text=True,
        timeout=30
    )

    # Check if PDF was created
    if pdf_path.exists():
        return True, "OK"
    else:
        error_msg = result.stderr if result.stderr else result.stdout
        return False, error_msg[:100]

def main():
    """Regenerate all chart PDFs in L04-L06."""
    success_count = 0
    fail_count = 0
    total_count = 0

    for lesson in LESSONS:
        lesson_dir = BASE_DIR / lesson
        if not lesson_dir.exists():
            print(f"Warning: {lesson} not found")
            continue

        # Find all chart.py files
        chart_files = list(lesson_dir.glob('**/chart.py'))
        print(f"\n{lesson}: Regenerating {len(chart_files)} charts")

        for chart_path in sorted(chart_files):
            total_count += 1
            relative_path = chart_path.relative_to(BASE_DIR)

            success, msg = regenerate_chart(chart_path)
            if success:
                print(f"  OK: {relative_path}")
                success_count += 1
            else:
                print(f"  FAIL: {relative_path}")
                print(f"    Error: {msg}")
                fail_count += 1

    print(f"\n{'='*60}")
    print(f"Summary: {success_count} OK, {fail_count} FAIL, {total_count} total")
    print(f"{'='*60}")

    if fail_count > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
