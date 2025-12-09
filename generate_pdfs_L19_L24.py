"""Generate all PDFs for L19-L24 (Batch 4)"""
import subprocess
import sys
from pathlib import Path

def run_chart_script(script_path):
    """Run a chart.py script and return success status."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=script_path.parent
        )
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

def main():
    base_dir = Path(__file__).parent

    # Lessons L19-L24
    lessons = [
        'L19_Multi_Panel_Figures',
        'L20_Data_Storytelling',
        'L21_Linear_Regression',
        'L22_Regularization',
        'L23_Regression_Metrics',
        'L24_Factor_Models'
    ]

    total = 0
    successful = 0
    failed = []

    for lesson in lessons:
        lesson_dir = base_dir / lesson
        if not lesson_dir.exists():
            print(f"WARNING: {lesson} directory not found")
            continue

        # Find all chart.py files
        chart_scripts = sorted(lesson_dir.glob('*/chart.py'))

        for script in chart_scripts:
            total += 1
            chart_name = script.parent.name
            print(f"Processing {lesson}/{chart_name}...", end=' ')

            success, error = run_chart_script(script)

            if success:
                successful += 1
                print("OK")
            else:
                failed.append(f"{lesson}/{chart_name}: {error}")
                print(f"FAILED: {error[:50]}...")

    print(f"\n{'='*50}")
    print(f"SUMMARY - Batch 4 (L19-L24)")
    print(f"{'='*50}")
    print(f"Total: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed charts:")
        for f in failed:
            print(f"  - {f}")

if __name__ == '__main__':
    main()
