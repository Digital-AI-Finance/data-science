"""
Verify all 48 charts (L01-L06) have working PDFs and check file sizes.
"""

from pathlib import Path

# Working directory
BASE_DIR = Path(__file__).parent

# All lessons to verify
LESSONS = [
    'L01_Python_Setup',
    'L02_Data_Structures',
    'L03_Control_Flow',
    'L04_Functions',
    'L05_DataFrames_Introduction',
    'L06_Selection_Filtering'
]

def verify_chart(chart_path):
    """Verify chart has a valid PDF."""
    chart_dir = chart_path.parent
    pdf_path = chart_dir / 'chart.pdf'

    if not pdf_path.exists():
        return False, "PDF not found", 0

    # Check file size
    size_kb = pdf_path.stat().st_size / 1024

    if size_kb < 1:
        return False, "PDF too small", size_kb

    return True, "OK", size_kb

def main():
    """Verify all L01-L06 charts."""
    ok_count = 0
    fail_count = 0
    total_count = 0
    total_size_kb = 0

    for lesson in LESSONS:
        lesson_dir = BASE_DIR / lesson
        if not lesson_dir.exists():
            print(f"Warning: {lesson} not found")
            continue

        # Find all chart.py files
        chart_files = list(lesson_dir.glob('**/chart.py'))
        print(f"\n{lesson}: Verifying {len(chart_files)} charts")

        lesson_ok = 0
        lesson_fail = 0

        for chart_path in sorted(chart_files):
            total_count += 1
            relative_path = chart_path.relative_to(BASE_DIR)

            success, msg, size_kb = verify_chart(chart_path)
            if success:
                print(f"  OK: {relative_path.parent.name} ({size_kb:.1f} KB)")
                ok_count += 1
                lesson_ok += 1
                total_size_kb += size_kb
            else:
                print(f"  FAIL: {relative_path.parent.name} - {msg}")
                fail_count += 1
                lesson_fail += 1

        print(f"  Lesson Summary: {lesson_ok} OK, {lesson_fail} FAIL")

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total Charts: {total_count}")
    print(f"OK: {ok_count}")
    print(f"FAIL: {fail_count}")
    print(f"Total Size: {total_size_kb:.1f} KB ({total_size_kb/1024:.1f} MB)")
    print(f"Avg Size: {total_size_kb/ok_count:.1f} KB per chart" if ok_count > 0 else "")
    print(f"{'='*60}")

    if fail_count == 0:
        print("\nSUCCESS: All 48 charts verified!")
    else:
        print(f"\nFAILURE: {fail_count} charts missing or invalid")

if __name__ == '__main__':
    main()
