"""
Verify style consistency across all L01-L06 charts.
Check that all charts use the standardized savefig/close pattern.
"""

from pathlib import Path
import re

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

# Expected patterns (allow both direct and variable-based)
EXPECTED_SAVEFIG_DIRECT = r"plt\.savefig\('chart\.pdf',\s*dpi=300,\s*bbox_inches='tight'\)"
EXPECTED_SAVEFIG_VAR = r"output_path\s*=\s*'chart\.pdf'[\s\S]*?plt\.savefig\(output_path,\s*dpi=300,\s*bbox_inches='tight'\)"
EXPECTED_CLOSE = r"plt\.close\(\)"

# Bad patterns (should not exist)
BAD_PATTERN_1 = r"format='pdf'"
BAD_PATTERN_2 = r"dpi=150"
BAD_PATTERN_3 = r"plt\.close\(fig\)"
BAD_PATTERN_4 = r";\s*plt\.close"  # Inline semicolon

def verify_chart_style(chart_path):
    """Verify chart uses correct savefig/close pattern."""
    content = chart_path.read_text(encoding='utf-8')

    issues = []

    # Check for bad patterns
    if re.search(BAD_PATTERN_1, content):
        issues.append("Uses format='pdf' (should be removed)")

    if re.search(BAD_PATTERN_2, content):
        issues.append("Uses dpi=150 (should be 300)")

    if re.search(BAD_PATTERN_3, content):
        issues.append("Uses plt.close(fig) (should be plt.close())")

    if re.search(BAD_PATTERN_4, content):
        issues.append("Uses inline semicolon pattern (should be multiline)")

    # Check for expected patterns (accept either direct or variable-based)
    has_savefig = re.search(EXPECTED_SAVEFIG_DIRECT, content) or re.search(EXPECTED_SAVEFIG_VAR, content)
    if not has_savefig:
        issues.append("Missing expected savefig pattern")

    if not re.search(EXPECTED_CLOSE, content):
        issues.append("Missing expected close pattern")

    return len(issues) == 0, issues

def main():
    """Verify style consistency across all charts."""
    ok_count = 0
    issue_count = 0
    total_count = 0

    for lesson in LESSONS:
        lesson_dir = BASE_DIR / lesson
        if not lesson_dir.exists():
            print(f"Warning: {lesson} not found")
            continue

        # Find all chart.py files
        chart_files = list(lesson_dir.glob('**/chart.py'))
        print(f"\n{lesson}: Checking {len(chart_files)} charts")

        lesson_ok = 0
        lesson_issues = 0

        for chart_path in sorted(chart_files):
            total_count += 1
            relative_path = chart_path.relative_to(BASE_DIR)

            is_ok, issues = verify_chart_style(chart_path)
            if is_ok:
                print(f"  OK: {relative_path.parent.name}")
                ok_count += 1
                lesson_ok += 1
            else:
                print(f"  ISSUES: {relative_path.parent.name}")
                for issue in issues:
                    print(f"    - {issue}")
                issue_count += 1
                lesson_issues += 1

        print(f"  Lesson Summary: {lesson_ok} OK, {lesson_issues} Issues")

    print(f"\n{'='*60}")
    print(f"STYLE VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Charts: {total_count}")
    print(f"OK: {ok_count}")
    print(f"Issues: {issue_count}")
    print(f"{'='*60}")

    if issue_count == 0:
        print("\nSUCCESS: All charts follow consistent style!")
    else:
        print(f"\nFAILURE: {issue_count} charts have style issues")

if __name__ == '__main__':
    main()
