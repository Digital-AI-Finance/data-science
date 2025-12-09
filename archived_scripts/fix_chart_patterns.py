"""
Fix savefig/close patterns in L04-L06 charts.

Changes:
- plt.savefig('chart.pdf', format='pdf', bbox_inches='tight', dpi=150); plt.close(fig)
TO:
- plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
- plt.close()
"""

from pathlib import Path
import re

# Working directory
BASE_DIR = Path(__file__).parent

# Lessons to fix
LESSONS = ['L04_Functions', 'L05_DataFrames_Introduction', 'L06_Selection_Filtering']

def fix_chart_file(chart_path):
    """Fix savefig/close pattern in a chart.py file."""
    content = chart_path.read_text(encoding='utf-8')
    original_content = content

    # Pattern 1: Inline semicolon pattern
    # plt.savefig('chart.pdf', format='pdf', bbox_inches='tight', dpi=150); plt.close(fig)
    pattern1 = r"plt\.savefig\('chart\.pdf',\s*format='pdf',\s*bbox_inches='tight',\s*dpi=150\);\s*plt\.close\(fig\)"
    replacement1 = "plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')\nplt.close()"
    content = re.sub(pattern1, replacement1, content)

    # Pattern 2: Multiline pattern with format='pdf' and dpi=150
    pattern2 = r"plt\.savefig\('chart\.pdf',\s*format='pdf',\s*bbox_inches='tight',\s*dpi=150\)\s*\n\s*plt\.close\(fig\)"
    replacement2 = "plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')\nplt.close()"
    content = re.sub(pattern2, replacement2, content)

    # Pattern 3: Any savefig with dpi=150 (catch-all)
    pattern3 = r"plt\.savefig\('chart\.pdf',\s*format='pdf',\s*bbox_inches='tight',\s*dpi=150\)"
    replacement3 = "plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')"
    content = re.sub(pattern3, replacement3, content)

    # Pattern 4: plt.close(fig) -> plt.close()
    pattern4 = r"plt\.close\(fig\)"
    replacement4 = "plt.close()"
    content = re.sub(pattern4, replacement4, content)

    if content != original_content:
        chart_path.write_text(content, encoding='utf-8')
        return True
    return False

def main():
    """Fix all chart.py files in L04-L06."""
    fixed_count = 0
    total_count = 0

    for lesson in LESSONS:
        lesson_dir = BASE_DIR / lesson
        if not lesson_dir.exists():
            print(f"Warning: {lesson} not found")
            continue

        # Find all chart.py files
        chart_files = list(lesson_dir.glob('**/chart.py'))
        print(f"\n{lesson}: Found {len(chart_files)} chart files")

        for chart_path in sorted(chart_files):
            total_count += 1
            relative_path = chart_path.relative_to(BASE_DIR)

            if fix_chart_file(chart_path):
                print(f"  FIXED: {relative_path}")
                fixed_count += 1
            else:
                print(f"  SKIP: {relative_path} (no changes needed)")

    print(f"\n{'='*60}")
    print(f"Summary: Fixed {fixed_count}/{total_count} chart files")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
