"""
Quality check for all charts in DataScience_3 project
- Verifies chart.pdf exists in each chart folder
- Checks file sizes for anomalies
- Reports statistics
"""
from pathlib import Path
import os

BASE = Path(__file__).parent

def check_all_charts():
    """Check all chart folders for chart.py and chart.pdf files."""
    results = {
        'total_lessons': 0,
        'total_chart_folders': 0,
        'charts_with_pdf': 0,
        'charts_with_py': 0,
        'missing_pdf': [],
        'missing_py': [],
        'small_files': [],  # < 1KB
        'large_files': [],  # > 1MB
        'sizes': [],
    }

    for lesson_dir in sorted(BASE.iterdir()):
        if not (lesson_dir.is_dir() and lesson_dir.name.startswith('L') and '_' in lesson_dir.name):
            continue

        results['total_lessons'] += 1
        lesson_name = lesson_dir.name

        for chart_folder in sorted(lesson_dir.iterdir()):
            # Skip non-chart folders (like 'data', 'temp')
            if not chart_folder.is_dir():
                continue
            if chart_folder.name in ['data', 'temp', 'previous']:
                continue
            # Only count folders with numeric prefix (chart folders)
            if not (chart_folder.name[0].isdigit() or chart_folder.name.startswith('0')):
                continue

            results['total_chart_folders'] += 1

            chart_pdf = chart_folder / 'chart.pdf'
            chart_py = chart_folder / 'chart.py'

            # Check PDF
            if chart_pdf.exists():
                results['charts_with_pdf'] += 1
                size = chart_pdf.stat().st_size
                results['sizes'].append(size)

                if size < 1024:  # < 1KB
                    results['small_files'].append((lesson_name, chart_folder.name, size))
                elif size > 1024 * 1024:  # > 1MB
                    results['large_files'].append((lesson_name, chart_folder.name, size))
            else:
                results['missing_pdf'].append(f"{lesson_name}/{chart_folder.name}")

            # Check PY
            if chart_py.exists():
                results['charts_with_py'] += 1
            else:
                results['missing_py'].append(f"{lesson_name}/{chart_folder.name}")

    return results

def print_report(results):
    """Print quality check report."""
    print("=" * 70)
    print("QUALITY CHECK REPORT - DataScience_3 Charts")
    print("=" * 70)

    print(f"\n[SUMMARY]")
    print(f"  Total lessons:       {results['total_lessons']}")
    print(f"  Total chart folders: {results['total_chart_folders']}")
    print(f"  Charts with PDF:     {results['charts_with_pdf']}")
    print(f"  Charts with PY:      {results['charts_with_py']}")

    # Calculate coverage
    if results['total_chart_folders'] > 0:
        pdf_coverage = results['charts_with_pdf'] / results['total_chart_folders'] * 100
        py_coverage = results['charts_with_py'] / results['total_chart_folders'] * 100
        print(f"\n  PDF coverage:        {pdf_coverage:.1f}%")
        print(f"  PY coverage:         {py_coverage:.1f}%")

    # Size statistics
    if results['sizes']:
        sizes = results['sizes']
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        total_size = sum(sizes)
        print(f"\n[FILE SIZES]")
        print(f"  Average:             {avg_size/1024:.1f} KB")
        print(f"  Min:                 {min_size/1024:.1f} KB")
        print(f"  Max:                 {max_size/1024:.1f} KB")
        print(f"  Total:               {total_size/1024/1024:.1f} MB")

    # Issues
    print(f"\n[ISSUES]")

    if results['missing_pdf']:
        print(f"\n  Missing PDFs ({len(results['missing_pdf'])}):")
        for path in results['missing_pdf'][:10]:
            print(f"    - {path}")
        if len(results['missing_pdf']) > 10:
            print(f"    ... and {len(results['missing_pdf']) - 10} more")
    else:
        print(f"  Missing PDFs: None")

    if results['missing_py']:
        print(f"\n  Missing PY files ({len(results['missing_py'])}):")
        for path in results['missing_py'][:10]:
            print(f"    - {path}")
        if len(results['missing_py']) > 10:
            print(f"    ... and {len(results['missing_py']) - 10} more")
    else:
        print(f"  Missing PY files: None")

    if results['small_files']:
        print(f"\n  Unusually small PDFs (<1KB): {len(results['small_files'])}")
        for lesson, chart, size in results['small_files']:
            print(f"    - {lesson}/{chart}: {size} bytes")
    else:
        print(f"  Unusually small PDFs: None")

    if results['large_files']:
        print(f"\n  Large PDFs (>1MB): {len(results['large_files'])}")
        for lesson, chart, size in results['large_files'][:5]:
            print(f"    - {lesson}/{chart}: {size/1024/1024:.2f} MB")
    else:
        print(f"  Large PDFs: None")

    # Final status
    print("\n" + "=" * 70)
    if not results['missing_pdf'] and not results['missing_py'] and not results['small_files']:
        print("STATUS: ALL CHARTS PASS QUALITY CHECK")
    else:
        issues = len(results['missing_pdf']) + len(results['missing_py']) + len(results['small_files'])
        print(f"STATUS: {issues} ISSUES FOUND - Review required")
    print("=" * 70)

def main():
    results = check_all_charts()
    print_report(results)

if __name__ == '__main__':
    main()
