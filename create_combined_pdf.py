"""
Create combined PDF from all 48 lesson PDFs
- Merges in order L01 through L48
- Adds bookmarks for navigation
- Output: DataScience_3_Complete.pdf
"""
from pathlib import Path
from PyPDF2 import PdfMerger, PdfReader
import re

BASE = Path(__file__).parent

def get_lesson_title(lesson_name):
    """Extract clean title from lesson folder name."""
    # L01_Python_Setup -> Python Setup
    match = re.match(r'L\d+_(.+)', lesson_name)
    if match:
        return match.group(1).replace('_', ' ')
    return lesson_name

def create_combined_pdf():
    """Merge all 48 lesson PDFs with bookmarks."""
    merger = PdfMerger()

    # Find all lesson folders
    lesson_folders = sorted([
        d for d in BASE.iterdir()
        if d.is_dir() and d.name.startswith('L') and '_' in d.name
    ], key=lambda x: int(re.match(r'L(\d+)', x.name).group(1)))

    print(f"Found {len(lesson_folders)} lessons to merge")
    print("-" * 50)

    total_pages = 0
    for i, lesson_folder in enumerate(lesson_folders, 1):
        lesson_name = lesson_folder.name
        pdf_name = f"{lesson_name}.pdf"
        pdf_path = lesson_folder / pdf_name

        if not pdf_path.exists():
            print(f"  [{i:02d}] SKIP - {lesson_name}: PDF not found")
            continue

        # Get page count
        try:
            reader = PdfReader(str(pdf_path))
            page_count = len(reader.pages)
        except Exception as e:
            print(f"  [{i:02d}] ERROR - {lesson_name}: {e}")
            continue

        # Create bookmark title
        lesson_num = re.match(r'L(\d+)', lesson_name).group(1)
        title = get_lesson_title(lesson_name)
        bookmark = f"L{lesson_num}: {title}"

        # Add to merger with bookmark
        merger.append(str(pdf_path), bookmark)
        total_pages += page_count

        print(f"  [{i:02d}] Added {lesson_name} ({page_count} pages)")

    # Write output
    output_path = BASE / "DataScience_3_Complete.pdf"
    print("-" * 50)
    print(f"Writing combined PDF...")

    merger.write(str(output_path))
    merger.close()

    # Get final file size
    size_mb = output_path.stat().st_size / 1024 / 1024

    print(f"\nOUTPUT: {output_path}")
    print(f"  Total lessons: {len(lesson_folders)}")
    print(f"  Total pages:   {total_pages}")
    print(f"  File size:     {size_mb:.1f} MB")

def main():
    print("=" * 50)
    print("Creating Combined PDF - DataScience_3")
    print("=" * 50)
    create_combined_pdf()
    print("=" * 50)
    print("DONE")
    print("=" * 50)

if __name__ == '__main__':
    main()
