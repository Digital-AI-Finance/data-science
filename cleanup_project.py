"""
Cleanup script for DataScience_3 project
- Removes empty placeholder folders (##_chart pattern)
- Archives redundant scripts to archived_scripts/
- Removes empty 'nul' file
"""
from pathlib import Path
import shutil

BASE = Path(__file__).parent

# Scripts to archive (legacy/one-time use)
SCRIPTS_TO_ARCHIVE = [
    'generate_all_charts.py',
    'generate_L02_L06_charts.py',
    'generate_L03_charts.py',
    'generate_L04_L06.py',
    'generate_L07_L12.py',
    'generate_L13_L18.py',
    'generate_L19_L48.py',
    'extract_all_charts_L01_L06.py',
    'final_extract_charts.py',
    'create_individual_chart_files.py',
    'fix_and_generate_all_charts.py',
    'fix_chart_patterns.py',
    'regenerate_l04_l06_charts.py',
    'review_charts.py',
    'verify_all_l01_l06_charts.py',
    'verify_style_consistency.py',
    'improve_all_charts_L01_L06.py',
    'generate_all_pdfs_L01_L06.py',
]

def remove_empty_folders():
    """Remove empty ##_chart folders from lesson directories."""
    removed = 0
    for lesson_dir in sorted(BASE.iterdir()):
        if lesson_dir.is_dir() and lesson_dir.name.startswith('L'):
            for subfolder in list(lesson_dir.iterdir()):
                # Match pattern like 01_chart, 02_chart, etc.
                if subfolder.is_dir() and subfolder.name.endswith('_chart'):
                    # Check if empty
                    if not any(subfolder.iterdir()):
                        print(f"  Removing empty: {subfolder.relative_to(BASE)}")
                        subfolder.rmdir()
                        removed += 1
    return removed

def archive_scripts():
    """Move legacy scripts to archived_scripts/ folder."""
    archive_dir = BASE / 'archived_scripts'
    archive_dir.mkdir(exist_ok=True)

    archived = 0
    for script_name in SCRIPTS_TO_ARCHIVE:
        script_path = BASE / script_name
        if script_path.exists():
            dest = archive_dir / script_name
            print(f"  Archiving: {script_name}")
            shutil.move(str(script_path), str(dest))
            archived += 1
    return archived

def remove_nul_file():
    """Remove the empty 'nul' file."""
    nul_file = BASE / 'nul'
    if nul_file.exists():
        nul_file.unlink()
        print("  Removed: nul")
        return 1
    return 0

def main():
    print("=" * 60)
    print("DataScience_3 Project Cleanup")
    print("=" * 60)

    print("\n[1/3] Removing empty placeholder folders...")
    empty_removed = remove_empty_folders()
    print(f"    -> Removed {empty_removed} empty folders")

    print("\n[2/3] Archiving legacy scripts...")
    scripts_archived = archive_scripts()
    print(f"    -> Archived {scripts_archived} scripts to archived_scripts/")

    print("\n[3/3] Cleaning up misc files...")
    misc_removed = remove_nul_file()
    print(f"    -> Removed {misc_removed} misc files")

    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE")
    print(f"  - Empty folders removed: {empty_removed}")
    print(f"  - Scripts archived: {scripts_archived}")
    print(f"  - Misc files removed: {misc_removed}")
    print("=" * 60)

if __name__ == '__main__':
    main()
