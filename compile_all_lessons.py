"""
Compile all 48 lesson .tex files to PDF using pdflatex.
Moves auxiliary files to temp folder after compilation.
"""

import subprocess
import os
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"

# Auxiliary file extensions to move
AUX_EXTENSIONS = ['.aux', '.log', '.nav', '.snm', '.toc', '.out', '.vrb']

def compile_tex(tex_file: Path) -> tuple[bool, str]:
    """Compile a single .tex file using pdflatex."""
    lesson_dir = tex_file.parent

    try:
        # Run pdflatex twice for proper references
        for run in range(2):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', tex_file.name],
                cwd=lesson_dir,
                capture_output=True,
                text=True,
                timeout=120
            )

        # Check if PDF was created
        pdf_file = tex_file.with_suffix('.pdf')
        if pdf_file.exists():
            return True, f"Success: {pdf_file.name}"
        else:
            return False, f"Failed: PDF not created"

    except subprocess.TimeoutExpired:
        return False, "Failed: Timeout"
    except FileNotFoundError:
        return False, "Failed: pdflatex not found"
    except Exception as e:
        return False, f"Failed: {str(e)}"

def move_aux_files(lesson_dir: Path, lesson_name: str):
    """Move auxiliary files to temp folder."""
    lesson_temp = TEMP_DIR / lesson_name
    lesson_temp.mkdir(parents=True, exist_ok=True)

    for ext in AUX_EXTENSIONS:
        for aux_file in lesson_dir.glob(f'*{ext}'):
            dest = lesson_temp / aux_file.name
            shutil.move(str(aux_file), str(dest))

def main():
    print("=" * 60)
    print("COMPILING ALL 48 LESSONS")
    print("=" * 60)

    # Create temp directory
    TEMP_DIR.mkdir(exist_ok=True)

    # Find all lesson folders
    lesson_folders = sorted([
        d for d in BASE_DIR.iterdir()
        if d.is_dir() and d.name.startswith('L') and '_' in d.name
    ])

    print(f"Found {len(lesson_folders)} lesson folders\n")

    success_count = 0
    failed_lessons = []

    for i, lesson_dir in enumerate(lesson_folders, 1):
        lesson_name = lesson_dir.name

        # Find .tex file in lesson folder
        tex_files = list(lesson_dir.glob('*.tex'))

        if not tex_files:
            print(f"[{i}/48] {lesson_name}: No .tex file found")
            failed_lessons.append((lesson_name, "No .tex file"))
            continue

        tex_file = tex_files[0]
        print(f"[{i}/48] {lesson_name}...", end=" ", flush=True)

        success, message = compile_tex(tex_file)

        if success:
            success_count += 1
            print("OK")
            # Move aux files
            move_aux_files(lesson_dir, lesson_name)
        else:
            print(message)
            failed_lessons.append((lesson_name, message))

    print("\n" + "=" * 60)
    print(f"COMPILATION COMPLETE: {success_count}/{len(lesson_folders)} successful")
    print("=" * 60)

    if failed_lessons:
        print("\nFailed lessons:")
        for name, reason in failed_lessons:
            print(f"  - {name}: {reason}")

    print(f"\nAuxiliary files moved to: {TEMP_DIR}")

if __name__ == "__main__":
    main()
