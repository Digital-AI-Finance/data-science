"""
Simple approach: Run existing generation scripts and fix colors afterward if needed.
Since L04-L06 already use correct colors, we just need to ensure L01-L03 do too.
"""
import subprocess
import sys
from pathlib import Path

def main():
    base_dir = Path(r"D:\Joerg\Research\slides\DataScience_3")

    print("="*60)
    print("GENERATING ALL CHARTS - USING EXISTING SCRIPTS")
    print("="*60)

    # Scripts that need to run
    scripts = [
        ('generate_all_charts.py', 'L01'),
        ('generate_L02_L06_charts.py', 'L02'),
        ('generate_L03_charts.py', 'L03'),
        ('generate_L04_L06.py', 'L04-L06'),
    ]

    for script_name, desc in scripts:
        script_path = base_dir / script_name
        if not script_path.exists():
            print(f"\n{desc}: Script not found - {script_name}")
            continue

        print(f"\n{desc}: Running {script_name}...")
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=base_dir,
                capture_output=True,
                text=True,
                timeout=180
            )
            if result.returncode == 0:
                print(f"{desc}: SUCCESS")
                if result.stdout:
                    for line in result.stdout.split('\n')[-10:]:
                        if line.strip():
                            print(f"  {line}")
            else:
                print(f"{desc}: FAILED")
                print(f"Error: {result.stderr[:200]}")
        except Exception as e:
            print(f"{desc}: ERROR - {str(e)}")

    print("\n" + "="*60)
    print("ALL GENERATION SCRIPTS EXECUTED")
    print("="*60)

if __name__ == "__main__":
    main()
