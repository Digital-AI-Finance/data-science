"""
Generate master course index in both Markdown and HTML formats
- Extracts lesson info from .tex files
- Counts slides and charts per lesson
- Outputs: COURSE_INDEX.md and index.html
"""
from pathlib import Path
import re
from datetime import datetime

BASE = Path(__file__).parent

# Course module structure
MODULES = {
    1: ("Python Fundamentals", range(1, 7)),
    2: ("Data Manipulation", range(7, 13)),
    3: ("Statistics & Visualization", range(13, 21)),
    4: ("ML: Regression", range(21, 25)),
    5: ("ML: Classification", range(25, 29)),
    6: ("ML: Unsupervised", range(29, 33)),
    7: ("Deep Learning", range(33, 37)),
    8: ("NLP & Text", range(37, 41)),
    9: ("Deployment", range(41, 45)),
    10: ("Capstone & Ethics", range(45, 49)),
}

def get_lesson_info(lesson_folder):
    """Extract info from lesson folder."""
    lesson_name = lesson_folder.name
    match = re.match(r'L(\d+)_(.+)', lesson_name)
    if not match:
        return None

    num = int(match.group(1))
    title = match.group(2).replace('_', ' ')

    # Count chart folders
    chart_folders = [
        d for d in lesson_folder.iterdir()
        if d.is_dir() and d.name[0].isdigit() and (d / 'chart.pdf').exists()
    ]

    # Read tex file for slide count
    tex_file = lesson_folder / f"{lesson_name}.tex"
    slides = 0
    if tex_file.exists():
        content = tex_file.read_text(encoding='utf-8', errors='ignore')
        # Count \begin{frame} occurrences
        slides = len(re.findall(r'\\begin\{frame\}', content))

    # Get chart names
    chart_names = sorted([d.name for d in chart_folders])

    return {
        'num': num,
        'title': title,
        'folder': lesson_name,
        'slides': slides,
        'charts': len(chart_folders),
        'chart_names': chart_names,
    }

def generate_markdown(lessons):
    """Generate COURSE_INDEX.md content."""
    lines = []
    lines.append("# Data Science with Python - Course Index")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    # Summary
    total_slides = sum(l['slides'] for l in lessons)
    total_charts = sum(l['charts'] for l in lessons)
    lines.append("## Course Overview")
    lines.append("")
    lines.append(f"- **Lessons:** 48")
    lines.append(f"- **Total Slides:** {total_slides}")
    lines.append(f"- **Total Charts:** {total_charts}")
    lines.append(f"- **Modules:** 10")
    lines.append("")

    # Module breakdown
    for mod_num, (mod_name, lesson_range) in MODULES.items():
        lines.append(f"## Module {mod_num}: {mod_name}")
        lines.append("")

        mod_lessons = [l for l in lessons if l['num'] in lesson_range]
        for lesson in mod_lessons:
            lines.append(f"### L{lesson['num']:02d}: {lesson['title']}")
            lines.append("")
            lines.append(f"- **Slides:** {lesson['slides']} | **Charts:** {lesson['charts']}")
            lines.append(f"- **Folder:** `{lesson['folder']}/`")
            if lesson['chart_names']:
                charts_str = ", ".join(lesson['chart_names'][:5])
                if len(lesson['chart_names']) > 5:
                    charts_str += f" (+{len(lesson['chart_names']) - 5} more)"
                lines.append(f"- **Charts:** {charts_str}")
            lines.append("")

    # Quick reference
    lines.append("---")
    lines.append("")
    lines.append("## Quick Reference")
    lines.append("")
    lines.append("### By Topic")
    lines.append("")
    lines.append("| Topic | Lessons |")
    lines.append("|-------|---------|")
    lines.append("| Python Basics | L01-L06 |")
    lines.append("| Pandas/NumPy | L05-L12 |")
    lines.append("| Statistics | L13-L16 |")
    lines.append("| Visualization | L17-L20 |")
    lines.append("| Regression | L21-L24 |")
    lines.append("| Classification | L25-L28 |")
    lines.append("| Clustering/PCA | L29-L32 |")
    lines.append("| Neural Networks | L33-L36 |")
    lines.append("| NLP | L37-L40 |")
    lines.append("| Deployment | L41-L44 |")
    lines.append("")

    lines.append("### By Difficulty")
    lines.append("")
    lines.append("| Level | Lessons | Topics |")
    lines.append("|-------|---------|--------|")
    lines.append("| Beginner | L01-L12 | Python, DataFrames, Basic Ops |")
    lines.append("| Intermediate | L13-L32 | Statistics, ML Algorithms |")
    lines.append("| Advanced | L33-L48 | Deep Learning, NLP, Deployment |")
    lines.append("")

    return "\n".join(lines)

def generate_html(lessons):
    """Generate index.html content."""
    total_slides = sum(l['slides'] for l in lessons)
    total_charts = sum(l['charts'] for l in lessons)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Science with Python - Course Index</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{ background: #2c3e50; color: white; padding: 30px 20px; margin-bottom: 30px; border-radius: 8px; }}
        h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .stats {{ display: flex; gap: 30px; margin-top: 15px; }}
        .stat {{ background: rgba(255,255,255,0.1); padding: 10px 20px; border-radius: 5px; }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.8; }}
        .module {{ background: white; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }}
        .module-header {{ background: #3498db; color: white; padding: 15px 20px; font-size: 1.2em; font-weight: bold; }}
        .lessons {{ padding: 0; }}
        .lesson {{ padding: 15px 20px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }}
        .lesson:last-child {{ border-bottom: none; }}
        .lesson:hover {{ background: #f8f9fa; }}
        .lesson-title {{ font-weight: 500; }}
        .lesson-num {{ color: #3498db; font-weight: bold; margin-right: 10px; }}
        .lesson-meta {{ display: flex; gap: 15px; font-size: 0.9em; color: #666; }}
        .badge {{ background: #ecf0f1; padding: 3px 10px; border-radius: 12px; }}
        footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.9em; }}
        .search {{ margin-bottom: 20px; }}
        .search input {{ width: 100%; padding: 12px 20px; font-size: 1em; border: 2px solid #ddd; border-radius: 25px; outline: none; }}
        .search input:focus {{ border-color: #3498db; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Data Science with Python</h1>
            <p>BSc Course - Complete Index</p>
            <div class="stats">
                <div class="stat"><div class="stat-value">48</div><div class="stat-label">Lessons</div></div>
                <div class="stat"><div class="stat-value">{total_slides}</div><div class="stat-label">Slides</div></div>
                <div class="stat"><div class="stat-value">{total_charts}</div><div class="stat-label">Charts</div></div>
                <div class="stat"><div class="stat-value">10</div><div class="stat-label">Modules</div></div>
            </div>
        </header>

        <div class="search">
            <input type="text" id="searchInput" placeholder="Search lessons..." onkeyup="filterLessons()">
        </div>
'''

    # Generate modules
    for mod_num, (mod_name, lesson_range) in MODULES.items():
        mod_lessons = [l for l in lessons if l['num'] in lesson_range]
        html += f'''
        <div class="module">
            <div class="module-header">Module {mod_num}: {mod_name}</div>
            <div class="lessons">
'''
        for lesson in mod_lessons:
            html += f'''                <div class="lesson" data-title="{lesson['title'].lower()}">
                    <div class="lesson-title">
                        <span class="lesson-num">L{lesson['num']:02d}</span>
                        {lesson['title']}
                    </div>
                    <div class="lesson-meta">
                        <span class="badge">{lesson['slides']} slides</span>
                        <span class="badge">{lesson['charts']} charts</span>
                    </div>
                </div>
'''
        html += '''            </div>
        </div>
'''

    html += f'''
        <footer>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            <p><a href="DataScience_3_Complete.pdf">Download Complete PDF (8.2 MB)</a></p>
        </footer>
    </div>

    <script>
        function filterLessons() {{
            const query = document.getElementById('searchInput').value.toLowerCase();
            document.querySelectorAll('.lesson').forEach(lesson => {{
                const title = lesson.dataset.title;
                lesson.style.display = title.includes(query) ? 'flex' : 'none';
            }});
        }}
    </script>
</body>
</html>
'''
    return html

def main():
    print("=" * 50)
    print("Generating Course Index")
    print("=" * 50)

    # Collect lesson info
    lessons = []
    for lesson_folder in sorted(BASE.iterdir()):
        if lesson_folder.is_dir() and lesson_folder.name.startswith('L') and '_' in lesson_folder.name:
            info = get_lesson_info(lesson_folder)
            if info:
                lessons.append(info)
                print(f"  L{info['num']:02d}: {info['title']} ({info['slides']} slides, {info['charts']} charts)")

    print(f"\nTotal: {len(lessons)} lessons")

    # Generate Markdown
    md_content = generate_markdown(lessons)
    md_path = BASE / "COURSE_INDEX.md"
    md_path.write_text(md_content, encoding='utf-8')
    print(f"\nCreated: {md_path.name}")

    # Generate HTML
    html_content = generate_html(lessons)
    html_path = BASE / "index.html"
    html_path.write_text(html_content, encoding='utf-8')
    print(f"Created: {html_path.name}")

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)

if __name__ == '__main__':
    main()
