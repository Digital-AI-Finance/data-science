"""
Generate master course index in both Markdown and HTML formats
- Extracts lesson info from .tex files
- Counts slides and charts per lesson
- Outputs: COURSE_INDEX.md and index.html (quality-optimized)
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
        slides = len(re.findall(r'\\begin\{frame\}', content))

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

    total_slides = sum(l['slides'] for l in lessons)
    total_charts = sum(l['charts'] for l in lessons)
    lines.append("## Course Overview")
    lines.append("")
    lines.append(f"- **Lessons:** 48")
    lines.append(f"- **Total Slides:** {total_slides}")
    lines.append(f"- **Total Charts:** {total_charts}")
    lines.append(f"- **Modules:** 10")
    lines.append("")

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
    """Generate quality-optimized index.html content."""
    total_slides = sum(l['slides'] for l in lessons)
    total_charts = sum(l['charts'] for l in lessons)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Data Science with Python BSc Course - 48 lessons covering Python, ML, NLP, and deployment. {total_slides} slides, {total_charts} charts.">
    <meta property="og:title" content="Data Science with Python - Course Index">
    <meta property="og:description" content="48 lessons, {total_slides} slides, {total_charts} charts - Complete BSc curriculum covering Python to ML deployment">
    <meta property="og:image" content="https://digital-ai-finance.github.io/data-science/og-image.png">
    <meta property="og:url" content="https://digital-ai-finance.github.io/data-science/">
    <meta property="og:type" content="website">
    <title>Data Science with Python - Course Index</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 100 100%27><text y=%27.9em%27 font-size=%2790%27>&#128202;</text></svg>">
    <style>
        :root {{
            --bg: #f5f5f5;
            --bg-card: white;
            --text: #333;
            --text-muted: #666;
            --border: #eee;
            --primary: #3498db;
            --primary-dark: #2c3e50;
            --hover: #f8f9fa;
            --badge-bg: #ecf0f1;
        }}
        [data-theme="dark"] {{
            --bg: #1a1a2e;
            --bg-card: #16213e;
            --text: #eee;
            --text-muted: #aaa;
            --border: #0f3460;
            --primary: #4fa3d1;
            --primary-dark: #0f3460;
            --hover: #1f4068;
            --badge-bg: #0f3460;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: var(--text); background: var(--bg); transition: background 0.3s ease, color 0.3s ease; }}
        a {{ color: var(--primary); text-decoration: none; transition: opacity 0.2s ease; }}
        a:hover {{ opacity: 0.8; }}

        /* Navigation */
        nav {{ position: sticky; top: 0; background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%); color: white; padding: 12px 20px; z-index: 100; box-shadow: 0 2px 10px rgba(0,0,0,0.2); }}
        .nav-content {{ max-width: 1400px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center; gap: 20px; }}
        .nav-brand {{ font-size: 1.2em; font-weight: bold; display: flex; align-items: center; gap: 8px; }}
        .nav-brand img {{ width: 28px; height: 28px; }}
        .nav-links {{ display: flex; gap: 15px; flex-wrap: wrap; }}
        .nav-links a {{ color: white; font-size: 0.9em; padding: 4px 8px; border-radius: 4px; transition: background 0.2s ease; }}
        .nav-links a:hover {{ background: rgba(255,255,255,0.2); opacity: 1; }}
        .nav-actions {{ display: flex; gap: 10px; align-items: center; }}
        .btn {{ padding: 8px 16px; border-radius: 20px; font-size: 0.9em; cursor: pointer; border: none; transition: all 0.2s ease; }}
        .btn-primary {{ background: white; color: var(--primary-dark); }}
        .btn-primary:hover {{ transform: translateY(-1px); box-shadow: 0 2px 8px rgba(0,0,0,0.2); }}
        .btn-icon {{ background: rgba(255,255,255,0.2); color: white; width: 36px; height: 36px; padding: 0; display: flex; align-items: center; justify-content: center; font-size: 1.2em; }}

        /* Layout */
        .layout {{ display: flex; max-width: 1400px; margin: 0 auto; min-height: calc(100vh - 60px); }}
        aside {{ width: 220px; padding: 20px; position: sticky; top: 60px; height: calc(100vh - 60px); overflow-y: auto; background: var(--bg-card); border-right: 1px solid var(--border); transition: background 0.3s ease; }}
        main {{ flex: 1; padding: 20px 30px; }}

        /* Sidebar */
        .sidebar-title {{ font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 15px; }}
        .sidebar-nav {{ list-style: none; }}
        .sidebar-nav li {{ margin-bottom: 8px; }}
        .sidebar-nav a {{ color: var(--text); font-size: 0.9em; display: block; padding: 6px 10px; border-radius: 6px; transition: all 0.2s ease; }}
        .sidebar-nav a:hover {{ background: var(--hover); color: var(--primary); }}
        .progress-box {{ margin-top: 20px; padding: 15px; background: var(--hover); border-radius: 8px; }}
        .progress-label {{ font-size: 0.8em; color: var(--text-muted); margin-bottom: 8px; }}
        .progress-bar {{ height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, var(--primary) 0%, #2ecc71 100%); width: 100%; }}

        /* Hero */
        .hero {{ background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%); color: white; padding: 40px 30px; border-radius: 12px; margin-bottom: 25px; }}
        .hero h1 {{ font-size: 2.2em; margin-bottom: 8px; }}
        .hero p {{ opacity: 0.9; margin-bottom: 20px; }}
        .stats {{ display: flex; gap: 25px; flex-wrap: wrap; }}
        .stat {{ background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 8px; text-align: center; min-width: 100px; transition: transform 0.2s ease; }}
        .stat:hover {{ transform: translateY(-2px); }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ font-size: 0.85em; opacity: 0.8; }}

        /* Search */
        .search {{ margin-bottom: 25px; position: relative; }}
        .search input {{ width: 100%; padding: 14px 20px 14px 45px; font-size: 1em; border: 2px solid var(--border); border-radius: 25px; outline: none; background: var(--bg-card); color: var(--text); transition: all 0.2s ease; }}
        .search input:focus {{ border-color: var(--primary); box-shadow: 0 0 0 3px rgba(52,152,219,0.2); }}
        .search-icon {{ position: absolute; left: 18px; top: 50%; transform: translateY(-50%); color: var(--text-muted); }}

        /* Modules */
        .modules {{ display: flex; flex-direction: column; gap: 20px; }}
        article {{ background: var(--bg-card); border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; transition: all 0.3s ease; }}
        article:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.12); }}
        details {{ }}
        summary {{ background: linear-gradient(135deg, var(--primary) 0%, #5dade2 100%); color: white; padding: 18px 25px; font-size: 1.15em; font-weight: 600; cursor: pointer; list-style: none; display: flex; justify-content: space-between; align-items: center; transition: background 0.2s ease; }}
        summary::-webkit-details-marker {{ display: none; }}
        summary::after {{ content: '+'; font-size: 1.4em; font-weight: 300; transition: transform 0.2s ease; }}
        details[open] summary::after {{ content: '-'; }}
        summary:hover {{ filter: brightness(1.05); }}
        .lessons {{ padding: 0; }}
        .lesson {{ padding: 16px 25px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; transition: background 0.2s ease; }}
        .lesson:last-child {{ border-bottom: none; }}
        .lesson:hover {{ background: var(--hover); }}
        .lesson-link {{ display: flex; align-items: center; gap: 12px; color: var(--text); flex: 1; }}
        .lesson-num {{ color: var(--primary); font-weight: bold; font-size: 0.95em; min-width: 35px; }}
        .lesson-title {{ font-weight: 500; }}
        .lesson-meta {{ display: flex; gap: 12px; font-size: 0.85em; }}
        .badge {{ background: var(--badge-bg); color: var(--text-muted); padding: 4px 12px; border-radius: 15px; transition: all 0.2s ease; }}
        .lesson:hover .badge {{ background: var(--primary); color: white; }}

        /* Footer */
        footer {{ text-align: center; padding: 30px 20px; color: var(--text-muted); font-size: 0.9em; border-top: 1px solid var(--border); margin-top: 30px; }}
        footer a {{ color: var(--primary); }}

        /* Responsive */
        @media (max-width: 1024px) {{
            aside {{ display: none; }}
            .layout {{ display: block; }}
            main {{ padding: 15px; }}
        }}
        @media (max-width: 768px) {{
            .nav-links {{ display: none; }}
            .hero {{ padding: 25px 20px; }}
            .hero h1 {{ font-size: 1.6em; }}
            .stats {{ gap: 15px; }}
            .stat {{ padding: 12px 18px; min-width: 80px; }}
            .stat-value {{ font-size: 1.5em; }}
            summary {{ padding: 14px 18px; font-size: 1em; }}
            .lesson {{ padding: 12px 18px; flex-wrap: wrap; gap: 8px; }}
            .lesson-meta {{ width: 100%; justify-content: flex-start; }}
        }}
        @media (max-width: 480px) {{
            .nav-brand span {{ display: none; }}
            .stats {{ flex-direction: column; gap: 10px; }}
            .stat {{ display: flex; justify-content: space-between; align-items: center; text-align: left; }}
        }}
    </style>
</head>
<body>
    <nav aria-label="Main navigation">
        <div class="nav-content">
            <div class="nav-brand">
                <span>Data Science with Python</span>
            </div>
            <div class="nav-links" role="navigation">
                <a href="#mod-1">Fundamentals</a>
                <a href="#mod-2">Data</a>
                <a href="#mod-3">Stats</a>
                <a href="#mod-4">Regression</a>
                <a href="#mod-5">Classification</a>
                <a href="#mod-6">Unsupervised</a>
                <a href="#mod-7">Deep Learning</a>
                <a href="#mod-8">NLP</a>
                <a href="#mod-9">Deploy</a>
                <a href="#mod-10">Capstone</a>
            </div>
            <div class="nav-actions">
                <a href="DataScience_3_Complete.pdf" class="btn btn-primary" aria-label="Download complete PDF">Download PDF</a>
                <button class="btn btn-icon" onclick="toggleTheme()" aria-label="Toggle dark mode" title="Toggle dark/light mode">&#9790;</button>
            </div>
        </div>
    </nav>

    <div class="layout">
        <aside aria-label="Module navigation">
            <div class="sidebar-title">Modules</div>
            <ul class="sidebar-nav">
'''

    # Sidebar module links
    for mod_num, (mod_name, _) in MODULES.items():
        short_name = mod_name.split(':')[-1].strip() if ':' in mod_name else mod_name
        html += f'                <li><a href="#mod-{mod_num}">M{mod_num}: {short_name}</a></li>\n'

    html += f'''            </ul>
            <div class="progress-box">
                <div class="progress-label">Course Progress</div>
                <div class="progress-bar"><div class="progress-fill"></div></div>
                <div style="font-size:0.9em; margin-top:8px; font-weight:600;">48 Lessons Complete</div>
            </div>
        </aside>

        <main role="main">
            <header class="hero">
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
                <span class="search-icon">&#128269;</span>
                <input type="text" id="searchInput" placeholder="Search lessons..." onkeyup="filterLessons()" aria-label="Search lessons">
            </div>

            <section class="modules" aria-label="Course modules">
'''

    # Generate modules with collapsible details
    for mod_num, (mod_name, lesson_range) in MODULES.items():
        mod_lessons = [l for l in lessons if l['num'] in lesson_range]
        html += f'''                <article id="mod-{mod_num}">
                    <details open>
                        <summary id="module-{mod_num}-heading">Module {mod_num}: {mod_name}</summary>
                        <div class="lessons" role="list" aria-labelledby="module-{mod_num}-heading">
'''
        for lesson in mod_lessons:
            pdf_path = f"{lesson['folder']}/{lesson['folder']}.pdf"
            html += f'''                            <div class="lesson" data-title="{lesson['title'].lower()}" role="listitem">
                                <a href="{pdf_path}" target="_blank" class="lesson-link" aria-label="Open {lesson['title']} PDF">
                                    <span class="lesson-num">L{lesson['num']:02d}</span>
                                    <span class="lesson-title">{lesson['title']}</span>
                                </a>
                                <div class="lesson-meta">
                                    <span class="badge">{lesson['slides']} slides</span>
                                    <span class="badge">{lesson['charts']} charts</span>
                                </div>
                            </div>
'''
        html += '''                        </div>
                    </details>
                </article>
'''

    html += f'''            </section>
        </main>
    </div>

    <footer>
        <p>Generated: {timestamp} | <a href="DataScience_3_Complete.pdf">Download Complete PDF (8.2 MB)</a></p>
        <p>Data Science with Python - BSc Course</p>
    </footer>

    <script>
        function filterLessons() {{
            const query = document.getElementById('searchInput').value.toLowerCase();
            document.querySelectorAll('.lesson').forEach(lesson => {{
                const title = lesson.dataset.title;
                lesson.style.display = title.includes(query) ? 'flex' : 'none';
            }});
            // Show all modules when searching
            if (query) {{
                document.querySelectorAll('details').forEach(d => d.open = true);
            }}
        }}

        function toggleTheme() {{
            const body = document.body;
            const currentTheme = body.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            body.setAttribute('data-theme', newTheme === 'dark' ? 'dark' : '');
            localStorage.setItem('theme', newTheme);
            // Update button icon
            const btn = document.querySelector('.btn-icon');
            btn.innerHTML = newTheme === 'dark' ? '&#9788;' : '&#9790;';
        }}

        // Load saved theme
        (function() {{
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'dark') {{
                document.body.setAttribute('data-theme', 'dark');
                const btn = document.querySelector('.btn-icon');
                if (btn) btn.innerHTML = '&#9788;';
            }}
        }})();
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
