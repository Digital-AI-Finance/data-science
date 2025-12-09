"""Project Overview - Final Project Introduction"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (10, 6),
    'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Final Project Overview: Putting It All Together', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Project goals
ax1 = axes[0, 0]
ax1.axis('off')

goals = '''
FINAL PROJECT GOALS

DEMONSTRATE YOUR SKILLS:
------------------------
- Apply data science concepts learned
- Build a complete end-to-end solution
- Show practical problem-solving ability


WHAT YOU WILL BUILD:
--------------------
1. A data-driven finance application
2. With machine learning predictions
3. Deployed to the cloud
4. Shareable with anyone


PROJECT SCOPE:
--------------
- Individual project (no teams)
- 2 weeks of work
- Presentation on final day
- 100% of course grade


DELIVERABLES:
-------------
1. Working deployed application
2. Source code (GitHub repository)
3. Documentation (README)
4. 5-7 minute presentation
5. Live demo


SUCCESS CRITERIA:
-----------------
- App works without errors
- Predictions are reasonable
- Code is clean and organized
- Presentation is clear
'''

ax1.text(0.02, 0.98, goals, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Project Goals', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Skills applied
ax2 = axes[0, 1]

skills = [
    ('Python Programming', 100, MLBLUE),
    ('Data Manipulation (pandas)', 95, MLBLUE),
    ('Data Visualization', 90, MLORANGE),
    ('Machine Learning', 85, MLGREEN),
    ('Model Evaluation', 80, MLGREEN),
    ('API Development', 70, MLPURPLE),
    ('Streamlit Apps', 75, MLPURPLE),
    ('Cloud Deployment', 65, MLRED)
]

skills_sorted = sorted(skills, key=lambda x: x[1], reverse=True)
names = [s[0] for s in skills_sorted]
values = [s[1] for s in skills_sorted]
colors = [s[2] for s in skills_sorted]

y_pos = np.arange(len(names))
bars = ax2.barh(y_pos, values, color=colors, alpha=0.7)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(names)
ax2.set_xlabel('Skill Application Level (%)')
ax2.set_xlim(0, 110)
ax2.set_title('Skills You Will Apply', fontsize=11, fontweight='bold', color=MLPURPLE)

for bar, val in zip(bars, values):
    ax2.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val}%',
             va='center', fontsize=8)

# Plot 3: Project timeline
ax3 = axes[1, 0]
ax3.axis('off')

# Draw timeline
phases = [
    ('Week 11', 'Setup & Planning', ['Topic selection', 'Data acquisition', 'Project setup'], MLBLUE),
    ('Week 11-12', 'Development', ['Model building', 'App creation', 'Testing'], MLGREEN),
    ('Week 12', 'Deployment', ['Cloud deploy', 'Bug fixes', 'Polish'], MLORANGE),
    ('Final Day', 'Present', ['Demo', 'Q&A', 'Submit'], MLRED)
]

for i, (time, phase, tasks, color) in enumerate(phases):
    x = 0.12 + i * 0.23
    ax3.add_patch(plt.Rectangle((x-0.08, 0.3), 0.18, 0.6, facecolor=color, alpha=0.2))
    ax3.text(x, 0.85, time, fontsize=9, ha='center', fontweight='bold')
    ax3.text(x, 0.75, phase, fontsize=10, ha='center', fontweight='bold', color=color)
    for j, task in enumerate(tasks):
        ax3.text(x, 0.6 - j*0.1, f'- {task}', fontsize=8, ha='center')

# Arrows
for i in range(3):
    x1 = 0.26 + i * 0.23
    ax3.annotate('', xy=(x1+0.04, 0.6), xytext=(x1-0.04, 0.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

ax3.set_xlim(0, 1)
ax3.set_ylim(0.2, 1)
ax3.set_title('Project Timeline', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: What makes a good project
ax4 = axes[1, 1]
ax4.axis('off')

good_project = '''
WHAT MAKES A GOOD PROJECT?

TECHNICAL EXCELLENCE:
---------------------
- Clean, readable code
- Proper error handling
- Efficient data processing
- Appropriate model choice
- Good evaluation metrics


USER EXPERIENCE:
----------------
- Intuitive interface
- Clear visualizations
- Responsive design
- Helpful instructions
- Professional appearance


DOCUMENTATION:
--------------
- Clear README
- Installation instructions
- Usage examples
- Known limitations
- Future improvements


PRESENTATION:
-------------
- Clear problem statement
- Logical structure
- Engaging demo
- Honest about limitations
- Professional delivery


BONUS POINTS:
-------------
- Creative approach
- Extra features
- Strong visualizations
- Novel data sources
- Clear business value
'''

ax4.text(0.02, 0.98, good_project, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('What Makes a Good Project?', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
