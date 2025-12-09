"""Next Steps - Your Data Science Journey Continues"""
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
fig.suptitle('Next Steps: Your Data Science Journey Continues', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Career paths
ax1 = axes[0, 0]
ax1.axis('off')

careers = '''
DATA SCIENCE CAREER PATHS

DATA ANALYST:
-------------
Focus: Reporting, visualization, SQL
Tools: Excel, Tableau, SQL, Python
Entry: Most accessible starting point


DATA SCIENTIST:
---------------
Focus: ML models, predictions, insights
Tools: Python, scikit-learn, TensorFlow
Entry: After more ML experience


ML ENGINEER:
------------
Focus: Production ML systems, scale
Tools: Docker, Kubernetes, MLOps
Entry: Requires software engineering


QUANT ANALYST:
--------------
Focus: Financial modeling, trading
Tools: Python, R, C++, statistics
Entry: Strong math background needed


DATA ENGINEER:
--------------
Focus: Data pipelines, infrastructure
Tools: SQL, Spark, Airflow, AWS
Entry: Software engineering focus


RESEARCH SCIENTIST:
-------------------
Focus: Novel algorithms, publications
Tools: PyTorch, academic methods
Entry: Usually requires PhD


YOUR PATH:
----------
This course prepared you for:
- Junior Data Analyst roles
- Entry-level Data Science
- Further specialized study
- Finance analytics positions
'''

ax1.text(0.02, 0.98, careers, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Career Paths', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Learning roadmap
ax2 = axes[0, 1]
ax2.axis('off')

# Draw learning path
levels = [
    ('NOW', 'Fundamentals\nComplete!', 0.1, MLGREEN),
    ('NEXT', 'Intermediate\nML/DL', 0.35, MLBLUE),
    ('THEN', 'Specialization\nChoose focus', 0.6, MLORANGE),
    ('GOAL', 'Expert\nPractitioner', 0.85, MLPURPLE)
]

for label, desc, x, color in levels:
    ax2.add_patch(plt.Circle((x, 0.6), 0.08, facecolor=color, alpha=0.5))
    ax2.text(x, 0.6, label, fontsize=10, ha='center', va='center', fontweight='bold')
    ax2.text(x, 0.4, desc, fontsize=8, ha='center')

# Arrows
for i in range(len(levels)-1):
    x1 = levels[i][2] + 0.08
    x2 = levels[i+1][2] - 0.08
    ax2.annotate('', xy=(x2, 0.6), xytext=(x1, 0.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

ax2.text(0.5, 0.9, 'YOUR LEARNING ROADMAP', fontsize=11, ha='center', fontweight='bold')

# Time estimates
ax2.text(0.22, 0.25, '3-6 months', fontsize=8, ha='center', color='gray')
ax2.text(0.47, 0.25, '6-12 months', fontsize=8, ha='center', color='gray')
ax2.text(0.72, 0.25, '1-2 years', fontsize=8, ha='center', color='gray')

ax2.set_xlim(0, 1)
ax2.set_ylim(0.1, 1)
ax2.set_title('Learning Roadmap', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Resources for continued learning
ax3 = axes[1, 0]
ax3.axis('off')

resources = '''
RESOURCES FOR CONTINUED LEARNING

ONLINE COURSES:
---------------
- Coursera: ML Specialization (Stanford)
- Fast.ai: Practical Deep Learning
- DataCamp: Data Science tracks
- Kaggle Learn: Free micro-courses


BOOKS:
------
- "Hands-On ML" (Geron)
- "Python for Data Analysis" (McKinney)
- "Deep Learning" (Goodfellow)
- "Introduction to Statistical Learning"


PRACTICE PLATFORMS:
-------------------
- Kaggle: Competitions & datasets
- LeetCode: Coding practice
- HackerRank: Data science challenges
- GitHub: Open source projects


COMMUNITIES:
------------
- Reddit: r/datascience, r/MachineLearning
- Stack Overflow: Technical Q&A
- LinkedIn: Professional networking
- Twitter: Follow researchers


CERTIFICATIONS:
---------------
- Google Data Analytics
- AWS Machine Learning
- TensorFlow Developer
- Microsoft Azure Data Scientist


FINANCE-SPECIFIC:
-----------------
- CFA (Chartered Financial Analyst)
- FRM (Financial Risk Manager)
- QuantNet: Quant finance courses
- Quantopian/QuantConnect: Algo trading


BUILD YOUR PORTFOLIO:
---------------------
1. Create GitHub profile
2. Add this project!
3. Build 2-3 more projects
4. Write blog posts
5. Contribute to open source
'''

ax3.text(0.02, 0.98, resources, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Learning Resources', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Final message
ax4 = axes[1, 1]
ax4.axis('off')

# Create inspiring final visual
ax4.add_patch(plt.Rectangle((0.1, 0.2), 0.8, 0.7, facecolor=MLLAVENDER, alpha=0.5))

ax4.text(0.5, 0.8, 'CONGRATULATIONS!', fontsize=16, ha='center',
         fontweight='bold', color=MLPURPLE)

ax4.text(0.5, 0.65, 'You have completed', fontsize=12, ha='center')
ax4.text(0.5, 0.55, 'Introduction to Data Science', fontsize=14, ha='center',
         fontweight='bold', color=MLBLUE)

achievements = [
    '48 lessons completed',
    '384 concepts learned',
    '1 deployed project',
    'Countless skills gained'
]

for i, achievement in enumerate(achievements):
    ax4.text(0.5, 0.42 - i*0.06, f'* {achievement}', fontsize=10, ha='center')

ax4.text(0.5, 0.15, '"The journey of a thousand miles\n begins with a single step."',
         fontsize=10, ha='center', style='italic', color='gray')

ax4.text(0.5, 0.05, 'Your data science journey has just begun!',
         fontsize=11, ha='center', fontweight='bold', color=MLGREEN)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title('Final Message', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
