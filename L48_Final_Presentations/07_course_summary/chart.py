"""Course Summary - What You've Learned"""
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
fig.suptitle('Course Summary: What You Have Learned', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Skills acquired
ax1 = axes[0, 0]
ax1.axis('off')

skills = '''
SKILLS YOU'VE ACQUIRED

PROGRAMMING (Weeks 1-3):
------------------------
- Python fundamentals
- pandas DataFrames
- NumPy arrays
- Data manipulation
- File handling


DATA SKILLS (Weeks 4-5):
------------------------
- Descriptive statistics
- Data visualization
- Exploratory analysis
- Data cleaning
- Feature engineering


MACHINE LEARNING (Weeks 6-8):
-----------------------------
- Regression models
- Classification models
- Clustering
- Model evaluation
- Cross-validation
- Hyperparameter tuning


DEEP LEARNING (Week 9):
-----------------------
- Neural network basics
- MLP architecture
- Training process
- Overfitting prevention


NLP (Week 10):
--------------
- Text preprocessing
- Bag of Words/TF-IDF
- Word embeddings
- Sentiment analysis


DEPLOYMENT (Week 11):
---------------------
- Model serialization
- API development
- Streamlit apps
- Cloud deployment


PROFESSIONAL (Week 12):
-----------------------
- Project management
- Presentation skills
- Ethics awareness
- Documentation
'''

ax1.text(0.02, 0.98, skills, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Skills Acquired', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Course journey visualization
ax2 = axes[0, 1]

weeks = ['W1-3', 'W4-5', 'W6-8', 'W9', 'W10', 'W11-12']
topics = ['Python\npandas', 'Stats\nViz', 'ML\nModels', 'Deep\nLearn', 'NLP', 'Deploy\nProject']
difficulty = [2, 3, 4, 4.5, 4, 3.5]
colors = [MLBLUE, MLORANGE, MLGREEN, MLPURPLE, MLRED, MLGREEN]

# Create journey path
x = np.arange(len(weeks))
ax2.plot(x, difficulty, 'o-', markersize=20, linewidth=3, color=MLBLUE, alpha=0.5)

for i, (topic, diff, color) in enumerate(zip(topics, difficulty, colors)):
    ax2.plot(i, diff, 'o', markersize=25, color=color, alpha=0.7)
    ax2.text(i, diff, topic, ha='center', va='center', fontsize=7, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(weeks)
ax2.set_ylabel('Complexity Level')
ax2.set_ylim(1, 5.5)
ax2.set_title('Your Learning Journey', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3, axis='y')

# Add milestone markers
ax2.annotate('You are\nHERE!', xy=(5, 3.5), xytext=(5, 4.5),
            fontsize=10, ha='center', color=MLRED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=MLRED, lw=2))

# Plot 3: Tools mastered
ax3 = axes[1, 0]

tools = ['Python', 'pandas', 'NumPy', 'matplotlib', 'seaborn', 'scikit-learn',
         'TensorFlow', 'NLTK', 'Streamlit', 'Git/GitHub']
proficiency = [85, 80, 75, 75, 70, 80, 60, 65, 75, 70]

y_pos = np.arange(len(tools))
colors = [MLGREEN if p >= 75 else MLBLUE if p >= 65 else MLORANGE for p in proficiency]

bars = ax3.barh(y_pos, proficiency, color=colors, alpha=0.7)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(tools)
ax3.set_xlabel('Proficiency Level (%)')
ax3.set_xlim(0, 100)
ax3.axvline(x=75, color='gray', linestyle='--', alpha=0.5)
ax3.set_title('Tools Mastered', fontsize=11, fontweight='bold', color=MLPURPLE)

for bar, p in zip(bars, proficiency):
    ax3.text(p + 2, bar.get_y() + bar.get_height()/2,
             f'{p}%', va='center', fontsize=8)

# Plot 4: Key takeaways
ax4 = axes[1, 1]
ax4.axis('off')

takeaways = '''
KEY COURSE TAKEAWAYS

1. DATA IS EVERYTHING
---------------------
"Garbage in, garbage out."
Quality data matters more than
fancy algorithms.


2. START SIMPLE
---------------
"Always start with a baseline."
Simple models often work well
and are easier to explain.


3. EVALUATE HONESTLY
--------------------
"Don't fool yourself."
Use proper validation and
acknowledge limitations.


4. DEPLOYMENT MATTERS
---------------------
"A model in a notebook
 helps nobody."
Real value comes from deployment.


5. ETHICS COUNT
---------------
"With great power..."
Consider impact of your models
on people and society.


6. KEEP LEARNING
----------------
"This is just the beginning."
Data science evolves rapidly.
Stay curious!


WHAT'S NEXT?
------------
- Practice on personal projects
- Explore advanced topics
- Build your portfolio
- Network with practitioners
- Consider certifications


CONGRATULATIONS!
----------------
You've completed a comprehensive
introduction to data science.
Now go build amazing things!
'''

ax4.text(0.02, 0.98, takeaways, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Key Takeaways', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
