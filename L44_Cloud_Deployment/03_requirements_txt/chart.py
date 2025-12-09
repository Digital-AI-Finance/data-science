"""Requirements.txt - Managing Dependencies"""
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
fig.suptitle('Requirements.txt: Managing Dependencies', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Creating requirements.txt
ax1 = axes[0, 0]
ax1.axis('off')

creating = '''
CREATING REQUIREMENTS.TXT

METHOD 1: MANUAL (RECOMMENDED)
------------------------------
Create file manually with exact versions:

# requirements.txt
streamlit==1.29.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
plotly==5.18.0
matplotlib==3.7.3


METHOD 2: PIP FREEZE
--------------------
# Generate from current environment
pip freeze > requirements.txt

Warning: Includes ALL packages!
You may need to clean it up.


METHOD 3: PIPREQS (BETTER)
--------------------------
# Install pipreqs
pip install pipreqs

# Generate from imports
pipreqs /path/to/project

Only includes packages you actually use!


FINDING VERSIONS:
-----------------
# Check installed version
pip show pandas
# Version: 2.0.3

# Or in Python
import pandas as pd
print(pd.__version__)


VERSION SPECIFIERS:
-------------------
==1.0.0    Exact version (recommended)
>=1.0.0    At least this version
<=1.0.0    At most this version
~=1.0.0    Compatible version
>=1.0,<2.0 Range
'''

ax1.text(0.02, 0.98, creating, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Creating Requirements', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Example requirements
ax2 = axes[0, 1]
ax2.axis('off')

example = '''
EXAMPLE REQUIREMENTS FOR STOCK DASHBOARD

# requirements.txt

# Core Streamlit
streamlit==1.29.0

# Data handling
pandas==2.0.3
numpy==1.24.3

# Machine Learning
scikit-learn==1.3.0
joblib==1.3.2

# Visualization
plotly==5.18.0
matplotlib==3.7.3

# Optional: Deep Learning (increases deploy time!)
# tensorflow==2.14.0

# Optional: NLP
# nltk==3.8.1
# transformers==4.35.0


COMMON PACKAGE NAMES:
---------------------
Be careful with names!

Package          | Correct Name
-----------------|----------------
scikit-learn     | scikit-learn
OpenCV           | opencv-python
PIL              | Pillow
sklearn          | scikit-learn (not sklearn!)
NLTK             | nltk
Keras            | keras or tensorflow


STREAMLIT CLOUD SPECIFIC:
-------------------------
Some packages pre-installed:
- pandas
- numpy
- matplotlib

But ALWAYS include them anyway!
Explicit > implicit.
'''

ax2.text(0.02, 0.98, example, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Example Requirements', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Testing requirements
ax3 = axes[1, 0]
ax3.axis('off')

testing = '''
TESTING REQUIREMENTS LOCALLY

STEP 1: CREATE CLEAN ENVIRONMENT
--------------------------------
# Create new virtual environment
python -m venv test_env

# Activate it
# Windows:
test_env\\Scripts\\activate
# Mac/Linux:
source test_env/bin/activate


STEP 2: INSTALL REQUIREMENTS
----------------------------
pip install -r requirements.txt


STEP 3: RUN YOUR APP
--------------------
streamlit run app.py


STEP 4: FIX ERRORS
------------------
# If ModuleNotFoundError:
# Add missing package to requirements.txt
# Repeat steps 2-3


COMMON ISSUES:
--------------
1. Missing package
   -> Add to requirements.txt

2. Version conflict
   -> Check compatible versions

3. Platform-specific package
   -> Use conditional or remove


VERIFICATION:
-------------
# List installed packages
pip list

# Check specific package
pip show pandas


CLEANUP:
--------
# Deactivate when done
deactivate

# Delete test environment
rmdir /s test_env  # Windows
rm -rf test_env    # Mac/Linux
'''

ax3.text(0.02, 0.98, testing, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Testing Requirements', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Common mistakes
ax4 = axes[1, 1]

mistakes = [
    ('Wrong package name\n(sklearn vs scikit-learn)', MLRED, 'Use pip show to find correct name'),
    ('Missing dependency\n(imported but not listed)', MLRED, 'Use pipreqs to auto-detect'),
    ('Version conflict\n(incompatible packages)', MLORANGE, 'Check compatibility matrix'),
    ('Too many packages\n(pip freeze overkill)', MLORANGE, 'List only what you import'),
    ('No version pinning\n(different on deploy)', MLORANGE, 'Always use == for versions'),
]

y_pos = np.arange(len(mistakes))

for i, (mistake, color, fix) in enumerate(mistakes):
    ax4.add_patch(plt.Rectangle((0, i-0.35), 0.5, 0.7, facecolor=color, alpha=0.2))
    ax4.text(0.02, i, mistake, fontsize=9, va='center')
    ax4.text(0.52, i, fix, fontsize=8, va='center', style='italic')

ax4.set_xlim(0, 1)
ax4.set_ylim(-0.5, len(mistakes)-0.5)
ax4.invert_yaxis()
ax4.axis('off')
ax4.set_title('Common Mistakes & Fixes', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
