"""Demo Tips - Making Your Live Demo Successful"""
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
fig.suptitle('Demo Tips: Making Your Live Demo Successful', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Demo preparation
ax1 = axes[0, 0]
ax1.axis('off')

preparation = '''
DEMO PREPARATION CHECKLIST

BEFORE PRESENTATION DAY:
------------------------
[ ] App deployed and tested
[ ] URL bookmarked in browser
[ ] Local backup ready
[ ] Screenshots as last resort
[ ] Know exact demo steps
[ ] Practiced 3+ times


MORNING OF PRESENTATION:
------------------------
[ ] Test deployed app works
[ ] Open browser tab with app
[ ] Have slides and app ready
[ ] Clear browser cache
[ ] Close unnecessary apps
[ ] Silence phone notifications


RIGHT BEFORE YOUR TURN:
-----------------------
[ ] App tab loaded and ready
[ ] Slides open on correct slide
[ ] Deep breath, stay calm
[ ] Remember: you know this!


DEMO ENVIRONMENT:
-----------------
Browser: Use Chrome or Edge
Tabs: Only presentation + app
Zoom: 100% or larger for visibility
Bookmarks: App URL clearly visible
Desktop: Clean, no distractions


THE GOLDEN RULE:
----------------
"Never show anything live that
 you haven't tested 10 times."


BACKUP HIERARCHY:
-----------------
1. Deployed cloud app
2. Local Streamlit app
3. Screenshots in slides
4. Verbal description
'''

ax1.text(0.02, 0.98, preparation, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Demo Preparation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Demo script
ax2 = axes[0, 1]
ax2.axis('off')

script = '''
DEMO SCRIPT TEMPLATE (2 MINUTES)

TRANSITION (10 sec):
--------------------
"Now let me show you the app
 in action."
[Switch to browser]


OVERVIEW (15 sec):
------------------
"This is my Stock Analysis Dashboard.
 On the left, you can select options.
 The main area shows the analysis."


INPUT (20 sec):
---------------
"Let me select Apple stock..."
[Click dropdown]
"And choose a 30-day prediction window..."
[Adjust slider]


ACTION (20 sec):
----------------
"When I click Analyze..."
[Click button]
[Wait for results]


RESULTS (30 sec):
-----------------
"The model predicts [result].
 You can see the confidence is [X]%.
 This chart shows [explanation]."


FEATURE (15 sec):
-----------------
"One nice feature is [feature].
 This helps users [benefit]."


WRAP-UP (10 sec):
-----------------
"That's the app in action.
 Let me go back to my slides."
[Return to presentation]


TOTAL: ~2 minutes


KEY PHRASES:
------------
"As you can see..."
"Notice how..."
"This shows that..."
"The key insight is..."
'''

ax2.text(0.02, 0.98, script, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Demo Script', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: What can go wrong
ax3 = axes[1, 0]

problems = ['App not loading', 'Slow response', 'Error message', 'Wrong result',
            'Screen sharing', 'Lost connection']
likelihood = [15, 35, 20, 10, 30, 10]
severity = [90, 40, 60, 30, 70, 85]

colors = [MLRED if s > 70 else MLORANGE if s > 40 else MLGREEN for s in severity]

ax3.scatter(likelihood, severity, s=200, c=colors, alpha=0.7, edgecolors='black')

for i, prob in enumerate(problems):
    ax3.annotate(prob, (likelihood[i], severity[i]), textcoords='offset points',
                xytext=(5, 5), fontsize=8)

ax3.set_xlabel('Likelihood (%)')
ax3.set_ylabel('Severity (impact)')
ax3.set_xlim(0, 50)
ax3.set_ylim(0, 100)
ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
ax3.axvline(x=25, color='gray', linestyle='--', alpha=0.3)
ax3.set_title('Demo Risk Matrix', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.grid(alpha=0.3)

# Quadrant labels
ax3.text(37, 75, 'HIGH\nRISK', fontsize=8, ha='center', color=MLRED, fontweight='bold')
ax3.text(12, 25, 'LOW\nRISK', fontsize=8, ha='center', color=MLGREEN, fontweight='bold')

# Plot 4: Recovery strategies
ax4 = axes[1, 1]
ax4.axis('off')

recovery = '''
DEMO RECOVERY STRATEGIES

IF APP DOESN'T LOAD:
--------------------
Say: "Let me refresh the page."
[Wait 5 seconds]
If still fails:
"I'll switch to my local version."
[Open local Streamlit]
If that fails:
"Let me show you screenshots."


IF APP IS SLOW:
---------------
Say: "The model is processing..."
[Talk about what's happening]
"It's analyzing [X] data points."
[If still waiting after 15 sec]
"Let me show you the typical result."


IF YOU GET AN ERROR:
--------------------
Don't panic! Say:
"Interesting, let me try that again."
[Try different input]
If still failing:
"This sometimes happens with [edge case]."
"Let me show a working example."


IF RESULT LOOKS WRONG:
----------------------
Say: "That's an interesting result."
"This might be because [reason]."
"Let me try another example."


UNIVERSAL RECOVERY PHRASES:
---------------------------
"Let me try that again..."
"Here's an alternative..."
"This demonstrates that..."
"As I mentioned in my limitations..."


MOST IMPORTANT:
---------------
Stay calm. Keep talking.
The audience understands that
live demos can be unpredictable!
'''

ax4.text(0.02, 0.98, recovery, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Recovery Strategies', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
