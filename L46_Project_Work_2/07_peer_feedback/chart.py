"""Peer Feedback - Learning from Each Other"""
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
fig.suptitle('Peer Feedback: Learning from Each Other', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: How to give feedback
ax1 = axes[0, 0]
ax1.axis('off')

giving_feedback = '''
HOW TO GIVE CONSTRUCTIVE FEEDBACK

STRUCTURE (SANDWICH METHOD):
----------------------------
1. Start with something positive
2. Suggest improvements
3. End with encouragement


GOOD FEEDBACK EXAMPLES:
-----------------------
"The visualization is really clear!
One thing that might help is adding
axis labels. Overall great progress!"

"I like how you handled missing data.
Have you considered trying a different
model? The UI looks professional!"

"Great choice of color scheme!
The prediction might be more useful
if you showed confidence scores.
Looking forward to the final version!"


BAD FEEDBACK (AVOID):
---------------------
- "This doesn't work" (not helpful)
- "I don't like it" (not specific)
- "You should have..." (judgmental)
- "My project is better" (comparing)


FEEDBACK CATEGORIES:
--------------------
1. FUNCTIONALITY
   - Does the app work?
   - Are predictions reasonable?

2. USER EXPERIENCE
   - Is it easy to use?
   - Are instructions clear?

3. PRESENTATION
   - Is the code organized?
   - Is documentation complete?

4. TECHNICAL
   - Model choice appropriate?
   - Data handling correct?
'''

ax1.text(0.02, 0.98, giving_feedback, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Giving Feedback', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: How to receive feedback
ax2 = axes[0, 1]
ax2.axis('off')

receiving_feedback = '''
HOW TO RECEIVE FEEDBACK

MINDSET:
--------
- Feedback is a GIFT
- It helps you improve
- Don't take it personally
- Every suggestion is optional


DURING FEEDBACK:
----------------
1. Listen actively
2. Don't interrupt
3. Take notes
4. Ask clarifying questions
5. Thank the person


AFTER FEEDBACK:
---------------
1. Review all notes
2. Prioritize issues:
   - Must fix (critical bugs)
   - Should fix (UX issues)
   - Could fix (nice to have)
3. Make changes
4. Test thoroughly


QUESTIONS TO ASK:
-----------------
"What confused you most?"
"What would you change first?"
"Was anything hard to find?"
"What did you expect to see?"
"Would you use this app?"


WHAT TO DO WITH FEEDBACK:
-------------------------
Category         | Action
-----------------+------------------
Bug/Error        | Fix immediately
Confusion        | Add clarity/help
Feature request  | Consider for v2
Personal taste   | Note but optional


DON'T:
------
- Argue or defend
- Dismiss feedback
- Get discouraged
- Try to fix everything
'''

ax2.text(0.02, 0.98, receiving_feedback, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Receiving Feedback', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Peer review checklist
ax3 = axes[1, 0]

categories = ['Functionality', 'User Experience', 'Documentation', 'Technical Quality', 'Presentation']
weights = [25, 20, 15, 25, 15]
colors = [MLBLUE, MLORANGE, MLGREEN, MLPURPLE, MLRED]

# Create radar chart style visualization
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

values = [4, 3.5, 4, 3, 4]  # Example scores out of 5
values += values[:1]

ax3.set_theta_offset(np.pi / 2)
ax3.set_theta_direction(-1)
ax3 = plt.subplot(2, 2, 3, projection='polar')

ax3.plot(angles, values, 'o-', linewidth=2, color=MLBLUE)
ax3.fill(angles, values, alpha=0.25, color=MLBLUE)
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(categories, fontsize=8)
ax3.set_ylim(0, 5)
ax3.set_title('Peer Review Dimensions', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Feedback form example
ax4 = axes[1, 1]
ax4.axis('off')

form = '''
PEER REVIEW FORM

Project: _________________________
Reviewer: ________________________
Date: ____________________________


1. FUNCTIONALITY (1-5): ___
   [ ] App loads successfully
   [ ] Core features work
   [ ] No critical errors
   Comments: ___________________


2. USER EXPERIENCE (1-5): ___
   [ ] Easy to navigate
   [ ] Instructions clear
   [ ] Results understandable
   Comments: ___________________


3. DOCUMENTATION (1-5): ___
   [ ] README complete
   [ ] Code commented
   [ ] Data source cited
   Comments: ___________________


4. TECHNICAL (1-5): ___
   [ ] Appropriate models
   [ ] Proper evaluation
   [ ] Clean code
   Comments: ___________________


5. OVERALL IMPRESSION:
   What I liked most:
   _____________________________

   What could improve:
   _____________________________

   One specific suggestion:
   _____________________________


SCORING GUIDE:
5 = Excellent, 4 = Good, 3 = OK
2 = Needs work, 1 = Incomplete
'''

ax4.text(0.02, 0.98, form, transform=ax4.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Feedback Form', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
