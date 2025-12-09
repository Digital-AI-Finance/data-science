"""Q&A Handling - Answering Questions Confidently"""
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
fig.suptitle('Q&A Handling: Answering Questions Confidently', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Common question types
ax1 = axes[0, 0]
ax1.axis('off')

question_types = '''
COMMON QUESTION TYPES

CLARIFICATION QUESTIONS:
------------------------
"Can you explain how you handled [X]?"
"What do you mean by [term]?"
Strategy: Explain clearly, use examples


METHODOLOGY QUESTIONS:
----------------------
"Why did you choose [model]?"
"How did you handle [issue]?"
Strategy: Justify your decisions


RESULTS QUESTIONS:
------------------
"What does [metric] mean?"
"Why is the accuracy [X]%?"
Strategy: Interpret honestly


LIMITATION QUESTIONS:
---------------------
"What are the weaknesses?"
"What would you do differently?"
Strategy: Be honest, show awareness


EXTENSION QUESTIONS:
--------------------
"What would you add next?"
"How would you improve this?"
Strategy: Show you've thought ahead


TECHNICAL QUESTIONS:
--------------------
"What hyperparameters did you use?"
"How did you prevent overfitting?"
Strategy: Know your implementation


CHALLENGING QUESTIONS:
----------------------
"Isn't this just [simpler approach]?"
"Why not use [other method]?"
Strategy: Acknowledge, explain rationale
'''

ax1.text(0.02, 0.98, question_types, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Common Question Types', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Answer framework
ax2 = axes[0, 1]
ax2.axis('off')

framework = '''
ANSWERING FRAMEWORK

THE 3-STEP APPROACH:
--------------------
1. ACKNOWLEDGE the question
2. ANSWER clearly and briefly
3. CONNECT to your work


EXAMPLE 1:
----------
Q: "Why did you use Random Forest?"

A: "Great question!" [Acknowledge]
   "I chose Random Forest because it
    handles non-linear relationships
    well and doesn't require much
    feature scaling." [Answer]
   "As I showed in my results, it
    outperformed the baseline by 15%."
    [Connect]


EXAMPLE 2:
----------
Q: "What are the limitations?"

A: "I appreciate you asking." [Ack]
   "The main limitations are:
    1. Trained only on US stocks
    2. Doesn't account for news events
    3. Historical patterns may not repeat"
    [Answer]
   "I documented these in my README
    and ethics section." [Connect]


TIMING:
-------
Ideal answer: 30-45 seconds
Maximum: 60 seconds
If longer needed: "I can discuss
more after if interested."


IF YOU DON'T KNOW:
------------------
"That's a great question. I didn't
explore that in this project, but
it would be interesting to
investigate."

NEVER make up an answer!
'''

ax2.text(0.02, 0.98, framework, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Answer Framework', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Prepare for these questions
ax3 = axes[1, 0]

questions = [
    'Why this model?',
    'How handle missing data?',
    'What are limitations?',
    'Next steps?',
    'Why these features?',
    'How prevent overfitting?',
    'Why this metric?',
    'What surprised you?'
]

# Likelihood of being asked (simulated)
likelihood = [90, 70, 85, 80, 75, 60, 65, 50]

y_pos = np.arange(len(questions))
colors = [MLRED if l > 80 else MLORANGE if l > 60 else MLGREEN for l in likelihood]

bars = ax3.barh(y_pos, likelihood, color=colors, alpha=0.7)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(questions, fontsize=9)
ax3.set_xlabel('Likelihood of Being Asked (%)')
ax3.set_xlim(0, 100)
ax3.axvline(x=80, color=MLRED, linestyle='--', alpha=0.5)
ax3.set_title('Prepare for These Questions', fontsize=11, fontweight='bold', color=MLPURPLE)

for bar, l in zip(bars, likelihood):
    ax3.text(l + 2, bar.get_y() + bar.get_height()/2,
             f'{l}%', va='center', fontsize=8)

# Plot 4: Do's and Don'ts
ax4 = axes[1, 1]
ax4.axis('off')

dos_donts = '''
Q&A DO'S AND DON'TS

DO:
---
- Listen to the full question
- Pause before answering
- Keep answers concise
- Admit when you don't know
- Thank the questioner
- Stay calm and professional
- Use your slides as reference
- Connect answers to your work


DON'T:
------
- Interrupt the question
- Get defensive
- Make up answers
- Give 5-minute responses
- Argue with the questioner
- Apologize excessively
- Panic if you don't know
- Dismiss valid criticism


DIFFICULT SITUATIONS:
---------------------
Hostile question:
"Thank you for that perspective.
 Here's my reasoning..."

Don't understand question:
"Could you clarify what you mean
 by [part of question]?"

Know you made a mistake:
"You're right, that could be
 improved. Good catch!"

Question outside scope:
"That's beyond what I explored,
 but it's a great idea for
 future work."


BODY LANGUAGE:
--------------
- Make eye contact
- Stand/sit confidently
- Don't fidget
- Nod while listening
- Smile when appropriate
'''

ax4.text(0.02, 0.98, dos_donts, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title("Q&A Do's and Don'ts", fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
