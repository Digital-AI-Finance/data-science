"""Merge Troubleshooting - Common issues and solutions"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

# Standard matplotlib configuration
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

# Course colors
MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(8, 9.5, 'Merge Troubleshooting Guide', ha='center',
        fontsize=16, fontweight='bold', color=MLPURPLE)

# Issues
issues = [
    {
        'problem': 'Duplicate Rows After Merge',
        'symptom': 'Result has more rows than expected',
        'cause': 'Many-to-many relationship (duplicates in key column)',
        'solution': "1. Check for duplicates: df['key'].duplicated().sum()\n2. Remove duplicates before merge: df.drop_duplicates(subset=['key'])",
        'color': MLRED
    },
    {
        'problem': 'Missing Rows After Merge',
        'symptom': 'Result has fewer rows than expected',
        'cause': "Key values don't match (spelling, case, whitespace)",
        'solution': "1. Check key overlap: set(df1['key']) & set(df2['key'])\n2. Clean keys: df['key'].str.strip().str.upper()\n3. Use how='outer' to keep all rows",
        'color': MLORANGE
    },
    {
        'problem': 'Column Name Conflicts',
        'symptom': "Columns with '_x' and '_y' suffixes",
        'cause': 'Both DataFrames have columns with same name',
        'solution': "1. Rename before: df.rename(columns={'col': 'new_col'})\n2. Use suffixes: merge(df1, df2, suffixes=('_left', '_right'))",
        'color': MLBLUE
    },
    {
        'problem': 'Data Type Mismatch',
        'symptom': 'No matches found despite matching values',
        'cause': 'Key columns have different dtypes (int vs str)',
        'solution': "1. Check types: df1['key'].dtype vs df2['key'].dtype\n2. Convert: df['key'] = df['key'].astype(str)",
        'color': MLGREEN
    },
]

y_start = 8.5
for issue in issues:
    # Issue box
    box = FancyBboxPatch((0.3, y_start - 1.8), 15.4, 2, boxstyle="round,pad=0.1",
                         edgecolor=issue['color'], facecolor='white', linewidth=2)
    ax.add_patch(box)

    # Problem title
    ax.text(0.6, y_start - 0.1, issue['problem'], fontsize=11,
            fontweight='bold', color=issue['color'])

    # Symptom
    ax.text(0.6, y_start - 0.5, f"Symptom: {issue['symptom']}", fontsize=9, color='gray')

    # Cause
    ax.text(0.6, y_start - 0.9, f"Cause: {issue['cause']}", fontsize=9, color='black')

    # Solution
    solution_lines = issue['solution'].split('\n')
    for i, line in enumerate(solution_lines):
        ax.text(0.6, y_start - 1.3 - i*0.3, line, fontsize=8, family='monospace', color=issue['color'])

    y_start -= 2.2

# Quick checks box
check_box = FancyBboxPatch((0.5, 0.2), 15, 1, boxstyle="round,pad=0.1",
                           edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.3)
ax.add_patch(check_box)
ax.text(8, 0.9, 'Pre-Merge Checklist:', ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)
checks = [
    "1. df1['key'].dtype == df2['key'].dtype",
    "2. df1['key'].duplicated().sum() == 0",
    "3. len(set(df1['key']) & set(df2['key'])) > 0",
]
ax.text(8, 0.4, '   |   '.join(checks), ha='center', fontsize=8, family='monospace', color='black')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
