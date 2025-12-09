"""Secrets Management - Handling Sensitive Data"""
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
fig.suptitle('Secrets Management: Handling Sensitive Data', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What are secrets
ax1 = axes[0, 0]
ax1.axis('off')

what_secrets = '''
WHAT ARE SECRETS?

DEFINITION:
-----------
Sensitive configuration data that should
NEVER be in your code or public repository.


EXAMPLES OF SECRETS:
--------------------
- API keys (OpenAI, financial data APIs)
- Database credentials
- Authentication tokens
- Private keys
- Passwords


WHY PROTECT SECRETS?
--------------------
1. Security: Prevent unauthorized access
2. Cost: API keys can be abused
3. Compliance: Regulatory requirements
4. Trust: Protect user data


WHAT NOT TO DO:
---------------
# NEVER DO THIS!
api_key = "sk-abc123xyz"  # Hardcoded secret!

# OR THIS!
# Committing secrets.toml to git


WHAT TO DO:
-----------
# Access from st.secrets
api_key = st.secrets["API_KEY"]

# Or environment variables
import os
api_key = os.environ.get("API_KEY")


STREAMLIT APPROACH:
-------------------
1. Local: .streamlit/secrets.toml
2. Cloud: Settings -> Secrets panel
3. Code: st.secrets["key"]
'''

ax1.text(0.02, 0.98, what_secrets, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What Are Secrets?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Local secrets
ax2 = axes[0, 1]
ax2.axis('off')

local_secrets = '''
LOCAL SECRETS SETUP

STEP 1: CREATE SECRETS FILE
---------------------------
# .streamlit/secrets.toml

# Simple key-value
API_KEY = "your-api-key-here"
DATABASE_URL = "postgresql://user:pass@host/db"

# Nested structure
[database]
host = "localhost"
port = 5432
user = "admin"
password = "secret123"

[api]
openai_key = "sk-..."
alpha_vantage_key = "DEMO"


STEP 2: ACCESS IN CODE
----------------------
import streamlit as st

# Simple access
api_key = st.secrets["API_KEY"]

# Nested access
db_host = st.secrets["database"]["host"]
db_pass = st.secrets["database"]["password"]


STEP 3: ADD TO .GITIGNORE
-------------------------
# .gitignore
.streamlit/secrets.toml


IMPORTANT:
----------
secrets.toml is for LOCAL ONLY.
Never commit to git!

For cloud deployment, use the
Streamlit Cloud secrets UI.


FILE STRUCTURE:
---------------
project/
  app.py
  requirements.txt
  .streamlit/
    config.toml     <- Can commit
    secrets.toml    <- DO NOT commit!
  .gitignore        <- Must include secrets.toml
'''

ax2.text(0.02, 0.98, local_secrets, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Local Secrets Setup', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Cloud secrets
ax3 = axes[1, 0]
ax3.axis('off')

cloud_secrets = '''
STREAMLIT CLOUD SECRETS

STEP 1: DEPLOY YOUR APP
-----------------------
First deploy without secrets.
App may show errors - that's OK!


STEP 2: GO TO APP SETTINGS
--------------------------
1. Open Streamlit Cloud dashboard
2. Find your deployed app
3. Click "..." menu
4. Select "Settings"


STEP 3: ADD SECRETS
-------------------
1. Go to "Secrets" section
2. Paste TOML format secrets:

API_KEY = "your-api-key"

[database]
host = "db.example.com"
password = "secret"

3. Click "Save"


STEP 4: REBOOT APP
------------------
App automatically reboots with new secrets.


ACCESSING IN CODE:
------------------
# Same code works locally AND in cloud!
api_key = st.secrets["API_KEY"]


CHECKING IF SECRET EXISTS:
--------------------------
if "API_KEY" in st.secrets:
    api_key = st.secrets["API_KEY"]
else:
    st.error("API key not configured!")
    st.stop()


SECURITY NOTES:
---------------
- Secrets are encrypted at rest
- Only you can see/edit them
- Rotated automatically on reboot
'''

ax3.text(0.02, 0.98, cloud_secrets, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Cloud Secrets Setup', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Security visualization
ax4 = axes[1, 1]
ax4.axis('off')

# Draw security comparison
# Bad practice
ax4.add_patch(plt.Rectangle((0.05, 0.55), 0.4, 0.35, facecolor=MLRED, alpha=0.2))
ax4.text(0.25, 0.85, 'BAD PRACTICE', fontsize=10, ha='center', fontweight='bold', color=MLRED)
ax4.text(0.25, 0.75, 'Hardcoded in code:', fontsize=9, ha='center')
ax4.text(0.25, 0.65, 'api_key = "sk-123"', fontsize=8, ha='center', fontfamily='monospace')
ax4.text(0.25, 0.58, 'Anyone can see!', fontsize=8, ha='center', color=MLRED)

# Good practice
ax4.add_patch(plt.Rectangle((0.55, 0.55), 0.4, 0.35, facecolor=MLGREEN, alpha=0.2))
ax4.text(0.75, 0.85, 'GOOD PRACTICE', fontsize=10, ha='center', fontweight='bold', color=MLGREEN)
ax4.text(0.75, 0.75, 'From secrets:', fontsize=9, ha='center')
ax4.text(0.75, 0.65, 'api_key = st.secrets["KEY"]', fontsize=8, ha='center', fontfamily='monospace')
ax4.text(0.75, 0.58, 'Secure!', fontsize=8, ha='center', color=MLGREEN)

# Workflow
ax4.text(0.5, 0.45, 'SECRETS WORKFLOW', fontsize=10, ha='center', fontweight='bold')

steps = [
    (0.15, 'Local Dev:\nsecrets.toml'),
    (0.4, '.gitignore:\nexclude secrets'),
    (0.65, 'Cloud:\nSecrets UI'),
    (0.9, 'Code:\nst.secrets[]')
]

for x, label in steps:
    ax4.add_patch(plt.Circle((x, 0.25), 0.08, facecolor=MLBLUE, alpha=0.3))
    ax4.text(x, 0.25, label, fontsize=7, ha='center', va='center')

for i in range(len(steps)-1):
    ax4.annotate('', xy=(steps[i+1][0]-0.1, 0.25), xytext=(steps[i][0]+0.1, 0.25),
                arrowprops=dict(arrowstyle='->', color='gray'))

ax4.set_xlim(0, 1)
ax4.set_ylim(0.1, 1)
ax4.set_title('Security Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
