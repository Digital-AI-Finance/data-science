"""Model Integration - ML Models in Streamlit"""
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
fig.suptitle('Model Integration: ML Models in Streamlit', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Loading models
ax1 = axes[0, 0]
ax1.axis('off')

loading_models = '''
LOADING ML MODELS

LOAD WITH CACHING:
------------------
import streamlit as st
import joblib


@st.cache_resource
def load_model():
    \"\"\"Load model once, share across sessions.\"\"\"
    return joblib.load('models/stock_classifier.joblib')


@st.cache_resource
def load_features():
    \"\"\"Load expected feature names.\"\"\"
    return joblib.load('models/features.joblib')


# Load at startup
model = load_model()
features = load_features()


WHY @st.cache_resource?
-----------------------
- Models are large objects
- Don't want to reload on every interaction
- Shared across all users
- Persists until app restarts


KERAS/TENSORFLOW:
-----------------
import tensorflow as tf

@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('model.h5')


ERROR HANDLING:
---------------
@st.cache_resource
def load_model_safe():
    try:
        model = joblib.load('model.joblib')
        st.success("Model loaded!")
        return model
    except FileNotFoundError:
        st.error("Model file not found!")
        return None
'''

ax1.text(0.02, 0.98, loading_models, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Loading Models', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Prediction interface
ax2 = axes[0, 1]
ax2.axis('off')

prediction_ui = '''
PREDICTION INTERFACE

SIMPLE PREDICTION FORM:
-----------------------
import streamlit as st
import numpy as np

st.title("Stock Direction Predictor")

# Input widgets
price = st.number_input("Current Price", min_value=0.0)
volume = st.number_input("Volume", min_value=0)
momentum = st.slider("Momentum", -1.0, 1.0, 0.0)

# Predict button
if st.button("Predict"):
    # Prepare features
    X = np.array([[price, volume, momentum]])

    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    # Display result
    if prediction == 1:
        st.success(f"Prediction: BUY")
    else:
        st.error(f"Prediction: SELL")

    st.write(f"Confidence: {max(probability):.1%}")


WITH COLUMNS:
-------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Features")
    price = st.number_input("Price", 0.0, 1000.0, 150.0)
    volume = st.number_input("Volume", 0, 10000000, 1000000)

with col2:
    st.subheader("Prediction")
    if st.button("Run"):
        pred = model.predict([[price, volume]])[0]
        st.metric("Signal", "BUY" if pred == 1 else "SELL")


BATCH UPLOAD:
-------------
uploaded = st.file_uploader("Upload CSV for batch prediction")
if uploaded:
    df = pd.read_csv(uploaded)
    df['prediction'] = model.predict(df[features])
    st.dataframe(df)
'''

ax2.text(0.02, 0.98, prediction_ui, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Prediction Interface', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Complete prediction app
ax3 = axes[1, 0]
ax3.axis('off')

complete_app = '''
COMPLETE PREDICTION APP

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Page config
st.set_page_config(page_title="Stock Predictor", layout="wide")


# Load model (cached)
@st.cache_resource
def load_model():
    return joblib.load('models/stock_classifier.joblib')

model = load_model()


# Sidebar - Model info
st.sidebar.title("Model Info")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Accuracy: 89%")
st.sidebar.write("Last trained: 2024-01-15")


# Main content
st.title("Stock Direction Predictor")

# Input section
st.header("Enter Stock Features")

col1, col2, col3 = st.columns(3)
with col1:
    price = st.number_input("Price ($)", 0.0, 1000.0, 150.0)
with col2:
    volume = st.number_input("Volume (M)", 0.0, 100.0, 1.0)
with col3:
    momentum = st.slider("Momentum", -1.0, 1.0, 0.0)


# Prediction section
if st.button("Make Prediction", type="primary"):
    X = np.array([[price, volume * 1e6, momentum]])

    with st.spinner("Predicting..."):
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

    # Display results
    st.header("Prediction Result")
    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("Signal: BUY", icon="arrow_upward")
        else:
            st.error("Signal: SELL", icon="arrow_downward")

    with col2:
        st.metric("Confidence", f"{max(proba):.1%}")
'''

ax3.text(0.02, 0.98, complete_app, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Complete App', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: App mockup
ax4 = axes[1, 1]
ax4.axis('off')

# Draw app mockup
# Header
ax4.add_patch(plt.Rectangle((0.05, 0.85), 0.9, 0.1, facecolor=MLPURPLE, alpha=0.3))
ax4.text(0.5, 0.9, 'Stock Direction Predictor', fontsize=12, ha='center', fontweight='bold')

# Input section
ax4.text(0.1, 0.78, 'Input Features', fontsize=10, fontweight='bold')
inputs = [('Price: $150.00', 0.15), ('Volume: 1.0M', 0.4), ('Momentum: 0.02', 0.65)]
for label, x in inputs:
    ax4.add_patch(plt.Rectangle((x, 0.65), 0.2, 0.08, facecolor='white', edgecolor='gray'))
    ax4.text(x+0.1, 0.69, label, fontsize=8, ha='center')

# Predict button
ax4.add_patch(plt.Rectangle((0.35, 0.5), 0.3, 0.08, facecolor=MLBLUE, alpha=0.8))
ax4.text(0.5, 0.54, 'Make Prediction', fontsize=10, ha='center', color='white')

# Result section
ax4.text(0.1, 0.4, 'Prediction Result', fontsize=10, fontweight='bold')

# Buy signal
ax4.add_patch(plt.Rectangle((0.15, 0.2), 0.3, 0.15, facecolor=MLGREEN, alpha=0.3))
ax4.text(0.3, 0.27, 'BUY', fontsize=14, ha='center', fontweight='bold', color=MLGREEN)

# Confidence
ax4.add_patch(plt.Rectangle((0.55, 0.2), 0.3, 0.15, facecolor=MLLAVENDER, alpha=0.5))
ax4.text(0.7, 0.3, 'Confidence', fontsize=8, ha='center')
ax4.text(0.7, 0.24, '87.5%', fontsize=14, ha='center', fontweight='bold')

ax4.set_xlim(0, 1)
ax4.set_ylim(0.1, 1)
ax4.set_title('App Preview', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
