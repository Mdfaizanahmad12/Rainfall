import json
import pickle
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import streamlit as st

MODEL_PATH = Path('rainfall_prediction_model.pkl')
METRICS_PATH = Path('training_metrics.json')

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        st.error('Model not found. Train first using the CLI command:')
        st.code('python rainfall_prediction_using_machine_learning.py train --data Rainfall.csv --no-gridsearch --no-plots')
        st.stop()
    with open(MODEL_PATH,'rb') as f:
        obj = pickle.load(f)
    return obj['model'], obj['feature_names']

@st.cache_data(show_spinner=False)
def load_metrics():
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}

def build_dataframe(payload: Dict[str, Any], feature_names):
    row = {f: payload.get(f, 0) for f in feature_names}
    return pd.DataFrame([row], columns=feature_names)

st.set_page_config(page_title='Rainfall Prediction', page_icon='🌧', layout='wide')

st.title('🌧 Rainfall Prediction')
st.caption('RandomForest classifier • Single sample inference with optional threshold')

with st.sidebar:
    st.header('Model')
    model, feature_names = load_model()
    metrics = load_metrics()
    if metrics:
        col1, col2 = st.columns(2)
        col1.metric('Test Accuracy', f"{metrics.get('test_accuracy', 0):.3f}")
        if metrics.get('cv_mean') is not None:
            col2.metric('CV Mean', f"{metrics['cv_mean']:.3f}")
        if metrics.get('best_params'):
            with st.expander('Best Params'):
                st.json(metrics['best_params'])
    else:
        st.info('Train the model to view metrics.')

    threshold = st.number_input('Decision Threshold (optional)', min_value=0.0, max_value=1.0, step=0.01, value=None, format='%0.2f')
    st.markdown('---')
    st.write('Fill the features and press Predict.')

st.subheader('Input Features')
cols_per_row = 4
inputs = {}

for i, feat in enumerate(feature_names):
    col = st.columns(cols_per_row)[i % cols_per_row]
    inputs[feat] = col.number_input(feat, value=0.0, format='%0.4f')

if st.button('Predict', type='primary'):
    df = build_dataframe(inputs, feature_names)
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = float(model.predict_proba(df)[0][1])
        if threshold is not None:
            pred = 1 if proba >= threshold else 0
        else:
            pred = int(model.predict(df)[0])
    else:
        pred = int(model.predict(df)[0])

    label = 'yes' if pred == 1 else 'no'
    st.success(f'Prediction: {label} (class={pred})')
    if proba is not None:
        st.progress(min(1.0, proba), text=f"Probability: {proba:.3f}")
        st.write(f"Threshold: {threshold if threshold is not None else 'model default'}")

if metrics and metrics.get('feature_importances'):
    st.subheader('Feature Importances')
    fi_df = pd.DataFrame(metrics['feature_importances'])
    st.dataframe(fi_df, use_container_width=True)

st.markdown('---')
st.caption('Deploy via: streamlit run streamlit_app.py')
