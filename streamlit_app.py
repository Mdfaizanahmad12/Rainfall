import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import streamlit as st

MODEL_PATH = Path('rainfall_prediction_model.pkl')
METRICS_PATH = Path('training_metrics.json')

def _attempt_retrain() -> bool:
    """Retrain model in-place if CSV & training script exist.
    Returns True on success, False otherwise."""
    csv = Path('Rainfall.csv')
    trainer = Path('rainfall_prediction_using_machine_learning.py')
    if not csv.exists() or not trainer.exists():
        return False
    cmd = [sys.executable, str(trainer), 'train', '--data', str(csv), '--no-gridsearch', '--no-plots']
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except Exception as e:  # pragma: no cover - defensive
        st.error(f"Auto-retrain failed: {e}")
        return False

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        st.warning('Model file missing; attempting automatic retrain...')
        if not _attempt_retrain():
            st.error('Model not found and retrain failed. Provide rainfall_prediction_model.pkl or Rainfall.csv.')
            st.stop()
    try:
        with open(MODEL_PATH,'rb') as f:
            obj = pickle.load(f)
        return obj['model'], obj['feature_names']
    except Exception as e:  # Likely version mismatch / incompatible pickle
        st.warning(f"Model load failed ({e.__class__.__name__}: {e}). Attempting retrain with current environment...")
        if _attempt_retrain():
            with open(MODEL_PATH,'rb') as f:
                obj = pickle.load(f)
            st.success('Model retrained successfully in current environment.')
            return obj['model'], obj['feature_names']
        st.error('Could not load or retrain model. See logs.')
        st.stop()

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
st.caption('If the original pickle is incompatible, the app auto-retrains using Rainfall.csv.')
