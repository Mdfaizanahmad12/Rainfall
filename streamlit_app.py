import json
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import streamlit as st
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = Path('rainfall_prediction_model.pkl')
METRICS_PATH = Path('training_metrics.json')
DATA_PATH = Path('Rainfall.csv')
SKLEARN_VER = sklearn.__version__

def ensure_dataset(min_rows: int = 60):
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            if len(df) >= 20 and 'rainfall' in df.columns:
                return
        except Exception:
            pass
    import random, math
    random.seed(42)
    rows = []
    for i in range(min_rows):
        pressure = 1000 + random.random()*30
        maxt = 15 + random.random()*15
        temp = maxt - random.random()*4
        mint = maxt - random.random()*8
        dew = temp - random.random()*3
        hum = int(60 + math.sin(i/15)*15 + random.random()*15)
        hum = max(40, min(100, hum))
        cloud = random.randint(0,100)
        sunshine = max(0, 12 - cloud/15 - random.random()*3)
        windspeed = 5 + random.random()*18
        winddirection = random.choice(['N','S','E','W'])
        rain = 'yes' if (hum>70 and cloud>50 and sunshine<5 and random.random()<0.75) else 'no'
        rows.append({
            'pressure': pressure,
            'maxtemp': maxt,
            'temparature': temp,
            'mintemp': mint,
            'dewpoint': dew,
            'humidity': hum,
            'cloud': cloud,
            'sunshine': sunshine,
            'windspeed': windspeed,
            'winddirection': winddirection,
            'rainfall': rain,
            'day': i+1
        })
    pd.DataFrame(rows).to_csv(DATA_PATH, index=False)

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    if 'day' in df.columns:
        df = df.drop(columns=['day'])
    df.columns = df.columns.str.strip()
    df['rainfall'] = df['rainfall'].map({'yes':1,'no':0})
    cat_cols = [c for c in df.columns if df[c].dtype=='object' and c!='rainfall']
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = df.drop(columns=['rainfall'])
    y = df['rainfall']
    return X, y, X.columns.tolist()

def train_model(fast: bool = True):
    ensure_dataset()
    raw = pd.read_csv(DATA_PATH)
    X, y, feats = preprocess(raw)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=150 if fast else 300, max_features='sqrt', random_state=42)
    model.fit(X_train, y_train)
    acc = float(accuracy_score(y_test, model.predict(X_test)))
    feat_imps = None
    if hasattr(model, 'feature_importances_'):
        feat_imps = [
            {"feature": f, "importance": float(i)}
            for f,i in sorted(zip(feats, model.feature_importances_), key=lambda z: z[1], reverse=True)
        ]
    metrics = {
        'test_accuracy': acc,
        'feature_importances': feat_imps,
        'n_features': len(feats),
        'sklearn_version': SKLEARN_VER
    }
    with open(MODEL_PATH,'wb') as f:
        pickle.dump({'model': model, 'feature_names': feats}, f)
    with open(METRICS_PATH,'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return model, feats, metrics

def _attempt_retrain():
    try:
        train_model(fast=True)
        return True
    except Exception as e:
        st.error(f'Retrain failed: {e}')
        return False

@st.cache_resource(show_spinner=False)
def load_model():
    force = os.getenv('RAIN_RETRAIN_ALWAYS') == '1'
    if force:
        st.info('RAIN_RETRAIN_ALWAYS=1 -> forcing retrain')
        _attempt_retrain()
    if not MODEL_PATH.exists():
        st.warning('Model not found. Training now...')
        _attempt_retrain()
    try:
        with open(MODEL_PATH,'rb') as f:
            obj = pickle.load(f)
        model, feats = obj['model'], obj['feature_names']
    except Exception:
        st.warning('Corrupt/old pickle. Retraining...')
        _attempt_retrain()
        with open(MODEL_PATH,'rb') as f:
            o2 = pickle.load(f)
        model, feats = o2['model'], o2['feature_names']
    # Version mismatch check
    try:
        if METRICS_PATH.exists():
            meta = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
            sv = meta.get('sklearn_version')
            if sv and sv != SKLEARN_VER:
                st.warning(f'Sklearn mismatch (saved {sv} vs runtime {SKLEARN_VER}). Retraining...')
                _attempt_retrain()
                with open(MODEL_PATH,'rb') as f:
                    obj3 = pickle.load(f)
                model, feats = obj3['model'], obj3['feature_names']
    except Exception:
        pass
    return model, feats

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
st.caption(f'RandomForest single inference • sklearn {SKLEARN_VER}')

# Immediate placeholder so something renders even if training takes time
init_box = st.empty()
init_box.info('Initializing / loading model ...')

with st.sidebar:
    st.header('Model')
    try:
        with st.spinner('Loading or training model...'):
            model, feature_names = load_model()
    except Exception as e:
        st.error(f'Fatal error loading model: {e}')
        st.stop()
    init_box.empty()
    metrics = load_metrics()
    if metrics:
        col1, col2 = st.columns(2)
        col1.metric('Accuracy', f"{metrics.get('test_accuracy', 0):.3f}")
        col2.metric('Features', metrics.get('n_features', len(feature_names)))
    if st.button('Force Retrain'):
        _attempt_retrain()
        st.experimental_rerun()
    threshold = st.number_input('Decision Threshold (optional)', min_value=0.0, max_value=1.0, step=0.01, value=None, format='%0.2f')
    st.markdown('---')
    st.write('Enter features then Predict.')

st.subheader('Input Features')
cols_per_row = 4
inputs = {}
for i, feat in enumerate(feature_names):
    col = st.columns(cols_per_row)[i % cols_per_row]
    inputs[feat] = col.number_input(feat, value=0.0, format='%0.4f')

if st.button('Predict', type='primary'):
    df = build_dataframe(inputs, feature_names)
    if hasattr(model, 'predict_proba'):
        proba = float(model.predict_proba(df)[0][1])
        pred = 1 if (threshold is not None and proba >= threshold) else int(model.predict(df)[0])
    else:
        proba = None
        pred = int(model.predict(df)[0])
    label = 'yes' if pred == 1 else 'no'
    st.success(f'Prediction: {label} (class={pred})')
    if proba is not None:
        st.progress(min(1.0, proba), text=f'Probability: {proba:.3f}')
        st.caption(f'Threshold: {threshold if threshold is not None else "model default"}')

metrics = load_metrics()
if metrics and metrics.get('feature_importances'):
    st.subheader('Feature Importances')
    fi_df = pd.DataFrame(metrics['feature_importances'])
    st.dataframe(fi_df, use_container_width=True)

st.markdown('---')
st.caption('Auto-trains if model missing or incompatible. Use Force Retrain to refresh.')

# Command to run the training outside of Streamlit (for reference)
# !python rainfall_prediction_using_machine_learning.py train --data Rainfall.csv --no-gridsearch --no-plots --model-out rainfall_prediction_model.pkl --metrics-out training_metrics.json

# To run the Streamlit app, use the following command in the terminal:
# streamlit run streamlit_app.py --server.port 8501

# (Removed stray raw pip command that caused a syntax error.)
