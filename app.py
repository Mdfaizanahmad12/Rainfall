import json
import pickle
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify
import pandas as pd

MODEL_PATH = Path('rainfall_prediction_model.pkl')
METRICS_PATH = Path('training_metrics.json')

app = Flask(__name__)
_model = None
_feature_names = None


def load_model():
    global _model, _feature_names
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError('Model file not found. Train the model first via CLI.')
        with open(MODEL_PATH, 'rb') as f:
            obj = pickle.load(f)
        _model = obj['model']
        _feature_names = obj['feature_names']
    return _model, _feature_names


def build_dataframe(payload: Dict[str, Any]):
    model, feature_names = load_model()
    row = {f: payload.get(f, 0) for f in feature_names}
    return pd.DataFrame([row], columns=feature_names)

def load_feature_importances():
    if METRICS_PATH.exists():
        try:
            data = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
            return data.get('feature_importances') or []
        except Exception:
            return []
    return []


@app.route('/')
def index():
    _, feature_names = load_model()
    return render_template('index.html', features=feature_names)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    model, feature_names = load_model()
    data = request.get_json(force=True)
    threshold = data.get('threshold')
    df = build_dataframe(data)
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = float(model.predict_proba(df)[0][1])
        if threshold is not None:
            pred = 1 if proba >= float(threshold) else 0
        else:
            pred = int(model.predict(df)[0])
    else:
        pred = int(model.predict(df)[0])
    return jsonify({
        'prediction': int(pred),
        'label': 'yes' if pred == 1 else 'no',
        'rainfall_probability': proba,
        'threshold': threshold
    })


@app.route('/api/feature-importances')
def api_feature_importances():
    imps = load_feature_importances()
    return jsonify({'feature_importances': imps})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
