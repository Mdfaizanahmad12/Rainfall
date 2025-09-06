import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'rainfall_prediction_using_machine_learning.py'


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return result


def ensure_dataset():
    csv_path = ROOT / 'Rainfall.csv'
    if not csv_path.exists():
        # minimal synthetic dataset
        import pandas as pd
        import random
        random.seed(0)
        rows = []
        for i in range(40):
            rows.append({
                'pressure': 1000 + random.random()*30,
                'maxtemp': 20 + random.random()*5,
                'temparature': 19 + random.random()*5,
                'mintemp': 15 + random.random()*5,
                'dewpoint': 10 + random.random()*3,
                'humidity': 60 + random.randint(0,30),
                'cloud': random.randint(0,100),
                'sunshine': random.random()*10,
                'windspeed': 5 + random.random()*15,
                'winddirection': random.choice(['N','S','E','W']),
                'rainfall': 'yes' if random.random() < 0.5 else 'no',
                'day': i+1
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False)


def test_train_and_predict():
    ensure_dataset()
    model_path = ROOT / 'test_model.pkl'
    metrics_path = ROOT / 'test_metrics.json'

    run([sys.executable, str(SCRIPT), 'train', '--data', 'Rainfall.csv', '--no-gridsearch', '--no-plots', '--model-out', str(model_path), '--metrics-out', str(metrics_path)])

    assert model_path.exists(), 'Model pickle not created'
    assert metrics_path.exists(), 'Metrics JSON not created'

    # Load metrics JSON and assert basic keys
    metrics = json.loads(metrics_path.read_text())
    for key in ['test_accuracy', 'confusion_matrix', 'classification_report']:
        assert key in metrics, f'Missing metrics key: {key}'

    # Build a feature json for prediction using stored feature names from pickle
    import pickle
    with open(model_path,'rb') as f:
        obj = pickle.load(f)
    feature_names = obj['feature_names']
    sample = {name: 0 for name in feature_names}

    # Predict
    result = run([sys.executable, str(SCRIPT), 'predict', '--model', str(model_path), '--input-json', json.dumps(sample)])
    out = result.stdout.strip()
    assert 'prediction' in out.lower(), 'Prediction output missing'
