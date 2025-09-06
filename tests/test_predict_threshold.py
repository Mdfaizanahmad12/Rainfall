import json
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'rainfall_prediction_using_machine_learning.py'


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return result


def ensure_model():
    model_path = ROOT / 'threshold_model.pkl'
    metrics_path = ROOT / 'threshold_metrics.json'
    if not model_path.exists():
        cmd = [sys.executable, str(SCRIPT), 'train', '--data', 'Rainfall.csv', '--no-gridsearch', '--no-plots', '--model-out', str(model_path), '--metrics-out', str(metrics_path)]
        run(cmd)
    return model_path


def test_predict_with_threshold():
    model_path = ensure_model()
    # Load feature names
    import pickle
    with open(model_path, 'rb') as f:
        obj = pickle.load(f)
    feature_names = obj['feature_names']
    sample = {name: 0 for name in feature_names}
    # Force some fields to non-zero plausible values
    if 'humidity' in sample: sample['humidity'] = 85
    if 'cloud' in sample: sample['cloud'] = 70
    if 'sunshine' in sample: sample['sunshine'] = 2

    cmd = [sys.executable, str(SCRIPT), 'predict', '--model', str(model_path), '--input-json', json.dumps(sample), '--threshold', '0.3']
    result = run(cmd)
    data = json.loads(result.stdout)
    assert 'threshold' in data and data['threshold'] == 0.3
    assert 'prediction' in data
    assert data['label'] in ['yes','no']
