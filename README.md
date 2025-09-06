# 🌧 Rainfall Prediction App

Streamlit-based machine learning app that trains (or auto-trains) a RandomForest classifier to predict rainfall (yes/no) from weather features. Includes:

- Auto-training if model pickle missing or incompatible (sklearn version aware)
- On-demand Force Retrain button
- Threshold-based probability decision
- Feature importance table
- Synthetic dataset fallback when `Rainfall.csv` absent
- CLI script for offline training / prediction

## Project Structure

Key files:

- `streamlit_app.py` – Main UI (self-contained training logic)
- `rainfall_prediction_using_machine_learning.py` – CLI (train / predict)
- `Rainfall.csv` – Optional dataset (synthetic generated if missing)
- `rainfall_prediction_model.pkl` / `training_metrics.json` – Model + metrics artifacts
- `tests/` – Pytest unit tests

## Local Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py --server.port 8501
```

Open: http://localhost:8501

Run tests:
```powershell
pytest -q
```

Retrain manually (outside Streamlit):
```powershell
python rainfall_prediction_using_machine_learning.py train --data Rainfall.csv --no-gridsearch --no-plots
```

## CLI Usage

Train:
```powershell
python rainfall_prediction_using_machine_learning.py train --data Rainfall.csv --model-out rainfall_prediction_model.pkl --metrics-out training_metrics.json --no-gridsearch --no-plots
```

Predict (JSON input):
```powershell
python rainfall_prediction_using_machine_learning.py predict --model rainfall_prediction_model.pkl --input sample_input.json
```

## Docker Deployment

Build image:
```powershell
docker build -t rainfall-app .
```

Run container:
```powershell
docker run -p 8501:8501 --name rainfall rainfall-app
```

Visit: http://localhost:8501

## Streamlit Cloud Deployment

1. Push repo to GitHub.
2. On https://share.streamlit.io create new app selecting this repo & branch `main`.
3. Entry point: `streamlit_app.py`
4. (Optional) Add secret config for future API keys.
5. App auto-installs `requirements.txt`; if an old model pickle causes mismatch it auto-retrains.

## Environment Variables

- `RAIN_RETRAIN_ALWAYS=1` – Force retrain on each cold start (not recommended for production unless data changes frequently).

## Healthcheck (Docker)

Dockerfile adds a basic healthcheck hitting Streamlit internal endpoint.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Blank page | Check terminal for syntax errors; ensure no stray raw commands in `streamlit_app.py`. |
| ModuleNotFoundError streamlit.cli | Reinstall Streamlit inside venv (`pip install --force-reinstall streamlit`). |
| Sklearn version mismatch warning | App auto-retrains; ensure pinned version in `requirements.txt`. |
| Docker build slow | Remove pre-train RUN layer or add `--build-arg` to skip. |

## License

MIT (adjust as needed).

