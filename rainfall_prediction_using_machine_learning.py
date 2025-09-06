# -*- coding: utf-8 -*-
"""Rainfall Prediction CLI (Train / Predict / Streamlit)

Features:
  * Train a RandomForest model (balanced by downsampling) with optional GridSearch
  * Fast mode & ability to skip grid search entirely for quick iterations
  * Saves trained model (pickle) with feature names
  * Exports metrics to JSON (train/test performance, CV, confusion matrix)
  * Predict subcommand to score a single sample from JSON string or file
  * Optional plots (disabled by default in many CI contexts)
  * Lower verbosity by default; opt-in verbose logs via --verbose
  * Streamlit app for interactive prediction and EDA

Examples (PowerShell):
  # Fast training (reduced param grid, no plots)
  python rainfall_prediction_using_machine_learning.py train --data Rainfall.csv --fast --no-plots

  # Minimal quickest training (no grid search at all)
  python rainfall_prediction_using_machine_learning.py train --data Rainfall.csv --no-gridsearch --no-plots

  # Predict using a JSON inline sample
  python rainfall_prediction_using_machine_learning.py predict --model rainfall_prediction_model.pkl --input-json '{"pressure":1015.0, "dewpoint":7, "humidity":80, "cloud":60, "sunshine":3, "windspeed":12, "winddirection_W":1}'

  # Run Streamlit app
  streamlit run rainfall_prediction_using_machine_learning.py --server.port 8501

Outputs:
  rainfall_prediction_model.pkl   - Pickled dict: {model, feature_names}
  training_metrics.json           - Metrics artifact
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Colab compatibility (skip if not available locally)
# ---------------------------------------------------------------------------
try:  # noqa: SIM105
    from google.colab import drive  # type: ignore
    COLAB = True
except ModuleNotFoundError:  # Not running inside Colab
    COLAB = False

if COLAB:  # Only attempt to mount in Colab
    try:
        drive.mount('/content/drive')
    except Exception as e:  # pragma: no cover - defensive
        print(f"[WARN] Could not mount Google Drive: {e}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import streamlit as st

LOG = logging.getLogger("rainfall")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Rainfall prediction training & inference CLI")
  parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
  sub = parser.add_subparsers(dest="command", required=False)

  # Train subcommand (default if no command given)
  p_train = sub.add_parser("train", help="Train a model")
  p_train.add_argument("--data", default="Rainfall.csv", help="Path to dataset CSV")
  p_train.add_argument("--fast", action="store_true", help="Smaller hyperparameter grid for speed")
  p_train.add_argument("--no-gridsearch", action="store_true", help="Skip GridSearch (use baseline model)")
  p_train.add_argument("--no-plots", action="store_true", help="Disable plotting")
  p_train.add_argument("--model-out", default="rainfall_prediction_model.pkl", help="Model pickle output path")
  p_train.add_argument("--metrics-out", default="training_metrics.json", help="Metrics JSON output path")

  # Predict subcommand
  p_pred = sub.add_parser("predict", help="Predict a single sample from JSON input")
  p_pred.add_argument("--model", default="rainfall_prediction_model.pkl", help="Path to model pickle")
  p_pred.add_argument("--input-json", required=True, help="Inline JSON string or path to JSON file with feature values")
  p_pred.add_argument("--threshold", type=float, default=None, help="Optional probability threshold for positive class (rainfall=yes). If omitted, model default decision is used.")

  return parser


def parse_args():
  parser = build_parser()
  args = parser.parse_args()
  # Backward compatibility: if no subcommand provided, assume train
  if args.command is None:
    args.command = "train"
  return args


def configure_logging(verbose: bool):
  level = logging.DEBUG if verbose else logging.INFO
  logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def load_dataset(path: str) -> pd.DataFrame:
  if not os.path.exists(path):
    raise FileNotFoundError(f"Dataset file not found: {path}")
  LOG.info("Loading dataset: %s", path)
  data = pd.read_csv(path)
  data.columns = data.columns.str.strip()
  return data


def preprocess(data: pd.DataFrame, enable_plots: bool) -> tuple[pd.DataFrame, pd.Series, list[str]]:
  # Drop non predictive column if present
  if 'day' in data.columns:
    data = data.drop(columns=['day'])

  # Fill missing where applicable
  if 'winddirection' in data.columns:
    data['winddirection'] = data['winddirection'].fillna(data['winddirection'].mode()[0])
  if 'windspeed' in data.columns:
    data['windspeed'] = data['windspeed'].fillna(data['windspeed'].median())

  if 'rainfall' not in data.columns:
    raise ValueError("Dataset must contain 'rainfall' column with yes/no values")
  data['rainfall'] = data['rainfall'].map({'yes': 1, 'no': 0})

  # Optional EDA
  if enable_plots:
    sns.set(style="whitegrid")
    numeric_hist_cols = [c for c in ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed'] if c in data.columns]
    if numeric_hist_cols:
      plt.figure(figsize=(15, 10))
      for i, column in enumerate(numeric_hist_cols, 1):
        plt.subplot(int(np.ceil(len(numeric_hist_cols)/3.0)), 3, i)
        sns.histplot(data[column], kde=True)
        plt.title(f"Distribution of {column}")
      plt.tight_layout(); plt.show()
    plt.figure(figsize=(6, 4))
    sns.countplot(x='rainfall', data=data)
    plt.title('Distribution of Rainfall'); plt.show()
    corr = data.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation heatmap'); plt.show()

  # Drop correlated columns if present
  correlated_cols = [c for c in ['maxtemp', 'temparature', 'mintemp'] if c in data.columns]
  if correlated_cols:
    data = data.drop(columns=correlated_cols)

  # Downsample for balance
  target_counts = data['rainfall'].value_counts()
  if len(target_counts) != 2:
    raise ValueError("Target 'rainfall' must have exactly two classes after mapping")
  maj_class = target_counts.idxmax()
  min_class = target_counts.idxmin()
  df_majority = data[data['rainfall'] == maj_class]
  df_minority = data[data['rainfall'] == min_class]
  df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
  df_downsampled = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)

  X_raw = df_downsampled.drop(columns=['rainfall'])
  y = df_downsampled['rainfall']
  cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
  if cat_cols:
    LOG.info("One-hot encoding categorical columns: %s", cat_cols)
    X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
  else:
    X = X_raw

  return X, y, X.columns.tolist()


def build_model_and_param_grid(fast: bool, skip_grid: bool) -> tuple[RandomForestClassifier, dict | None]:
  model = RandomForestClassifier(random_state=42)
  if skip_grid:
    return model, None
  if fast:
    grid = {"n_estimators": [100], "max_features": ["sqrt"], "max_depth": [None, 20], "min_samples_split": [2], "min_samples_leaf": [1]}
  else:
    grid = {"n_estimators": [50, 100, 200], "max_features": ["sqrt", "log2"], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
  return model, grid


def train_pipeline(X: pd.DataFrame, y: pd.Series, fast: bool, skip_grid: bool) -> tuple[RandomForestClassifier, dict, list[float]]:
  model, grid = build_model_and_param_grid(fast, skip_grid)
  if grid is None:
    LOG.info("Training baseline RandomForest (no grid search)")
    model.fit(X, y)
    best_model = model
    best_params = model.get_params()
    cv_scores = cross_val_score(best_model, X, y, cv=3)  # quick CV
  else:
    if fast:
      LOG.info("Using FAST hyperparameter grid")
    grid_search = GridSearchCV(model, param_grid=grid, cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_scores = cross_val_score(best_model, X, y, cv=5)
  return best_model, best_params, cv_scores


def evaluate_and_save(model: RandomForestClassifier, X_train, X_test, y_train, y_test, feature_names: list[str], model_path: str, metrics_path: str, *, best_params: dict | None = None, cv_scores: list[float] | None = None):
  y_pred = model.predict(X_test)
  acc = accuracy_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred).tolist()
  clf_report = classification_report(y_test, y_pred, output_dict=True)
  model_data = {"model": model, "feature_names": feature_names}
  with open(model_path, "wb") as f:
    pickle.dump(model_data, f)
  LOG.info("Model saved -> %s", model_path)
  feat_importances = None
  if hasattr(model, "feature_importances_"):
    # Pair feature importance with feature names
    feat_importances = sorted(
      [
        {"feature": f, "importance": float(i)}
        for f, i in zip(feature_names, model.feature_importances_)
      ], key=lambda x: x["importance"], reverse=True
    )
  metrics = {
    "best_params": best_params if best_params is not None else getattr(model, 'get_params', lambda: {})(),
    "test_accuracy": acc,
    "confusion_matrix": cm,
    "classification_report": clf_report,
    "cv_scores": [float(s) for s in cv_scores] if cv_scores is not None else None,
    "cv_mean": float(np.mean(cv_scores)) if cv_scores is not None else None,
    "n_train_samples": int(getattr(X_train, 'shape', [len(X_train)])[0]),
    "n_test_samples": int(getattr(X_test, 'shape', [len(X_test)])[0]),
    "feature_count": len(feature_names),
    "feature_importances": feat_importances,
    "timestamp": datetime.now(timezone.utc).isoformat()
  }
  with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
  LOG.info("Metrics saved -> %s", metrics_path)
  return metrics


def cmd_train(args) -> int:
  data = load_dataset(args.data)
  LOG.debug("Raw shape: %s", (data.shape,))
  X, y, feature_names = preprocess(data, enable_plots=not args.no_plots)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model, best_params, cv_scores = train_pipeline(X_train, y_train, fast=args.fast, skip_grid=args.no_gridsearch)
  LOG.info("Best params: %s", best_params)
  LOG.info("CV scores: %s | mean=%.4f", cv_scores, float(np.mean(cv_scores)))
  evaluate_and_save(
    model, X_train, X_test, y_train, y_test, feature_names,
    args.model_out, args.metrics_out,
    best_params=best_params, cv_scores=cv_scores
  )
  return 0


def _load_model(path: str):
  with open(path, 'rb') as f:
    data = pickle.load(f)
  if not isinstance(data, dict) or 'model' not in data or 'feature_names' not in data:
    raise ValueError("Pickle file format invalid")
  return data['model'], data['feature_names']


def _parse_input_json(arg: str) -> dict:
  if os.path.isfile(arg):
    with open(arg, 'r', encoding='utf-8') as f:
      return json.load(f)
  return json.loads(arg)


def cmd_predict(args) -> int:
  model, feature_names = _load_model(args.model)
  sample = _parse_input_json(args.input_json)
  # Build dataframe with all expected features; fill missing with 0
  row = {f: sample.get(f, 0) for f in feature_names}
  X_df = pd.DataFrame([row], columns=feature_names)
  proba = None
  threshold = args.threshold
  if threshold is not None and not (0.0 <= threshold <= 1.0):
    raise ValueError("--threshold must be between 0 and 1")
  if hasattr(model, 'predict_proba'):
    proba = float(model.predict_proba(X_df)[0][1])
    if threshold is not None:
      pred = 1 if proba >= threshold else 0
    else:
      pred = int(model.predict(X_df)[0])
  else:
    pred = int(model.predict(X_df)[0])
  label = 'yes' if pred == 1 else 'no'
  output = {
    "prediction": int(pred),
    "label": label,
    "rainfall_probability": proba,
    "threshold": threshold
  }
  print(json.dumps(output, indent=2))
  return 0


def run_streamlit_app():
  st.title("Rainfall Prediction")
  st.write("## Train a new model")
  data_file = st.file_uploader("Upload CSV data file", type=["csv"])
  if data_file is not None:
    # Train model with uploaded data
    with st.spinner("Training model..."):
      try:
        data = load_dataset(data_file)
        X, y, feature_names = preprocess(data, enable_plots=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model, best_params, cv_scores = train_pipeline(X_train, y_train, fast=False, skip_grid=False)
        st.success("Model trained successfully!")
      except Exception as e:
        st.error(f"Error training model: {e}")

  st.write("## Predict using a JSON sample")
  json_input = st.text_area("JSON input", height=150)
  if st.button("Predict"):
    if not json_input.strip():
      st.error("JSON input is required")
    else:
      # Predict using JSON input
      with st.spinner("Making prediction..."):
        try:
          model, feature_names = _load_model("rainfall_prediction_model.pkl")
          sample = json.loads(json_input)
          row = {f: sample.get(f, 0) for f in feature_names}
          X_df = pd.DataFrame([row], columns=feature_names)
          proba = float(model.predict_proba(X_df)[0][1])
          pred = int(model.predict(X_df)[0])
          label = 'yes' if pred == 1 else 'no'
          st.write(f"Prediction: {label} (probability: {proba:.2f})")
        except Exception as e:
          st.error(f"Error making prediction: {e}")

  st.write("## Model Metrics")
  if os.path.exists("training_metrics.json"):
    with open("training_metrics.json", "r", encoding="utf-8") as f:
      metrics = json.load(f)
    st.json(metrics)
  else:
    st.write("No metrics found. Train a model first.")

  st.write("## About")
  st.info("This app predicts rainfall using a trained RandomForest model. Upload your CSV data to train a new model, or use the default model to make predictions.")

def main():
  args = parse_args()
  configure_logging(args.verbose)
  # Suppress overly verbose 3rd party logs when not in verbose mode
  if not args.verbose:
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)
  try:
    if args.command == 'train':
      return cmd_train(args)
    if args.command == 'predict':
      return cmd_predict(args)
    if args.command == 'streamlit':
      run_streamlit_app()
      return 0
    raise ValueError(f"Unknown command: {args.command}")
  except Exception as exc:  # pragma: no cover - top-level safety
    LOG.error("Error: %s", exc, exc_info=args.verbose)
    return 1


if __name__ == "__main__":  # pragma: no cover
  sys.exit(main())



