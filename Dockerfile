FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
		PYTHONUNBUFFERED=1 \
		PIP_NO_CACHE_DIR=1 \
		PORT=8501

WORKDIR /app

# System deps (if any future need for build)
RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Disable usage stats in container
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# Pre-train model during build (optional; ignore failure so container still starts and can auto-train)
RUN python rainfall_prediction_using_machine_learning.py train --data Rainfall.csv --no-gridsearch --no-plots --model-out rainfall_prediction_model.pkl --metrics-out training_metrics.json || true

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \\
	CMD wget -qO- http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
