# ---------------------------------------------------------------------------
# Mobile Price Prediction - Docker image
# Trains the model from the notebook at BUILD time, then serves the
# Streamlit app at CONTAINER RUN time.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# System deps needed by some scientific-python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY Group2.ipynb .
COPY app.py .
COPY Mobile_Price_Prediction_train.csv .
COPY Mobile_Price_Prediction_test.csv .

# Execute the notebook at build time to generate the .pkl model artifacts
RUN jupyter nbconvert --to notebook --execute --inplace Group2.ipynb \
    --ExecutePreprocessor.timeout=600

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
