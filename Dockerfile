FROM python:3.11-slim

# Create a non-privileged user for container security
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application code, champion model, reference data, and reports
COPY app.py .
COPY src/ src/
COPY models/model.pkl models/model.pkl
COPY reports/ reports/
COPY data/processed/X_train.csv data/processed/X_train.csv

# Ensure permissions for runtime monitoring logs
RUN mkdir -p data/monitoring reports && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
