FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-railway.txt .
RUN pip install --no-cache-dir -r requirements-railway.txt

COPY app.py .
COPY specialist_models/ specialist_models/
COPY "rl_agent_weights (1).pth" .
COPY "rl_training_metadata (1).json" .
COPY models/ models/

ENV MODEL_DIR=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "app.py"]
