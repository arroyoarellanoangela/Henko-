# ---- Stage 1: Frontend build ----
FROM node:22-alpine AS frontend

WORKDIR /app/frontend
COPY ["kaizen front/package.json", "kaizen front/package-lock.json", "./"]
RUN npm ci --production=false
COPY ["kaizen front/", "./"]
RUN npm run build


# ---- Stage 2: Python backend ----
FROM python:3.12-slim AS backend

# System deps for torch + chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY rag/ ./rag/
COPY finetune/ ./finetune/
COPY tests/ ./tests/
COPY api.py app.py ingest.py bench.py pyproject.toml ./

# Copy built frontend into static serving dir
COPY --from=frontend /app/frontend/dist ./static/

# Data dirs (will be mounted as volumes in production)
RUN mkdir -p /app/data/chroma /app/data/eval /app/data/domains /app/data/finetune

# Default env vars (override via docker-compose or .env)
ENV CHROMA_DIR=/app/data/chroma \
    KNOWLEDGE_DIR=/app/data/knowledge \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8000/api/health',timeout=3); assert r.status_code==200"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
