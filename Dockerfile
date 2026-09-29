FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Memory optimization for 512MB RAM container
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TORCH_NUM_THREADS=1

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install CPU-only PyTorch to keep image small and memory lean
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt setup.py /app/
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache HuggingFace embeddings during docker build so container starts in <1 second (no network wait at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy source code (respecting .dockerignore)
COPY . /app/

EXPOSE 8080

# Single worker with 2 threads and request recycling to guarantee memory stays under 250MB
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 2 --timeout 120 --max-requests 500 --max-requests-jitter 50 app:app"]
