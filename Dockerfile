FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install CPU-only PyTorch to keep the image lightweight (saves ~2GB)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt setup.py /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (respecting .dockerignore)
COPY . /app/

EXPOSE 8080

# Production gunicorn with multi-threading for SSE streaming
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 120 app:app"]