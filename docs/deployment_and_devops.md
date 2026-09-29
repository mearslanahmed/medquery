# Deployment & DevOps Runbook

## Overview

MedQuery is deployed on Render's free tier using Docker. Running a PyTorch-based RAG app within Render's free 512 MB RAM limit requires specific memory and startup optimizations. This document details the exact problems we ran into and how we solved them.

---

## 1. Problem 1: The 502 Bad Gateway (Port Scan Timeout)

### What happened:
When Render booted our container, the logs repeatedly showed:
```text
==> No open HTTP ports detected on 0.0.0.0, continuing to scan...
Warning: You are sending unauthenticated requests to the HF Hub.
==> Port scan timeout reached, no open HTTP ports detected.
```

### The Root Cause:
`app.py` was downloading the 90 MB HuggingFace embedding model (`all-MiniLM-L6-v2`) from the internet at runtime inside Gunicorn's worker boot sequence. Because downloading across the network took 60–90 seconds, Gunicorn had not opened port 10000 yet. Render's health prober checked port 10000, found it closed, timed out after 5 minutes, and threw a 502 Bad Gateway.

### The Fix:
We baked the model directly into the Docker image during build time:
```dockerfile
# Pre-cache model weights during docker build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```
Now, when the container boots on Render, it loads the model from the local disk in **0.4 seconds** instead of downloading over the internet. Port 10000 opens instantly.

---

## 2. Problem 2: Memory Limit Exceeded (512 MB RAM Crash)

### What happened:
Render emailed us: *"Web Service medquery exceeded its memory limit"*.

### The Root Causes:
1. **CUDA PyTorch:** By default, `pip install sentence-transformers` pulls PyTorch with full NVIDIA CUDA GPU libraries (~2.5 GB). When PyTorch initializes with CUDA on Linux, it allocates 600–800 MB of RAM immediately, blowing past Render's 512 MB ceiling.
2. **Linux Glibc Memory Fragmentation:** Linux glibc's memory allocator creates $8 \times \text{number of CPU cores}$ memory arenas. On Render's 8-core host nodes, glibc created 64 arenas, fragmenting virtual memory.
3. **No `.dockerignore`:** The build was copying 680 MB of PDFs from `data/` and local virtualenvs into the container.

### The Fixes in `Dockerfile`:
```dockerfile
FROM python:3.11-slim

# Fix 1: Restrict glibc memory arenas to 2 to stop memory fragmentation
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TORCH_NUM_THREADS=1

# Fix 2: Install CPU-only PyTorch (shrinks package from 2.5 GB to 150 MB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Fix 3: Run Gunicorn with 1 worker and 2 threads to prevent duplicate models in RAM
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 2 --timeout 120 --max-requests 500 app:app"]
```

Result: Runtime memory stays stable at **~180 MB**, well below the 512 MB limit.

---

## 3. Problem 3: Cold Starts (Free Tier Sleeping)

### What happened:
Render puts free services to sleep after 15 minutes of zero traffic. The next user who opens the site has to wait 30–50 seconds for the container to wake up.

### The Fix (`.github/workflows/keep_alive.yaml`):
We set up a GitHub Actions cron job that pings `/api/health` every 12 minutes:

```yaml
name: Keep MedQuery Awake (Free 24/7 Uptime)

on:
  schedule:
    # Runs every 12 minutes (Render sleeps after 15 min idle)
    - cron: '*/12 * * * *'
  workflow_dispatch:

jobs:
  heartbeat-ping:
    runs-on: ubuntu-latest
    steps:
      - name: Send Keep-Alive Ping
        run: |
          TARGET_URL="https://medquery-chatbot.onrender.com"
          curl -fsS -m 30 "$TARGET_URL/api/health" || echo "Ping dispatched"
```

Because Render sees incoming HTTP traffic every 12 minutes, the 15-minute idle countdown resets continuously. The app never sleeps and cold starts are eliminated.

---

## 4. Deploying via `render.yaml`

We created `render.yaml` so you can deploy with one click:

```yaml
services:
  - type: web
    name: medquery-chatbot
    runtime: docker
    plan: free
    region: oregon
    branch: main
    autoDeploy: true
    healthCheckPath: /api/health
    dockerfilePath: ./Dockerfile
    dockerContext: .
    envVars:
      - key: PINECONE_API_KEY
        sync: false
      - key: GEMINI_API_KEY
        sync: false
      - key: GROQ_API_KEY
        sync: false
      - key: OPENROUTER_API_KEY
        sync: false
```

Just add your 4 API keys in Render's dashboard and Render handles the rest on every `git push`.
