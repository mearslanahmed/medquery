# MedQuery Documentation

This folder contains the technical documentation for MedQuery. It covers how the multi-agent system is structured, how we ingested and indexed 23,000+ medical textbook passages into Pinecone, how the API works, and how the app is deployed on Render's free tier without running out of memory.

---

## Documentation Index

### [1. Architecture & Multi-Agent Flow](architecture.md)
How the request pipeline works from user input to output stream:
- Triage filter that drops non-medical prompts in ~100ms
- Query contextualizer that rewrites follow-up questions using chat history
- Multi-provider LLM fallback pool (Groq -> OpenRouter -> Gemini) so queries never fail on rate limits
- Citation parser that extracts textbook names and page numbers

### [2. Vector Indexing & RAG Pipeline](rag_and_indexing.md)
How we turned 6 medical textbooks into searchable vectors:
- Text extraction and chunking settings (1,000 chars, 100 overlap)
- Embedding model choices (`all-MiniLM-L6-v2`, 384 dimensions)
- Pinecone serverless index setup
- Memory-safe batch ingestion so scripts don't crash on large PDFs

### [3. API Reference](api_reference.md)
Endpoints, payloads, and integration examples:
- `POST /api/chat/stream` (Server-Sent Events for streaming answers)
- `POST /api/chat` (Standard JSON response)
- `GET /api/health` and `GET /ping` (Uptime probes)
- Client examples in cURL, Python, and JavaScript

### [4. Deployment & DevOps Runbook](deployment_and_devops.md)
Production notes and how we keep it running on Render for free:
- Why we bake embedding models during `docker build` instead of runtime
- Fixing 512 MB RAM crashes using `MALLOC_ARENA_MAX=2` and CPU-only PyTorch
- Gunicorn process and thread configuration
- GitHub Actions cron job that pings `/api/health` every 12 minutes to stop cold starts
- Render blueprint (`render.yaml`)

### [5. Clinical Safety & Prompting Rules](clinical_safety_guidelines.md)
Rules enforced on the model outputs:
- Medical-only domain guardrails
- Why markdown tables are banned (they break mobile screens)
- Removing conversational filler ("Certainly!", "As an AI language model...")
- Grounding answers in textbook references

---

## Quick System Specs

| Property | Value |
| :--- | :--- |
| **Backend** | Python 3.11, Flask, Gunicorn |
| **Orchestration** | LangChain |
| **Primary Model** | Groq (`openai/gpt-oss-120b`) |
| **Fallback 1** | OpenRouter (`google/gemma-4-31b-it:free`) |
| **Fallback 2** | Google Gemini 3.5 Flash |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| **Vector DB** | Pinecone Serverless (Cosine) |
| **Data Volume** | 23,167 passages from 6 medical textbooks |
| **Container Target** | Render Free Tier (512 MB RAM) |
| **Live URL** | https://medquery-chatbot.onrender.com |
