# 🩺 MedQuery — Clinical Multi-Agent RAG Assistant

[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-0071e3?style=for-the-badge&logo=render&logoColor=white)](https://medquery-chatbot.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-23k+_Vectors-000000?style=for-the-badge&logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-Automated-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/mearslanahmed/medquery)

> **MedQuery** is an enterprise-grade Clinical Decision Support and Medical Reference Assistant powered by a **Multi-Agent Retrieval-Augmented Generation (RAG)** architecture, an indexed knowledge base of **23,000+ medical textbook passages**, and a resilient **3-tier LLM fallback pool** (Groq, OpenRouter, Google Gemini).

🔗 **Live Web Application:** [https://medquery-chatbot.onrender.com](https://medquery-chatbot.onrender.com)

---

## 🌟 Key Highlights

- **Multi-Agent Architecture:**
  - 🛡️ **Triage Agent:** Sub-second classification of user queries to enforce strict clinical and biomedical domain boundaries.
  - 📚 **Clinical RAG Agent:** History-aware query reformulation coupled with dense semantic search against 23,000+ textbook passages in Pinecone.
  - 🔍 **Safety & Citation Agent:** Automatic deduplication and extraction of clinical sources, book titles, page numbers, and exact verbatim context snippets.
- **Resilient Multi-Provider Fallback Pool:**
  - Automatic zero-downtime cascading failover: **Groq (`openai/gpt-oss-120b`)** &rarr; **OpenRouter Free Tier (`google/gemma-4-31b-it`)** &rarr; **Google Gemini 1.5**.
- **Clean Clinical Formatting (No Robotic AI Chatter):**
  - Strictly bans markdown tables and conversational pleasantries (*"Certainly!"*, *"As an AI..."*). Delivers high-yield clinical sections: Overview, Mechanism of Action, Indications, Dosage, and Precautions.
- **Apple-Inspired Consultation Interface:**
  - Dark/Light mode, voice dictation, SSE streaming, consultation export, interactive source cards, and an instant zero-latency splash screen.
- **Production & 24/7 Keep-Alive CI/CD:**
  - Containerized with memory-optimized CPU-only PyTorch (<180 MB RAM footprint).
  - Automated GitHub Actions keep-alive heartbeat preventing cloud sleep.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([User Consultation Input]) --> Triage{Triage Agent}
    Triage -- Off-Topic Question --> Refusal[Strict Medical Boundary Refusal]
    Triage -- Health / Biomedical Query --> RAG[Clinical RAG Agent]
    
    subgraph Vector Knowledge Base
        Books[(5 Medical Textbooks\n23,167 Passages)] --> Pinecone[(Pinecone Vector Store)]
    end
    
    RAG <--> |Dense Embedding Search| Pinecone
    
    subgraph Resilient LLM Pool
        Groq[Tier 1: Groq Cloud\ngpt-oss-120b] -->|Failover 429| OpenRouter[Tier 2: OpenRouter\ngemma-4-31b]
        OpenRouter -->|Failover| Gemini[Tier 3: Google Gemini\nFlash]
    end
    
    RAG <--> Resilient LLM Pool
    RAG --> Safety[Safety & Citation Agent]
    Safety --> Stream([Server-Sent Events Stream & Citations])
```

---

## 🛠️ Tech Stack

- **Backend Framework:** Python 3.11, Flask, Gunicorn
- **LLM Orchestration:** LangChain Core, LangChain Classic
- **Embeddings & Vector Database:** `sentence-transformers/all-MiniLM-L6-v2`, Pinecone Serverless
- **Inference Providers:** Groq API, OpenRouter API, Google Generative AI
- **Frontend:** Vanilla JavaScript (ES6+), Server-Sent Events (SSE), Marked.js, Lucide Icons, Custom Apple-Style CSS Design System
- **DevOps & Cloud:** Docker, GitHub Actions CI/CD, Render Web Service

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/mearslanahmed/medquery.git
cd medquery
```

### 2. Set Up Virtual Environment
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: .\myenv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
PINECONE_API_KEY="your-pinecone-api-key"
GROQ_API_KEY="your-groq-api-key"
OPENROUTER_API_KEY="your-openrouter-api-key"
GEMINI_API_KEY="your-gemini-api-key"
```

### 4. Run the Application Locally
```bash
python app.py
```
Visit `http://localhost:8080` in your browser.

---

## 🐳 Running with Docker

```bash
docker build -t medquery:latest .
docker run -p 8080:8080 --env-file .env medquery:latest
```

---

## 📄 License
This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.