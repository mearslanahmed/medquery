# Architecture & Multi-Agent Design

## How MedQuery Works

MedQuery is built as a pipeline of small, focused agents rather than one giant prompt. This keeps latency low, stops off-topic questions from wasting embedding and vector tokens, and makes sure the user always gets an answer even if one LLM provider goes down or hits a rate limit.

```mermaid
flowchart TD
    UserQuery([User Input]) --> Triage[TriageAgent]
    
    Triage -- Not Medical --> Reject[Polite Medical Refusal]
    Triage -- Medical Question --> Rewriter[History-Aware Contextualizer]
    
    ChatHistory[(Chat History)] --> Rewriter
    
    Rewriter --> StandaloneQuery[Rewritten Standalone Query]
    
    StandaloneQuery --> Pinecone[(Pinecone Index\n23,167 Passages)]
    Pinecone --> Docs[Top 3 Relevant Passages]
    
    Docs --> RAGAgent[Clinical RAG Agent]
    
    subgraph Fallback Pool
        Groq[Tier 1: Groq\ngpt-oss-120b] -->|429 Rate Limit| OpenRouter[Tier 2: OpenRouter\ngemma-4-31b]
        OpenRouter -->|Outage / Network Error| Gemini[Tier 3: Gemini\n3.5 Flash]
    end
    
    RAGAgent <--> Fallback Pool
    Docs --> CitationAgent[Safety Citation Agent]
    
    RAGAgent --> AnswerStream([Text Stream])
    CitationAgent --> SourceCards([Source Cards & Page Numbers])
```

---

## The Agents Breakdown

### 1. TriageAgent (Domain Guard)
- **File:** `src/agents.py`
- **What it does:** Runs a fast binary check on the user's question before doing any vector searches. If someone asks about Python code, crypto prices, or football scores, it shuts it down immediately.
- **Why it matters:** 
  1. Dense vector searches against Pinecone aren't free in terms of latency. 
  2. Medical models shouldn't give random advice on coding or finance.
- **Speed:** Takes ~100–150ms on Groq.
- **Fallback behavior:** If the triage LLM call fails for any network reason, we default to `True` so genuine patients or medical users aren't locked out.

### 2. ClinicalRAGAgent (Context & Retrieval)
- **File:** `src/agents.py`
- **What it does:** Handles two steps:
  1. **Query Rewriting:** If a user says *"What are its side effects?"* after asking about Amoxicillin, standard vector search would fail because "its" has no meaning on its own. The history rewriter turns this into *"What are the side effects of Amoxicillin?"* before searching Pinecone.
  2. **Answer Synthesis:** Pulls the top 3 passages from Pinecone, feeds them to the LLM, and formats the response using clean clinical headings.

### 3. The Multi-Provider Fallback Pool
- **File:** `src/agents.py:get_resilient_llm()`
- **The Problem:** Free-tier LLM endpoints frequently return HTTP 429 (rate limit exceeded) during busy hours. If you rely on just one provider, your app crashes or throws errors to the user.
- **Our Solution:** We chained three providers using LangChain's `.with_fallbacks()`:

```python
# 1. Primary: Groq (ultra fast, ~400 tokens/sec)
groq_llm = ChatGroq(model_name="openai/gpt-oss-120b", ...)

# 2. Secondary: OpenRouter Free Tier (reliable open weights)
openrouter_llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1", model="google/gemma-4-31b-it:free", ...)

# 3. Backup: Google Gemini (high availability safety net)
gemini_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", ...)
```

If Groq returns a 429, LangChain catches the exception in under 50ms and redirects the exact same prompt to OpenRouter. If OpenRouter is down, it drops down to Gemini. The user never notices anything broke.

### 4. SafetyCitationAgent (Citations & Source Cards)
- **File:** `src/agents.py`
- **What it does:** When Pinecone returns chunks, each chunk contains metadata showing the PDF filename and the physical page number.
- **Deduplication:** Often two chunks come from the same page of Harrison's Internal Medicine. The agent deduplicates by `(filename, page_number)` so the UI doesn't show 3 identical citations for the same page. It also grabs a 200-character snippet so users can verify the source text themselves.
