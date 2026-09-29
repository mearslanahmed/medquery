import os
import json
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain.chains import create_retrieval_chain, create_history_aware_retriever
    from langchain.chains.combine_documents import create_stuff_documents_chain

from src.prompt import system_prompt

load_dotenv()

# 1. RESILIENT MULTI-PROVIDER FALLBACK POOL

def get_resilient_llm():
    """
    Creates an LLM with automatic 3-tier fallback:
      Tier 1: Groq Cloud (Ultra-fast open models)
      Tier 2: OpenRouter Free Tier (Deep open-source models)
      Tier 3: Google Gemini Free Tier (Reliable backup)
    If any provider hits a 429 quota or connection error, it seamlessly fails over.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("Gemini_API_KEY")

    models = []

    # 1. Primary: Groq
    if groq_key:
        groq_llm = ChatGroq(
            model_name="openai/gpt-oss-120b",
            groq_api_key=groq_key,
            temperature=0.2,
            max_retries=1
        )
        models.append(groq_llm)

    # 2. Secondary: OpenRouter Free Tier
    if openrouter_key:
        openrouter_llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            model="google/gemma-4-31b-it:free",
            default_headers={
                "HTTP-Referer": "http://localhost:8080",
                "X-Title": "MedQuery Medical Assistant"
            },
            temperature=0.2,
            max_retries=1
        )
        models.append(openrouter_llm)

    # 3. Tertiary: Gemini
    if gemini_key:
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            google_api_key=gemini_key,
            temperature=0.2,
            max_retries=1
        )
        models.append(gemini_llm)
    if not models:
        raise ValueError("No LLM API keys found. Please set GROQ_API_KEY, OPENROUTER_API_KEY, or GEMINI_API_KEY in .env.")
    # Chain with fallbacks: models[0] falls back to models[1], etc.
    primary = models[0]
    if len(models) > 1:
        return primary.with_fallbacks(models[1:])
    return primary


# 2. AGENT 1: TRIAGE & DOMAIN GUARD AGENT

class TriageAgent:
    """
    Rapidly classifies if a question is health/medical related.
    Rejects off-topic inquiries (currency, coding, math, general chat)
    in milliseconds without querying Pinecone or wasting token quota.
    """
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an automated medical domain triage filter. "
                "Classify whether the user input is related to medicine, health, symptoms, biology, "
                "treatments, anatomy, pharmacology, wellness, or medical advice.\n"
                "Respond ONLY with 'MEDICAL' if health-related, or 'OFF_TOPIC' if not (e.g. coding, finance, currency, sports, weather)."
            ),
            ("human", "{input}")
        ])
        self.chain = self.prompt | self.llm

    def is_medical(self, user_query: str) -> bool:
        try:
            response = self.chain.invoke({"input": user_query})
            decision = response.content.strip().upper()
            return "MEDICAL" in decision
        except Exception:
            # On fallback/network hiccup, default to True so user is not blocked
            return True


# 3. AGENT 2: CLINICAL RAG AGENT

class ClinicalRAGAgent:
    """
    Performs history-aware conversational query reformulation,
    queries the Pinecone medical index, and synthesizes clinical answers.
    """
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        # Contextualize question prompt
        contextualize_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_prompt
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)

    def invoke(self, user_query: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        return self.rag_chain.invoke({
            "input": user_query,
            "chat_history": chat_history
        })
    def stream(self, user_query: str, chat_history: List[BaseMessage]):
        return self.rag_chain.stream({
            "input": user_query,
            "chat_history": chat_history
        })

# 4. AGENT 3: CLINICAL SAFETY & CITATION AGENT

class SafetyCitationAgent:
    """
    Extracts, deduplicates, and formats clinical citations from retrieved context documents.
    """
    @staticmethod
    def format_sources(raw_docs: list) -> List[Dict[str, Any]]:
        sources = []
        seen = set()
        for doc in raw_docs:
            source_path = doc.metadata.get("source", "Medical Reference Guide")
            file_name = os.path.basename(source_path) if source_path else "Medical Reference"
            page_num = doc.metadata.get("page")
            display_page = page_num + 1 if isinstance(page_num, int) else None
            # Deduplicate by file and page
            key = (file_name, display_page)
            if key in seen:
                continue
            seen.add(key)
            snippet = doc.page_content.strip()
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            sources.append({
                "file": file_name,
                "page": display_page,
                "snippet": snippet
            })
        return sources


# 5. MULTI-AGENT ORCHESTRATOR
class MedQueryMultiAgent:
    """
    Unified multi-agent orchestrator:
      1. Triage Agent checks domain
      2. If off-topic -> returns immediate medical refusal
      3. If medical -> Clinical RAG Agent retrieves Pinecone context and generates answer
      4. Safety Agent formats verified source citations
    """
    def __init__(self, retriever):
        self.llm = get_resilient_llm()
        self.triage_agent = TriageAgent(self.llm)
        self.clinical_rag_agent = ClinicalRAGAgent(self.llm, retriever)
        self.safety_agent = SafetyCitationAgent()
    def process_query(self, user_query: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        # Step 1: Triage domain check
        if not self.triage_agent.is_medical(user_query):
            return {
                "answer": (
                    "I am a specialized medical assistant and can only answer questions related to health, "
                    "medicine, and clinical topics. Please ask a health-related question."
                ),
                "sources": [],
                "is_medical": False
            }
        # Step 2: Clinical RAG execution
        rag_output = self.clinical_rag_agent.invoke(user_query, chat_history)
        answer = rag_output.get("answer", "No answer generated.")
        # Step 3: Safety and citation extraction
        raw_docs = rag_output.get("context", [])
        sources = self.safety_agent.format_sources(raw_docs)
        return {
            "answer": answer,
            "sources": sources,
            "is_medical": True
        }

    def process_query_stream(self, user_query: str, chat_history: List[BaseMessage]) -> Generator[Dict[str, Any], None, None]:
        # Step 1: Triage domain check
        if not self.triage_agent.is_medical(user_query):
            refusal = (
                "I am a specialized medical assistant and can only answer questions related to health, "
                "medicine, and clinical topics. Please ask a health-related question."
            )
            yield {"type": "token", "token": refusal}
            yield {"type": "done"}
            return

        # Step 2: Clinical RAG streaming
        sources_sent = False
        for chunk in self.clinical_rag_agent.stream(user_query, chat_history):
            if "context" in chunk and not sources_sent:
                raw_docs = chunk["context"]
                sources = self.safety_agent.format_sources(raw_docs)
                sources_sent = True
                yield {"type": "sources", "sources": sources}

            if "answer" in chunk:
                yield {"type": "token", "token": chunk["answer"]}

        yield {"type": "done"}