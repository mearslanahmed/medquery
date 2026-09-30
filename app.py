from flask import Flask, render_template, jsonify, request, Response, stream_with_context
import json
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
try:
    from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain.chains import create_retrieval_chain, create_history_aware_retriever
    from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY or ""

import gc

embeddings = download_hugging_face_embeddings()
gc.collect()

from src.agents import MedQueryMultiAgent

index_name = "medquery"
med_agent = None

def get_med_agent():
    global med_agent
    if med_agent is None:
        try:
            docsearch = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=embeddings
            )
            # Fetch top-10 candidates from Pinecone (wide net).
            # CrossEncoderReranker in MedQueryMultiAgent will score all 10
            # with a cross-encoder and return only the best 3 to the LLM.
            # Net result: same token cost, but much higher retrieval precision.
            retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 10})
            med_agent = MedQueryMultiAgent(retriever)
            gc.collect()
        except Exception as e:
            print(f"Error initializing Pinecone/Multi-Agent: {e}")
            raise
    return med_agent

# Eager warm-up attempt without blocking Flask startup
try:
    get_med_agent()
except Exception as init_err:
    print(f"Startup notice: Agent warm-up will retry on first request: {init_err}")


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "system": "Multi-Agent Clinical Pipeline",
        "providers": "Groq + OpenRouter + Gemini Fallback Pool",
        "index": index_name
    })

def format_user_friendly_error(err: Exception) -> str:
    return "The medical assistant service is temporarily experiencing high traffic or a connection delay. Please try your question again in a moment."

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.get_json(force=True, silent=True) or {}
        msg = data.get("msg") or request.form.get("msg", "").strip()
        history_data = data.get("history", [])
        
        if not msg:
            return jsonify({"error": "Message cannot be empty", "status": "error"}), 400
        
        # Convert incoming JSON chat history into LangChain message objects
        chat_history = []
        for turn in history_data:
            role = turn.get("role")
            text = turn.get("text") or turn.get("content", "")
            if role == "user" and text:
                chat_history.append(HumanMessage(content=text))
            elif role in ("assistant", "bot") and text:
                chat_history.append(AIMessage(content=text))
        
        # Process through the multi-agent pipeline
        agent = get_med_agent()
        result = agent.process_query(msg, chat_history)
        
        return jsonify({
            "status": "success",
            "answer": result["answer"],
            "sources": result["sources"],
            "query": msg
        })
    except Exception as e:
        friendly_error = format_user_friendly_error(e)
        return jsonify({
            "status": "error",
            "answer": friendly_error,
            "error": friendly_error
        }), 500

@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    try:
        data = request.get_json(force=True, silent=True) or {}
        msg = data.get("msg") or request.form.get("msg", "").strip()
        history_data = data.get("history", [])

        if not msg:
            return jsonify({"error": "Message cannot be empty", "status": "error"}), 400

        chat_history = []
        for turn in history_data:
            role = turn.get("role")
            text = turn.get("text") or turn.get("content", "")
            if role == "user" and text:
                chat_history.append(HumanMessage(content=text))
            elif role in ("assistant", "bot") and text:
                chat_history.append(AIMessage(content=text))

        agent = get_med_agent()

        def generate():
            try:
                for event in agent.process_query_stream(msg, chat_history):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as ex:
                friendly_err = format_user_friendly_error(ex)
                yield f"data: {json.dumps({'type': 'error', 'error': friendly_err})}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    except Exception as e:
        friendly_err = format_user_friendly_error(e)
        return jsonify({"error": friendly_err, "status": "error"}), 500

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg") or request.args.get("msg", "")
    if not msg:
        return "Please provide a query."
    try:
        agent = get_med_agent()
        result = agent.process_query(msg, [])
        return str(result.get("answer", ""))
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    is_debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true")
    app.run(host="0.0.0.0", port=port, debug=is_debug)

