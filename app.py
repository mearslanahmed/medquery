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

embeddings = download_hugging_face_embeddings()

index_name = "medquery"
# Embed each chunk and upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY
)

# 1. Contextualize Question Prompt for Conversational Memory
# This reformulates follow-up queries (e.g., "What are its symptoms?") into standalone search queries for Pinecone
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    chatModel, retriever, contextualize_q_prompt
)

# 2. Answer Generation Prompt including chat history & retrieved medical context
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": "gemini-1.5-flash",
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
        
        # Invoke history-aware retrieval and QA chain
        response = rag_chain.invoke({
            "input": msg,
            "chat_history": chat_history
        })
        answer = response.get("answer", "No answer generated.")
        
        # Extract source citations from retrieved context documents
        sources = []
        raw_docs = response.get("context", [])
        for doc in raw_docs:
            source_path = doc.metadata.get("source", "Medical Reference Guide")
            file_name = os.path.basename(source_path) if source_path else "Medical Reference"
            page_num = doc.metadata.get("page")
            
            # PyPDF uses 0-indexed page numbers, convert to 1-indexed for humans
            display_page = page_num + 1 if isinstance(page_num, int) else None
            snippet = doc.page_content.strip()
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
                
            sources.append({
                "file": file_name,
                "page": display_page,
                "snippet": snippet
            })
        
        return jsonify({
            "status": "success",
            "answer": answer,
            "sources": sources,
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

        def generate():
            try:
                sources_sent = False
                for chunk in rag_chain.stream({"input": msg, "chat_history": chat_history}):
                    # Stream context sources if available
                    if "context" in chunk and not sources_sent:
                        sources = []
                        for doc in chunk["context"]:
                            source_path = doc.metadata.get("source", "Medical Reference Guide")
                            file_name = os.path.basename(source_path) if source_path else "Medical Reference"
                            page_num = doc.metadata.get("page")
                            display_page = page_num + 1 if isinstance(page_num, int) else None
                            snippet = doc.page_content.strip()
                            if len(snippet) > 220:
                                snippet = snippet[:220] + "..."
                            sources.append({
                                "file": file_name,
                                "page": display_page,
                                "snippet": snippet
                            })
                        sources_sent = True
                        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

                    # Stream answer tokens character-by-character / word-by-word
                    if "answer" in chunk:
                        token = chunk["answer"]
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"
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
        response = rag_chain.invoke({"input": msg})
        return str(response.get("answer", ""))
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
