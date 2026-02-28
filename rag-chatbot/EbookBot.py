import argparse
import os
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = "ebook_rag"
DEFAULT_PDF = os.path.join(os.path.dirname(__file__), "data", "Ebook-Agentic-AI.pdf")
EMBED_DIM = 3072
BATCH_SIZE = 1000


# ── State ────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str


# ── Globals initialised once ─────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY,
    max_output_tokens=2048,
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def _get_or_create_index(client):
    """Return the Endee index, creating + populating it if it doesn't exist."""
    from endee.exceptions import ConflictException

    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            space_type="cosine",
            precision="float32",
        )
        print(f"[endee] Created index '{INDEX_NAME}'. Ingesting PDF …")

        index = client.get_index(name=INDEX_NAME)
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_documents(docs)

        vectors = embeddings.embed_documents([c.page_content for c in chunks])

        for start in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[start : start + BATCH_SIZE]
            batch_vectors = vectors[start : start + BATCH_SIZE]
            rows = [
                {
                    "id": f"chunk_{start + i}",
                    "vector": v if isinstance(v, list) else v.tolist(),
                    "meta": {"text": c.page_content},
                }
                for i, (c, v) in enumerate(zip(batch_chunks, batch_vectors))
            ]
            index.upsert(rows)
            print(f"  upserted {start + len(rows)}/{len(chunks)}")

        print("[endee] Ingestion complete.")
        return index

    except ConflictException:
        print(f"[endee] Index '{INDEX_NAME}' already exists — skipping ingestion.")
        return client.get_index(name=INDEX_NAME)


# ── Graph nodes ──────────────────────────────────────────────────────────────

def retrieve(state: ChatState) -> dict:
    """Embed the latest user message and query Endee for relevant chunks."""
    user_msg = state["messages"][-1].content
    query_vec = embeddings.embed_query(user_msg)
    results = endee_index.query(vector=query_vec, top_k=5)
    context = "\n\n".join(r["meta"]["text"] for r in results)
    return {"context": context}


def generate(state: ChatState) -> dict:
    """Build a RAG prompt from context + conversation history, then call LLM."""
    context = state.get("context", "")
    history = state["messages"]

    system_prompt = (
        "You are a helpful assistant that answers questions about a document. "
        "Use ONLY the retrieved context below to answer. "
        "If the answer is not in the context, say you don't have enough information. "
        "Give complete, concise answers — never stop mid-sentence.\n\n"
        f"Retrieved context:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m.type, "content": m.content} for m in history
    ]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ── Build the graph ──────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)


# ── CLI chat loop ────────────────────────────────────────────────────────────

def main():
    from endee import Endee

    parser = argparse.ArgumentParser(description="RAG chatbot over a PDF using Endee + LangGraph")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to the PDF file to ingest")
    args = parser.parse_args()

    global PDF_PATH, endee_index
    PDF_PATH = args.pdf

    if not os.path.isfile(PDF_PATH):
        print(f"Error: PDF not found at '{PDF_PATH}'")
        print("Usage: python EbookBot.py --pdf path/to/your.pdf")
        return

    client = Endee()
    endee_index = _get_or_create_index(client)

    app = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\n=== Document Q&A Chatbot (type 'quit' to exit) ===\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = app.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )

        assistant_reply = result["messages"][-1].content
        print(f"\nAssistant: {assistant_reply}\n")


if __name__ == "__main__":
    main()
