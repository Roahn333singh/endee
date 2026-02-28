# RAG Chatbot with Endee + LangGraph

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about PDF documents using **Endee** as the vector database and **LangGraph** for agentic orchestration.

## Problem Statement

Large PDF documents are difficult to search and extract specific information from. Traditional keyword search fails when users ask natural-language questions. This project solves that by building a conversational chatbot that:

- Ingests a PDF into a vector database (Endee)
- Retrieves the most relevant passages for each user question using semantic search
- Generates accurate, context-grounded answers using an LLM (Google Gemini)
- Maintains conversation history across turns for follow-up questions

## System Design & Technical Approach

```
┌────────────────────────────────────────────────────────┐
│                    LangGraph State Machine              │
│                                                        │
│   User Question                                        │
│        │                                               │
│        ▼                                               │
│   ┌──────────┐    embed query     ┌───────────────┐   │
│   │ Retrieve  │ ────────────────► │  Endee Vector  │   │
│   │   Node    │ ◄──── top-k ──── │   Database     │   │
│   └────┬─────┘    similar chunks  └───────────────┘   │
│        │                                               │
│        ▼                                               │
│   ┌──────────┐    context +       ┌───────────────┐   │
│   │ Generate  │    history        │  Google Gemini │   │
│   │   Node    │ ────────────────► │     LLM       │   │
│   └────┬─────┘                    └───────┬───────┘   │
│        │                                  │            │
│        ◄──────── answer ─────────────────┘            │
│        │                                               │
│        ▼                                               │
│   Assistant Reply                                      │
│                                                        │
│   [InMemorySaver persists state across turns]          │
└────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vector Database | **Endee** | Stores document embeddings, performs cosine similarity search |
| Embeddings | Google Gemini `embedding-001` | Converts text chunks into 3072-dim vectors |
| LLM | Google Gemini `gemini-2.5-flash` | Generates answers from retrieved context |
| Orchestration | **LangGraph** | Manages the retrieve → generate pipeline as a stateful graph |
| Memory | LangGraph `InMemorySaver` | Persists conversation history for multi-turn chat |
| Document Loader | LangChain `PyPDFLoader` | Extracts text from PDF files |
| Text Splitter | `RecursiveCharacterTextSplitter` | Splits documents into overlapping chunks for better retrieval |

### How Endee Is Used

Endee is the **core** of this project's retrieval pipeline:

1. **Index Creation** — On first run, a cosine-similarity index (`ebook_rag`) is created in Endee with dimension 3072 to match the Gemini embedding model output.

2. **Document Ingestion** — The PDF is split into small overlapping chunks (100 chars, 20 overlap). Each chunk is embedded using Gemini and upserted into Endee with the original text stored as metadata.

3. **Semantic Search** — When the user asks a question, the query is embedded and Endee's `index.query()` retrieves the top-5 most similar chunks via cosine similarity.

4. **Context for RAG** — The retrieved chunk texts are concatenated and passed to the LLM as context, grounding the generated answer in the actual document content.

Endee handles all vector storage and similarity search, making it the backbone of the retrieval step.

## Project Structure

```
rag-chatbot/
├── EbookBot.py          # Main chatbot application
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Endee server configuration
├── .env.example         # Environment variable template
├── data/
│   └── Ebook-Agentic-AI.pdf   # Sample PDF document
└── README.md            # This file
```

## Setup & Execution Instructions

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- A Google API key with Gemini access ([Get one here](https://aistudio.google.com/apikey))

### 1. Clone the Repository

```bash
git clone https://github.com/Roahn333singh/endee.git
cd endee/rag-chatbot
```

### 2. Start the Endee Vector Database

```bash
docker compose up -d
```

Verify it's running:

```bash
curl http://localhost:8080/api/v1/health
```

### 3. Set Up Python Environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your Google API key:

```
GOOGLE_API_KEY=your-actual-api-key
```

### 5. Run the Chatbot

Using the included sample PDF:

```bash
python EbookBot.py
```

Or with your own PDF:

```bash
python EbookBot.py --pdf path/to/your-document.pdf
```

### 6. Chat

```
=== Document Q&A Chatbot (type 'quit' to exit) ===

You: What is agentic AI?