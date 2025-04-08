# 🧠 RAG – Medical Retrieval-Augmented Generation System

This project implements a custom **RAG (Retrieval-Augmented Generation)** system for intelligent document understanding and question answering. It's built using:

- ⚙️ **FastAPI** for backend APIs
- 🧠 **Cohere** for both embeddings & LLM answer generation
- 📊 **PGVector** as the vector similarity database
- 🎛️ **Streamlit** interface for file upload and user interaction

---

## 🔧 Features

- ✅ Upload `.txt` or `.pdf` medical documents
- ✅ Automatic chunking of text into digestible segments
- ✅ Generation of **384-dimensional embeddings** using Cohere
- ✅ Storage of vectors in **PGVector** for semantic retrieval
- ✅ LLM-powered answer generation based on query + retrieved context

---

## 📁 Project Structure

```
mini-rag/
├── src/
│   ├── main.py                  # FastAPI app entry point
│   ├── routes/                  # API endpoints (data, NLP)
│   ├── controllers/             # Business logic (file processing, indexing, search)
│   ├── models/                  # DB models + ORM layer
│   ├── stores/
│   │   ├── llm/                 # Embedding + generation (Cohere/OpenAI)
│   │   └── vectordb/            # PGVector or Qdrant integration
│   ├── views/                   # Streamlit frontend
│
├── docker/                      # Docker for PGVector/Mongo setup
├── .env.example                 # Sample env config
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Create `.env` file

```bash
cp .env.example .env
```

Then fill in the values for:
- `COHERE_API_KEY`
- `POSTGRES_USERNAME`, `POSTGRES_PASSWORD`, `POSTGRES_PORT`, etc.
- `GENERATION_MODEL_ID` (e.g., `command-a-03-2025`)
- `EMBEDDING_MODEL_ID` (e.g., `embed-multilingual-light-v3.0`)

---

### 3. Run Alembic DB Migration

```bash
alembic upgrade head
```

This sets up the DB tables for projects, assets, and chunks.

---

## 🐳 Start Vector DB via Docker

```bash
cd docker
cp .env.example .env  # optional but recommended
docker compose up -d
```

---

## 🚀 Run the FastAPI Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

Go to:  
📄 Swagger API Docs: http://localhost:5000/docs  
Use this interface to test all endpoints: upload, process, index, search, and get answers

---

## 🖼️ Streamlit App (File Uploader & QA)

```bash
streamlit run src/views/file_upload_app.py
```

> Upload a medical document, and it will be chunked, embedded, indexed, and ready for querying.

---

## 🔎 RAG Workflow Overview

1. **Upload File** → TXT / PDF  
2. **Chunk Text** → via custom splitter  
3. **Generate Embeddings** → using Cohere’s multilingual model  
4. **Store Vectors** → in PGVector (Postgres)  
5. **User Query** → embedded and compared via **cosine similarity**  
6. **Retrieve Top Chunks** → most relevant context  
7. **LLM Prompt + Answer** → using Cohere LLM (`command-a-03-2025`)  

---

## 📦 API Endpoints (Examples)

| Endpoint                            | Description                           |
|-------------------------------------|---------------------------------------|
| `POST /api/v1/data/upload/{id}`     | Uploads a document                    |
| `POST /api/v1/data/process/{id}`    | Chunks & embeds document              |
| `POST /api/v1/nlp/index/push/{id}`  | Indexes all chunks into PGVector      |
| `POST /api/v1/nlp/index/search/{id}`| Search relevant chunks via query      |
| `POST /api/v1/nlp/index/answer/{id}`| Generates an LLM answer               |

---

## 🧠 Models Used

| Purpose     | Model                                  |
|-------------|----------------------------------------|
| Embeddings  | `embed-multilingual-light-v3.0`(Cohere)|
| Generation  | `command-a-03-2025` (Cohere)           |

Embeddings have **384 dimensions** representing text meaning in vector space.  
Similarity is calculated using **cosine similarity**.



