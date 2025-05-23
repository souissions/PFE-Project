# 🧠 RAG – Medical Retrieval-Augmented Generation System

## 🚧 Current Status: Work in Progress
This project is under active development. The codebase and documentation are subject to ongoing changes and improvements.

---

## 📝 Overview
This project implements a custom **RAG (Retrieval-Augmented Generation)** system for intelligent document understanding and question answering in the medical domain. It now features a modular **LangGraph agent workflow** for advanced multi-step reasoning, intent classification, and triage, in addition to classic RAG retrieval.

### Core Technologies
- ⚙️ **FastAPI** for backend APIs
- 🧠 **Cohere**, **OpenAI**, **Hugging Face**, and **Google** for embeddings & LLM answer generation
- 📊 **PGVector** as the vector similarity database
- 🎛️ **Streamlit** interface for file upload and user interaction
- 🕸️ **LangGraph** for agent workflow orchestration (triage, intent, multi-step reasoning)

> **Note:** The system is designed using a factory pattern, enabling support for multiple providers. 

---

## 🔧 Features
- ✅ Upload `.txt` or `.pdf` medical documents
- ✅ Automatic chunking of text into digestible segments
- ✅ Generation of **384-dimensional embeddings** using Cohere
- ✅ Storage of vectors in **PGVector** for semantic retrieval
- ✅ LLM-powered answer generation based on query + retrieved context
- ✅ **LangGraph agent workflow** for:
  - Intent classification (triage, info request, off-topic)
  - Symptom gathering and follow-up
  - Medical relevance checking
  - Multi-step reasoning and explanation refinement

---

## ⚙️ Setup & Installation

### 1. Install dependencies
```bash
conda create -n rag-env python=3.10
conda activate rag-env
pip install -r src/requirements.txt
```

### 2. Create `.env` file
```bash
cp .env.example .env
```
Then fill in the missing values

### 3. Run Alembic DB Migration
```bash
alembic upgrade head
```
This sets up the DB tables for projects, assets, and chunks.

---

## 🐳 Start Vector DB via Docker (WSL recommended)
```bash
cd docker
cp .env.example .env  # optional but recommended
docker compose up -d
```

---

## 🚀 Run the FastAPI Server (inside WSL/conda)
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

## 🔎 RAG & LangGraph Workflow Overview
1. **Upload File** → TXT / PDF  
2. **Chunk Text** → via custom splitter  
3. **Generate Embeddings** → using Cohere’s multilingual model  
4. **Store Vectors** → in PGVector (Postgres)  
5. **User Query** → embedded and compared via **cosine similarity**  
6. **Retrieve Top Chunks** → most relevant context  
7. **LLM Prompt + Answer** → using Cohere LLM (`command-a-03-2025`)  
8. **LangGraph Agent** → classifies intent, gathers symptoms, checks relevance, and generates/refines explanations


---

## 🕸️ About LangGraph Integration
LangGraph enables a flexible, multi-step agent workflow for:
- Intent classification (triage, info, off-topic)
- Symptom gathering with follow-up questions
- Medical relevance checking
- Final analysis, explanation evaluation, and refinement
- Modular, extensible state machine for complex medical QA

---


