

# 📑 Airtable Extension RAG Backend
```markdown
This is a **FastAPI backend** for a Policy Retrieval-Augmented Generation (RAG) system.  
It integrates with **Airtable** via a webhook to listen for table updates and automatically embed documents into a vector store for retrieval.


## 📂 Project Structure


├───fast-api-backend
    ├── code\_backups/        # Old app files and Jupyter notebooks
    ├── config.py            # Centralized environment & app configuration
    ├── core/                # Core logic (Airtable, LLMs, embeddings, PDF handling)
    ├── main.py              # FastAPI entrypoint
    ├── models/              # Pydantic models (e.g., query request schema)
    ├── requirements.txt     # Python dependencies
    ├── routes/              # API routes (ask, search, webhook)
    ├── state/               # Vector index storage (e.g., FAISS)
    ├── utils/               # Helper functions (e.g., similarity search)
    └── venv/                # Virtual environment

````

---

## ⚙️ Features

- **Webhook Listener**: Listens for Airtable table updates.
- **Automatic Embedding**: Ingests and embeds updated documents into a vector store (FAISS).
- **Search & Ask Endpoints**: Query the embedded policies via semantic search or LLM.
- **Configurable Setup**: Environment variables are centralized in `config.py`.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <repo>
cd Chrome-Extension-Airtable/fast-api-backend
````

### 2. Create and Activate Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Update **`config.py`** with your keys and environment variables:

* Airtable API Key
* Airtable Base/Table IDs
* Vector DB settings
* LLM API keys (e.g., OpenAI, Gemini)

### 5. Run the Backend

```bash
uvicorn main:app --reload
```

The server will start at:
👉 `http://127.0.0.1:8000`

---

## 📡 API Routes

* `POST /ask` → Ask a question against the embedded policies.
* `GET /search` → Search policies in the vector database.
* `POST /webhook` → Webhook endpoint to receive Airtable updates and trigger embeddings.

---

## 🗄️ Vector Store

* Uses **FAISS** (default) stored under `state/faiss_index`.
* New documents from Airtable are automatically embedded and added.

---

## 🛠️ Development Notes

* Backup code and experiments are under `code_backups/`.
* Compiled Python cache files are ignored.
* Keep environment secrets **only in `config.py` or `.env` (if extended)**.

---


