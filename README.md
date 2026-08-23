HELLO EVERYONE, MYSELF ABU WAQAS, I HAVE MADE THIS RAG PROJECT.DETAILS REGARDING PROJECT ARE GIVEN BELOW:

# 🩺 MediBot — AI Medical Chatbot

An AI-powered medical chatbot that answers health-related questions using **Retrieval-Augmented Generation (RAG)**. It retrieves relevant information from medical PDFs and generates accurate responses using a large language model.

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend API** | FastAPI |
| **Frontend UI** | Streamlit |
| **LLM** | Groq (openai/gpt-oss-120b) |
| **Vector Store** | FAISS |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) |
| **Framework** | LangChain |

## Architecture

```
User → Streamlit UI (port 8501) → FastAPI Backend (port 8000) → FAISS + Groq LLM → Response
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/waqas-dev-hub/Medical-Chatbot.git
cd Medical-Chatbot
```

### 2. Create a virtual environment

```bash
conda create -p venv python==3.11 -y
conda activate venv/
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

- Get your Groq API key from [console.groq.com](https://console.groq.com)
- Get your HuggingFace token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 5. Add medical PDFs

Place your medical PDF files in the `data/` folder.

### 6. Build the vector store

```bash
python create_memory_for_llm.py
```

This processes the PDFs and creates the FAISS vector store in `vectorstore/`.

### 7. Run the application

Start the **FastAPI backend** (Terminal 1):

```bash
uvicorn app:app --reload
```

Start the **Streamlit frontend** (Terminal 2):

```bash
streamlit run medibot.py
```

### 8. Open the app

| Service | URL |
|---------|-----|
| Chat UI | http://localhost:8501 |
| API Docs (Swagger) | http://127.0.0.1:8000/docs |

## API Usage

You can also call the API directly:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is diabetes?"}'
```

**Response:**

```json
{
  "answer": "Diabetes is a chronic condition...",
  "source_documents": [
    {
      "page_content": "...",
      "metadata": {"source": "data/medical_book.pdf", "page": 42}
    }
  ]
}
```

## Project Structure

```
Medical-Chatbot/
├── app.py                    # FastAPI backend (RAG API)
├── medibot.py                # Streamlit frontend (Chat UI)
├── create_memory_for_llm.py  # Script to build FAISS vectorstore from PDFs
├── connect_memory_with_llm.py# Standalone CLI test script
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed)
├── .gitignore
├── data/                     # Medical PDF files
└── vectorstore/              # FAISS index (auto-generated)
```

