import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from dotenv import load_dotenv

load_dotenv()


rag_chain = None

class ChatRequest(BaseModel):
    query: str


class SourceDocument(BaseModel):
    page_content: str
    metadata: dict


class ChatResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain

    DB_FAISS_PATH = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.5,
        max_tokens=512,
    )

    retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            ("human", "{input}"),
        ]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(
        db.as_retriever(search_kwargs={"k": 3}), combine_docs_chain
    )

    print("[OK] RAG chain ready")
    yield
    print("[STOP] Shutting down")

app = FastAPI(
    title="MediBot API",
    description="Medical chatbot powered by RAG (FAISS + Groq LLM)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "MediBot API is running. POST your query to /chat"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    response = rag_chain.invoke({"input": request.query})

    source_documents = [
        SourceDocument(
            page_content=doc.page_content[:300],
            metadata=doc.metadata,
        )
        for doc in response["context"]
    ]

    return ChatResponse(
        answer=response["answer"],
        source_documents=source_documents,
    )
