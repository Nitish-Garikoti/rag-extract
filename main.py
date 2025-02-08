import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.chain import answer_the_question, create_model_chain
from rag.loader import load_documents
from rag.vectorstore import create_pinecone_index, create_vector_db, get_embeddings
from utils.config import settings

app = FastAPI(
    title="TIFIN RAG System",
    description="RAG-powered Q&A over TIFIN financial documents",
    version="0.1.0",
)

_retriever_cache: dict = {}


def get_retriever(
    model_name: str,
    index_name: str = "pinecone-vdb-tifin-default",
    k: int = settings.default_k,
    search_type: str = settings.default_search_type,
):
    cache_key = f"{index_name}_{search_type}_{k}"
    if cache_key in _retriever_cache:
        return _retriever_cache[cache_key]

    docs = load_documents(settings.pdf_path)
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=300)
    chunks = splitter.split_documents(docs)

    create_pinecone_index(index_name)
    embeddings = get_embeddings()
    vector_db = create_vector_db(chunks, embeddings, index_name)
    retriever = vector_db.as_retriever(search_type=search_type, search_kwargs={"k": k})

    _retriever_cache[cache_key] = retriever
    return retriever


class QueryRequest(BaseModel):
    question: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    k: Optional[int] = None
    search_type: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    model_used: str
    contexts: list[str]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    model_name = request.model or settings.default_model
    temperature = request.temperature if request.temperature is not None else settings.default_temperature
    k = request.k or settings.default_k
    search_type = request.search_type or settings.default_search_type

    try:
        retriever = get_retriever(model_name=model_name, k=k, search_type=search_type)
        chain = create_model_chain(retriever, model_name=model_name, temperature=temperature)
        result = answer_the_question(chain, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=request.question,
        answer=result["response"],
        model_used=model_name,
        contexts=[ctx.page_content for ctx in result["context"]],
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    return {
        "default": settings.default_model,
        "available": [
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
        ],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
