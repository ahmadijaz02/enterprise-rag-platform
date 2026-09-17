from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.rag_service import RAGService

app = FastAPI(title="Enterprise RAG Platform", version="1.0.0")
service = RAGService(get_settings())


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents/index")
def index_documents() -> dict:
    count = service.index_documents()
    return {"indexed_documents": count}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    return service.query(request.question)
