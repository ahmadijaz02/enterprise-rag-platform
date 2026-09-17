from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Enterprise RAG Platform",
    version="1.1.1",
    description="Document ingestion and grounded question answering over a private document collection.",
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


def get_service():
    from app.config import get_settings
    from app.services.rag_service import RAGService

    return RAGService(get_settings())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents/index")
def index_documents() -> dict:
    try:
        count = get_service().index_documents()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Document indexing failed.") from exc
    return {"indexed_documents": count}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return get_service().query(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Query processing failed.") from exc
