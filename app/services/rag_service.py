from __future__ import annotations

from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from app.config import Settings
from app.ingestion.loader import load_documents


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.index: VectorStoreIndex | None = None
        if settings.openai_api_key:
            LlamaSettings.llm = OpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
            LlamaSettings.embed_model = OpenAIEmbedding(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
            )

    def index_documents(self) -> int:
        documents = load_documents(self.settings.data_dir)
        if not documents:
            self.index = None
            return 0
        self.index = VectorStoreIndex.from_documents(documents)
        return len(documents)

    def query(self, question: str) -> dict:
        if self.index is None:
            self.index_documents()
        if self.index is None:
            return {"answer": "No documents are indexed yet.", "sources": []}

        engine = self.index.as_query_engine(similarity_top_k=self.settings.top_k)
        response = engine.query(question)
        sources = []
        for node in getattr(response, "source_nodes", []):
            metadata = node.node.metadata or {}
            sources.append(
                {
                    "file": metadata.get("file_name") or metadata.get("filename"),
                    "score": node.score,
                }
            )
        return {"answer": str(response), "sources": sources}
