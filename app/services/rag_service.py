from __future__ import annotations

from collections import defaultdict

from llama_index.core import Settings as LlamaSettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import Settings
from app.ingestion.loader import load_documents
from app.retrieval.hybrid import KeywordIndex, reciprocal_rank_fusion


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.index: VectorStoreIndex | None = None
        self.nodes = []
        self.nodes_by_id: dict[str, object] = {}
        self.keyword_index = KeywordIndex()
        self.client = QdrantClient(url=settings.qdrant_url)

        if settings.openai_api_key:
            LlamaSettings.llm = OpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
            LlamaSettings.embed_model = OpenAIEmbedding(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
            )

    def index_documents(self) -> int:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required before indexing documents.")

        nodes = load_documents(self.settings.data_dir)
        if not nodes:
            self.index = None
            self.nodes = []
            self.nodes_by_id = {}
            self.keyword_index.build([])
            return 0

        if self.client.collection_exists(self.settings.qdrant_collection):
            self.client.delete_collection(self.settings.qdrant_collection)

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.qdrant_collection,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex(nodes, storage_context=storage_context)
        self.nodes = nodes
        self.nodes_by_id = {node.node_id: node for node in nodes}
        self.keyword_index.build([(node.node_id, node.get_content()) for node in nodes])
        return len(nodes)

    def _vector_results(self, question: str):
        if self.index is None:
            return []
        retriever = self.index.as_retriever(similarity_top_k=self.settings.top_k)
        return retriever.retrieve(question)

    def query(self, question: str) -> dict:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required before querying.")

        if self.index is None:
            self.index_documents()
        if self.index is None:
            return {"answer": "No documents are indexed yet.", "sources": []}

        vector_results = self._vector_results(question)
        vector_ids = [item.node.node_id for item in vector_results]
        keyword_ids = self.keyword_index.search(question, top_k=self.settings.top_k)
        fused_ids = reciprocal_rank_fusion([vector_ids, keyword_ids])[: self.settings.rerank_top_k]

        selected = [self.nodes_by_id[node_id] for node_id in fused_ids if node_id in self.nodes_by_id]
        if not selected:
            selected = [item.node for item in vector_results[: self.settings.rerank_top_k]]

        context_parts = []
        for node in selected:
            metadata = node.metadata or {}
            source = metadata.get("source_file") or metadata.get("file_name") or "unknown"
            context_parts.append(f"Source: {source}\n{node.get_content()}")

        prompt = (
            "Answer the user's question using only the provided context. "
            "If the context does not contain the answer, say that the information is not available. "
            "Do not invent facts.\n\n"
            f"Context:\n{'\n\n'.join(context_parts)}\n\n"
            f"Question: {question}\nAnswer:"
        )
        response = LlamaSettings.llm.complete(prompt)

        sources = []
        score_by_id = {item.node.node_id: item.score for item in vector_results}
        for node in selected:
            metadata = node.metadata or {}
            sources.append(
                {
                    "file": metadata.get("source_file") or metadata.get("file_name") or "unknown",
                    "score": score_by_id.get(node.node_id),
                }
            )

        return {"answer": str(response), "sources": sources}
