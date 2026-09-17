from pathlib import Path

from llama_index.core import SimpleDirectoryReader

from app.ingestion.chunker import get_chunker

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_documents(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        return []

    reader = SimpleDirectoryReader(input_files=files, filename_as_id=True)
    documents = reader.load_data()
    for document in documents:
        document.metadata["source_file"] = document.metadata.get("file_name", document.doc_id)

    return get_chunker().get_nodes_from_documents(documents)
