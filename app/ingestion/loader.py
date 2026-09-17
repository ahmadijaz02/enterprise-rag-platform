from pathlib import Path

from llama_index.core import SimpleDirectoryReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_documents(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        return []

    reader = SimpleDirectoryReader(input_files=files, filename_as_id=True)
    return reader.load_data()
