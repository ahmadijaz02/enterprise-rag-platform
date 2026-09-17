from llama_index.core.node_parser import SentenceSplitter


def get_chunker(chunk_size: int = 700, chunk_overlap: int = 100) -> SentenceSplitter:
    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
