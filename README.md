# Enterprise RAG Platform

A modular document retrieval platform for searching private business documents through a FastAPI service.

## Features

- PDF, DOCX, TXT and Markdown ingestion
- Metadata-aware chunking
- Semantic vector retrieval
- BM25 keyword retrieval
- Hybrid retrieval with reciprocal-rank fusion
- Optional cross-encoder reranking
- Source-aware responses
- REST API for indexing and querying
- Qdrant-backed vector storage
- Docker-ready local development

## Architecture

```text
Documents -> Ingestion -> Chunking -> Embeddings -> Qdrant
                                  |                    |
                                  +------ BM25 --------+
                                               |
                                               v
                                      Hybrid Retrieval
                                               |
                                               v
                                           Reranker
                                               |
                                               v
                                        Context Builder
                                               |
                                               v
                                        Answer Generator
                                               |
                                               v
                                      API + Sources
```

## Stack

Python 3.11+, FastAPI, LlamaIndex, Qdrant, BM25, Sentence Transformers, Pydantic Settings and Docker.

## Structure

```text
app/
  config.py
  ingestion/
  retrieval/
  services/
api/
data/sample/
tests/
.env.example
Dockerfile
docker-compose.yml
requirements.txt
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Start Qdrant:

```bash
docker compose up -d qdrant
```

Run the API:

```bash
uvicorn api.main:app --reload
```

Open `/docs` for the interactive API documentation.

## API

`GET /health` checks service status.

`POST /documents/index` indexes files from the configured data directory.

`POST /query` accepts:

```json
{"question":"What does the document say about leave policy?"}
```

The query response contains the generated answer and source metadata.

## Configuration

See `.env.example` for the available settings. The default local setup uses Qdrant and an OpenAI-compatible provider.

## Third-Party Components

This repository contains application code and uses third-party libraries as dependencies. LlamaIndex is used as an external framework/library; its source code is not included here. Third-party components remain subject to their respective licenses.

## License

MIT
