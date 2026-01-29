# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Arley is a RAG (Retrieval-Augmented Generation) assistant for answering legal questions via email. It monitors an IMAP mailbox, processes incoming questions through an LLM with document-grounded context from ChromaDB, and sends responses via SMTP. Supports German and English. Designed for local or small-cluster (K3s) deployment.

## Build & Development Commands

```bash
make install          # Create venv and install all dependencies
make tests            # Run pytest (via venv)
make lint             # Format code with black (120 char line length)
make isort            # Sort imports with isort
make tcheck           # Run mypy type checking on: *.py scripts dbaccess aux arley
make prepare          # Run tests + pre-commit hooks
make build            # Build Docker image (multi-arch)
make build-psql       # Build Postgres+pgvector image
```

Run a single test:
```bash
pytest tests/test_base.py
```

## CI Pipeline

GitHub Actions runs on push to main and PRs:
1. mypy type checking (`make tcheck`)
2. pytest
3. On success: multi-arch Docker build and push to `xomoxcc/arley`

## Architecture

### Entry Point and Main Loops

`main.py` dispatches to two independent processing loops:

- **IMAPLOOP** (`arley/emailinterface/imapadapter.py`) — Connects to IMAP, monitors for new emails via IDLE, parses them, persists to Postgres as `ArleyEmailInDB` records, moves to WORKING folder.
- **OLLAMALOOP** (`arley/emailinterface/ollamaemailreply.py`) — Polls Postgres for pending emails, detects language, retrieves RAG context from ChromaDB, generates LLM response via Ollama, sends reply via SMTP, updates status.

### Module Layout

- `arley/config.py` — Pydantic v2 BaseSettings with YAML file sources. Detects Kubernetes vs local deployment. Key env vars: `ARLEY_CONFIG_DIR_PATH`, `ARLEY_CONFIG_PATH`, `ARLEY_CONFIG_LOCAL_PATH`, `LOGURU_LEVEL`, `OLLAMA_BASE_HOST`, `CHROMADB_HOST`, `OLLAMA_MODEL`.
- `arley/llm/` — Ollama adapter (`ollama_adapter.py`), RAG orchestration (`ollama_chromadb_rag.py`), tool calling (`ollama_tools.py`), evaluation framework (`ollama_basic_evalrun.py`), Jinja2 prompt templates in `templates/v2/`.
- `arley/vectorstore/` — ChromaDB HTTP client (`chroma_adapter.py`), document ingestion pipeline (`importhelper.py` — handles DOCX, PDF, Markdown). `pgvector_adapter.py` is stubbed.
- `arley/emailinterface/` — IMAP/SMTP handling and email reply logic. Jinja2 response templates in `mailtemplates/`.
- `arley/dbobjects/` — SQLAlchemy models for email persistence (`emailindb.py`), Pydantic models for RAG documents (`ragdoc.py`).
- `arley/Helper.py` — Singleton metaclass and shared utilities.
- `dbaccess/` — Legacy SQLAlchemy layer (marked for phase-out, do not extend).

### Email Processing States

Emails flow through: `undefined` → `pending` → `working` → `processed` | `rejected` | `failed`, with corresponding IMAP folder moves (WORKING, WORKED, REJECTED, FAILED).

### Configuration

Two YAML files loaded by Pydantic settings: `config.yaml` (committed defaults) and `config.local.yaml` (local overrides with credentials, not committed). Environment variables can override any setting.

### External Services

Ollama (LLM inference), ChromaDB (vector store, HTTP API), PostgreSQL (email persistence), Redis (caching), IMAP/SMTP server.

## Code Style

- **Python 3.14** required
- **black** with 120-character line length
- **isort** for import sorting
- **mypy** strict type checking (follows imports silently, excludes .venv/tests)
- **loguru** for logging (not stdlib logging)

## Docker

Multi-arch (amd64/arm64) images built via buildx. Base image: `xomoxcc/pythonpandasmultiarch`. Locale: `de_DE.UTF-8`, timezone: `Europe/Berlin`. Non-root user `pythonuser` (UID 1234). Entrypoint uses `tini` for signal handling.

### Postgres + pgvector Image

`postgrespgvector/` contains a separate Docker image (`xomoxcc/postgreslocaled`) that compiles pgvector from source into the official Postgres image. On x86_64, the build pins OPTFLAGS to `x86-64-v3` (AVX2) to avoid AVX-512 auto-detection that causes SIGILL on older CPUs; on arm64 it uses default flags. The build script (`build_postgres_localed.sh`) supports local-only builds (`./build_postgres_localed.sh onlylocal`) and multi-arch build+push. Registry credentials are sourced from `scripts/include.local.sh` (not committed). See `postgrespgvector/README.md` for full details.