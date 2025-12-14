[![mypy and pytests](https://github.com/vroomfondel/arley/actions/workflows/mypynpytests.yml/badge.svg)](https://github.com/vroomfondel/arley/actions/workflows/mypynpytests.yml)
[![BuildAndPushMultiarch](https://github.com/vroomfondel/arley/actions/workflows/buildmultiarchandpush.yml/badge.svg)](https://github.com/vroomfondel/arley/actions/workflows/buildmultiarchandpush.yml)
![Cumulative Clones](https://img.shields.io/endpoint?logo=github&url=https://gist.githubusercontent.com/vroomfondel/e9ebbb132f37e56b5e95d42d36bd2062/raw/arley_clone_count.json)
[![Docker Pulls](https://img.shields.io/docker/pulls/xomoxcc/arley?logo=docker)](https://hub.docker.com/r/xomoxcc/arley/tags)

[![Gemini_Generated_Image_arley_y0svvgy0svvgy0sv_250x250.png](Gemini_Generated_Image_arley_y0svvgy0svvgy0sv_250x250.png)](https://hub.docker.com/r/xomoxcc/arley/tags)

# WIP — Arley

Attempt of setting up an LLM RAG‑assistant mainly for answering legal‑related questions.

Based on the current generation of LLMs (e.g., Mixtral, Llama 3.x), this project explores a retrieval‑augmented generation (RAG) assistant focused on legal use cases. The aim is to reduce hallucinations via strong grounding (LangChain/LlamaIndex, document context, vector stores) and, where appropriate, light web verification. The intent is to run locally and/or in a small cluster and experiment with distributed inference for larger models.

The primary user interface is planned via email for compliance and ease of adoption.

Note: This repository is under active development (WIP).

## Overview

- Language: Python (project code in the arley package; entry module main.py)
- Frameworks/Libraries: LangChain, LlamaIndex, Loguru, Pydantic v2 (pydantic-settings), httpx, redis, SQLAlchemy, psycopg2-binary, BeautifulSoup4, Jinja2, etc. See requirements.txt.
- Vector DB options: ChromaDB (HTTP) and Postgres with pgvector (see Dockerfile_postgres and dbaccess/).
- LLM runtime: Ollama (local or containerized). Some scripts help run/update models.
- Containerization: Docker multi-stage image; k3s/Kubernetes manifests present (kubernetes/).
- CI: GitHub Actions for tests and multi-arch builds.

## Requirements

Hardware (depending on chosen models):
- GPU recommended (NVidia; CUDA) for larger models
- Fast NVMe storage for model weights and I/O
- Sufficient RAM/VRAM and CPU

Software:
- Python 3.14 (Makefile targets expect python3.14; local venv recommended)
- Docker (and optionally NVIDIA Container Toolkit for GPU pass-through)
- Ollama (local binary or Docker image)
- Optional: Kubernetes/k3s for cluster deployment

## Project Structure

- main.py — CLI entry that dispatches IMAPLOOP and OLLAMALOOP
- arley/
  - config.py, config.yaml, config.local.yaml — configuration and settings (pydantic v2)
  - emailinterface/ — IMAP processing and Ollama email replies
  - llm/ — LLM integrations and helpers (Ollama, evaluations, tools)
  - vectorstore/ — ChromaDB adapters and import helpers
  - dbobjects/ — ORM/data models used with Postgres
  - Helper.py and other utilities
- dbaccess/ — DB related helpers
- scripts/ — helper scripts (Ollama start/update, Postgres setup, etc.)
- kubernetes/ — manifests for cluster operation (stripped-down configs)
- tests/ — pytest scaffolding
- Dockerfile, Dockerfile_postgres — container builds
- requirements.txt, requirements-dev.txt — dependencies

## Setup (Local Development)

Using Make:
- make install — prepare venv and install dev requirements
- make tests — run pytest
- make lint — run black
- make isort — sort imports
- make tcheck — run mypy on key paths
- make build — build Docker image via build.sh

Manual venv:
- python3.14 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements-dev.txt

## Running

Local Python:
- IMAP processing loop: python main.py IMAPLOOP
- Ollama reply loop: python main.py OLLAMALOOP

Prerequisites for the above include a reachable IMAP/SMTP server (see arley/config.yaml) and an accessible Ollama host (see env vars below). Adjust configuration paths as needed.

Docker (application image):
- Build: make build (calls ./build.sh)
- Run an interactive container: docker run --rm -it --network=host xomoxcc/arley:latest /bin/bash
  - Inside the container, activate venv and run: source /app/python_venv.sh && python /app/main.py OLLAMALOOP (or IMAPLOOP)
  - Note: The default CMD tails logs; you must run the desired loop explicitly.

Docker (Postgres with pgvector, de_DE locale):
- Dockerfile_postgres builds a utility image. See scripts/3_createdb_arley.sh for example initialization. Image publishing is referenced in badges.

Ollama helper (local or Docker):
- scripts/startollama.sh — starts Ollama with configurable env vars (KV cache type, context length, etc.). Requires NVIDIA toolkit for GPU if using Docker.
- scripts/update_all_ollama_models.sh — pulls/updates configured models on a local Ollama instance.
- scripts/update_all_ollama_models_kubectl.sh — updates models in a k8s context.

Kubernetes/k3s:
- There are manifests under kubernetes/. Integrations are WIP and require customization for your cluster. TODO: Provide minimal values and instructions.

Historical note: scripts/runcompare_to_log.sh was noted as not working as of 2025‑03‑23. Current status: TODO/unknown.

## Configuration and Environment Variables

Configuration files:
- arley/config.yaml — base configuration (committed; contains example values)
- arley/config.local.yaml — local overrides (in-repo example; provide your own with secrets redacted)

Config path overrides (env vars):
- ARLEY_CONFIG_DIR_PATH — directory containing config.yaml and config.local.yaml
- ARLEY_CONFIG_PATH — path to config.yaml
- ARLEY_CONFIG_LOCAL_PATH — path to config.local.yaml

Logging:
- LOGURU_LEVEL — default DEBUG; set INFO/WARNING/ERROR in production

Ollama:
- OLLAMA_BASE_HOST — default 127.0.0.1
- OLLAMA_PORT — default 11434
- OLLAMA_BASE_HOST_CLUSTER — default ollama.ollama.svc.cluster.local
- Application-level model defaults are defined in arley/config.yaml under ollama. You can override via config files.

ChromaDB:
- CHROMADB_HOST — default 127.0.0.1
- CHROMADB_HOST_CLUSTER — default chromadb.chromadb.svc.cluster.local
- CHROMADB_PORT — default 8000
- CHROMADB_DEFAULT_COLLECTIONNAME — default arley
- CHROMADB_OLLAMA_EMBED_MODEL — default nomic-embed-text:latest

PostgreSQL:
- Settings primarily from config files (host, port, username, password, dbname). Optional URL may be constructed as needed.

Redis:
- host/port via config; host_in_cluster optional

Docker image build metadata (set during docker build):
- GITHUB_REF, GITHUB_SHA, BUILDTIME — injected by CI/build scripts; used for logging

Ollama runtime script variables (scripts/startollama.sh):
- OLLAMA_MODELDIR (default $HOME/ollama_models)
- OLLAMA_RUNDIR (default $HOME/ollama)
- OLLAMA_ALLOW_LOCAL_EXECUTABLE (0/1)
- OLLAMA_NUM_PARALLEL (default 1)
- OLLAMA_FLASH_ATTENTION (default 1)
- OLLAMA_KV_CACHE_TYPE (default q8_0)
- OLLAMA_CONTEXT_LENGTH (default 8192)

## Tests

- Pytest configuration in pytest.ini (discovers tests/*.py)
- Run: make tests or pytest .
- Type checks: make tcheck (mypy)
- Linting/format: make lint (black), make isort
- Pre-commit (optional): make commit-checks, make prepare

## Scripts (selected)

- scripts/startollama.sh — start Ollama (local binary or Docker)
- scripts/update_all_ollama_models.sh — update models via Ollama API
- scripts/update_all_ollama_models_kubectl.sh — same for k8s
- scripts/3_createdb_arley.sh — example Postgres init
- scripts/run_openwebui.sh — helper to run OpenWebUI (requires local adjustments)
- scripts/doimport.sh — import helper (see arley/vectorstore/importhelper.py)

## License

This project is licensed under the LGPL where applicable/possible — see [LICENSE.md](LICENSE.md). Some files/parts may be governed by other licenses and/or licensors, such as [MIT](LICENSEMIT.md) | [GPL](LICENSEGPL.md) | [LGPL](LICENSELGPL.md). Please also check file headers/comments.

## Acknowledgments

See inline comments in the codebase for inspirations and references.

## TODOs / Open Items

- Document minimal kubernetes/ deployment steps and values
- Provide sample config.local.yaml with redacted values and instructions
- Add end‑to‑end run examples (local + Docker) with ChromaDB/Postgres
- Revisit distributed inference (Ollama vs. vLLM) and document chosen path
- Confirm status of scripts/runcompare_to_log.sh and either fix or remove

## Quantizations / models

References and resources:
- https://huggingface.co/nitsuai/Meta-Llama-3-70B-Instruct-GGUF
- https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9
- https://artificialanalysis.ai/models/llama-3-instruct-70b/providers

Tuning / training:
- https://unsloth.ai/blog/llama3-1
- https://github.com/unslothai/unsloth

Notes:
- Some logs may be created in logdir/
- This code predates native tool-call support in Ollama; history was squashed for public release.


## ⚠️ Disclaimer

This is a development/experimental project. For production use, review security settings, customize configurations, and test thoroughly in your environment. Provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software. Use at your own risk.