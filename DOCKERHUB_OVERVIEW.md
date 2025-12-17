# Arley - Legal RAG Assistant with LLM

A retrieval-augmented generation (RAG) assistant focused on answering legal-related questions using modern LLMs (Mixtral, Llama 3.x, etc.) with strong grounding through vector stores and document context.

## 🎯 Purpose

Arley reduces LLM hallucinations through retrieval-augmented generation, combining:
- Vector databases (ChromaDB, Postgres with pgvector)
- Document context and semantic search
- Local or distributed LLM inference via Ollama
- Email-based user interface for compliance and ease of adoption

**Status:** Work in progress (WIP) - under active development

## 🚀 Quick Start

```shell script
# Pull the image
docker pull xomoxcc/arley:latest

# Run 
please make sure to have create a local config file ("config.local.yaml") like [this](https://github.com/vroomfondel/arley/blob/main/arley/config.yaml).<br />
docker run --rm -it --name arleyephemeral --network=host -v $(pwd)/config.local.yaml:/app/arley/config.local.yaml xomoxcc/arley:latest /app/python_venv.sh main.py OLLAMALOOP<br />
docker run --rm -it --name arleyimaploopephemeral --network=host -v $(pwd)/config.local.yaml:/app/arley/config.local.yaml xomoxcc/arley:latest /app/python_venv.sh main.py IMAPLOOP
```


## 📦 What's Inside

- **Python 3.14** runtime with virtualenv
- **LangChain/LlamaIndex** for RAG orchestration
- **Ollama integration** for local LLM inference
- **Vector store adapters** (ChromaDB HTTP, Postgres pgvector)
- **Email interface** (IMAP/SMTP) for user interaction
- **Multi-arch support** (amd64, arm64)

## 🔧 Requirements

**Dependencies (provide separately or in-cluster):**
- Ollama service (local or containerized)
- ChromaDB or Postgres with pgvector
- IMAP/SMTP mail server (for email interface)
- Optional: Redis for caching

**Hardware recommendations:**
- GPU (NVIDIA with CUDA) for larger models
- Fast NVMe storage
- Sufficient RAM/VRAM

## ⚙️ Configuration

Configure via environment variables or mounted config files:

**Ollama:**
- `OLLAMA_BASE_HOST` (default: 127.0.0.1)
- `OLLAMA_PORT` (default: 11434)
- `OLLAMA_BASE_HOST_CLUSTER` (for k8s: ollama.ollama.svc.cluster.local)

**ChromaDB:**
- `CHROMADB_HOST` / `CHROMADB_PORT`
- `CHROMADB_DEFAULT_COLLECTIONNAME`

**Logging:**
- `LOGURU_LEVEL` (DEBUG/INFO/WARNING/ERROR)

**Config files:** Mount `config.yaml` and `config.local.yaml` or set `ARLEY_CONFIG_DIR_PATH`

## 🏗️ Architecture

- **main.py** - CLI dispatcher (IMAPLOOP / OLLAMALOOP)
- **arley/** - Core package (config, email interface, LLM adapters, vector stores)
- **dbaccess/** - Database helpers (SQLAlchemy ORM)
- **scripts/** - Ollama management, DB setup utilities

## 🐳 Related Images

- **xomoxcc/postgreslocaled** - Postgres with pgvector and de_DE locale

## 📊 CI/CD

[![BuildAndPushMultiarch](https://github.com/vroomfondel/arley/actions/workflows/buildmultiarchandpush.yml/badge.svg)](https://github.com/vroomfondel/arley/actions/workflows/buildmultiarchandpush.yml)

Automated multi-architecture builds via GitHub Actions with mypy type checking and pytest validation.

## 📄 License

LGPL where applicable. Some components may be under MIT/GPL - see repository for details.

## 🔗 Links

- [GitHub Repository](https://github.com/vroomfondel/arley)
- [Documentation](https://github.com/vroomfondel/arley/blob/main/README.md)
- [Kubernetes Manifests](https://github.com/vroomfondel/arley/tree/main/kubernetes)

## ⚠️ Disclaimer

This is a development/experimental project. For production use, review security settings, customize configurations, and test thoroughly in your environment. Provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software. Use at your own risk.
