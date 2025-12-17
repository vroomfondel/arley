SRC: https://github.com/vroomfondel/arley/blob/main/postgrespgvector/Dockerfile_postgres

# PostgreSQL with pgvector Extension – Multi-Architecture Docker Image

This Docker image provides a production-ready PostgreSQL database server with the **pgvector extension** pre-compiled and installed, enabling high-performance vector similarity search for machine learning, embeddings, and retrieval-augmented generation (RAG) workflows.

## Overview

Built on the official PostgreSQL Docker image, this container extends the base functionality by compiling and installing the latest **pgvector** extension from source. It is designed for developers and teams working with vector embeddings, semantic search, and AI/ML applications requiring efficient similarity queries (cosine distance, L2 distance, inner product) on high-dimensional vectors.

The image includes a pre-configured German UTF-8 locale (`de_DE.UTF-8`, `LANG=de_DE.utf8`) to support internationalized applications, and is optimized for use in local development, continuous integration pipelines, and small to medium production deployments.

## Key Features

- **pgvector Extension**: Compiled from the official [pgvector GitHub repository](https://github.com/pgvector/pgvector) and installed into PostgreSQL, supporting vector data types and approximate nearest neighbor (ANN) indexes (IVFFLAT, HNSW).
- **Multi-Architecture Support**: Available for **amd64 (x86_64)** and **arm64 (aarch64)** platforms via Docker Buildx, ensuring compatibility across Intel/AMD servers, Apple Silicon (M1/M2/M3), and ARM-based cloud instances.
- **Versioned Tagging**: Images are tagged with precise version information (e.g., `18-trixie-pgvector-0.8.1`) based on PostgreSQL version, Debian base, and pgvector release, as well as a `:latest` convenience tag.
- **German Locale (de_DE.UTF-8)**: Pre-configured for projects requiring German language support and UTF-8 encoding.
- **Turnkey Operation**: Inherits all PostgreSQL environment variables and configuration options from the official image; pgvector can be enabled with a single `CREATE EXTENSION` command.

## Use Cases

- **Vector Databases for AI/ML**: Store and query embeddings from language models (e.g., OpenAI, Sentence Transformers, Ollama) for semantic search, question answering, and recommendation systems.
- **RAG (Retrieval-Augmented Generation)**: Complement LLM workflows with vector-based document retrieval to reduce hallucinations and improve grounding.
- **Similarity Search**: Perform KNN (k-nearest neighbors) queries on high-dimensional data (text embeddings, image features, etc.) with support for cosine similarity, L2 distance, and inner product metrics.
- **Development & CI/CD**: Lightweight, reproducible vector database for testing and development pipelines.

## Quick Start

### Running the Container

```shell script
docker run --rm -d \
  --name postgres-pgvector \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_USER=your_user \
  -e POSTGRES_DB=your_database \
  -p 5432:5432 \
  xomoxcc/postgreslocaled:latest
```


### Enabling the pgvector Extension

After the container starts, connect to your database and enable the extension:

```shell script
psql postgresql://your_user:your_password@127.0.0.1:5432/your_database \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```


### Example: Creating a Vector Table

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table with 768-dimensional embeddings
CREATE TABLE documents (
  id bigserial PRIMARY KEY,
  content text NOT NULL,
  embedding vector(768)
);

-- Create an approximate nearest neighbor index (IVFFLAT)
CREATE INDEX documents_embedding_idx 
  ON documents USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Optimize ANN recall
SET ivfflat.probes = 10;

-- Query: Find 5 most similar documents
SELECT id, content
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ..., 0.768]'::vector
LIMIT 5;
```


## Image Tags and Versioning

Images follow the naming convention:

- `xomoxcc/postgreslocaled:<postgres_version>-<debian_version>-pgvector-<pgvector_version>`
  - Example: `18-trixie-pgvector-0.8.1`
- `xomoxcc/postgreslocaled:latest` (tracks the most recent build)

**Default Versions** (as of latest build):
- PostgreSQL: 18
- Debian: trixie
- pgvector: Latest stable release from upstream

## Configuration and Environment Variables

This image inherits all configuration options from the [official PostgreSQL Docker image](https://hub.docker.com/_/postgres). Key environment variables:

- `POSTGRES_PASSWORD` (required): Superuser password
- `POSTGRES_USER` (optional, default: `postgres`): Superuser username
- `POSTGRES_DB` (optional, default: value of `POSTGRES_USER`): Default database name
- `POSTGRES_INITDB_ARGS` (optional): Additional arguments for `initdb`
- `PGDATA` (optional, default: `/var/lib/postgresql/data`): Data directory

See the [official Postgres image documentation](https://hub.docker.com/_/postgres) for advanced configuration (init scripts, custom configs, etc.).

## Performance Tuning

For optimal vector search performance:

1. **Choose the right index type**: `ivfflat` (faster build, moderate recall) or `hnsw` (slower build, better recall).
2. **Tune index parameters**: Adjust `lists` (IVFFLAT) or `m`/`ef_construction` (HNSW) based on dataset size.
3. **Set runtime parameters**: Use `SET ivfflat.probes = N;` to control query-time accuracy/speed tradeoff.
4. **Distance functions**: 
   - `<=>` (cosine distance) for normalized embeddings
   - `<->` (L2 distance) for Euclidean similarity
   - `<#>` (inner product) for dot-product similarity

Refer to the [pgvector documentation](https://github.com/pgvector/pgvector) for detailed tuning guidance.

## Architecture Support

- **amd64 (x86_64)**: Intel and AMD processors
- **arm64 (aarch64)**: Apple Silicon (M1/M2/M3), AWS Graviton, and other ARM64 platforms

Multi-arch images are built using Docker Buildx and pushed as manifest lists for automatic platform detection.

## Integration with Arley Project

This image was originally developed as an optional vector database backend for the [Arley RAG assistant project](https://github.com/vroomfondel/arley), providing an alternative to ChromaDB for vector storage in legal/document-oriented AI workflows. It can be used standalone or integrated with any application requiring PostgreSQL with vector capabilities.

## Building from Source

The Dockerfile and build scripts are available in the source repository. To build locally:

```shell script
git clone https://github.com/vroomfondel/arley.git
cd arley/postgrespgvector
./build_postgres_localed.sh onlylocal
```


For multi-architecture builds, see the `build_postgres_localed.sh` script and ensure Docker Buildx and QEMU/binfmt are configured.

## License

This image is provided under the same licensing terms as the parent Arley project (primarily LGPL where applicable). The underlying PostgreSQL software is licensed under the PostgreSQL License, and pgvector is licensed under the PostgreSQL License. See repository license files for details.

## Support and Contributions

- **Issues**: Report bugs or request features via the [GitHub repository](https://github.com/vroomfondel/arley/issues).
- **Documentation**: Refer to the [pgvector documentation](https://github.com/pgvector/pgvector) and [PostgreSQL documentation](https://www.postgresql.org/docs/) for detailed usage.

## ⚠️ Disclaimer

This is a development/experimental project. For production use, review security settings, customize configurations, and test thoroughly in your environment. Provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software. Use at your own risk.