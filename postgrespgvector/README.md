[![Docker Pulls](https://img.shields.io/docker/pulls/xomoxcc/postgreslocaled?logo=docker&label=PGVectorPostgres%20docker%20pulls)](https://hub.docker.com/r/xomoxcc/postgreslocaled/tags)

![PG + pgvector](Gemini_Generated_Image_postgres_pgvector_vbw0gsvbw0gsvbw0_250x250.png)

PostgreSQL with pgvector (Docker)

This directory builds a Docker image based on the official Postgres image with the pgvector extension compiled and installed. It is used by the Arley project as an optional vector database backend for RAG workflows, complementing the default ChromaDB option. The image also sets up a German UTF‑8 locale (`de_DE.UTF-8`) commonly used in this repository.

Why this is useful
- Turnkey Postgres with pgvector for local development, CI, or small deployments.
- Multi‑arch builds (amd64 and arm64) via Docker Buildx, with automated tagging based on upstream pgvector releases.
- Matches Arley expectations for locale and extension availability; integrates with `dbaccess/` and vector‑store code paths.

Contents
- Dockerfile_postgres — builds from `postgres:<version>-<debian>` and compiles pgvector from source.
- build_postgres_localed.sh — helper to build/push single‑arch and multi‑arch images, with optional registry login and QEMU/binfmt setup.
- docker-config/ (created at build time) — used for Docker auth when pushing (not committed).

Resulting image and tags
- Base image: `postgres:${postgres_version}-${debian_version}` (defaults set in the Dockerfile/script: Postgres 18, Debian trixie).
- pgvector built from: `https://github.com/pgvector/pgvector` at `pgvector_version` (latest tag by default).
- Produced tags (by the build script):
  - `xomoxcc/postgreslocaled:${postgres_version}-${debian_version}-pgvector-${pgvector_version_without_v}`
  - plus `:latest` on the same repository for convenience

Prerequisites
- Docker with Buildx enabled.
- For cross‑arch: binfmt/QEMU; the script can install via `tonistiigi/binfmt` (requires privileged Docker).
- Optional: Docker registry credentials (stored locally via `scripts/include.local.sh`).

Security and configuration
The build helper sources `../scripts/include.sh`, which may in turn load a local, untracked file such as `scripts/include.local.sh`. Keep secrets out of version control and use the local include file for credentials.

Example `scripts/include.local.sh` (do not commit):
```
export DOCKER_TOKENUSER="your-docker-username"
export DOCKER_TOKEN="<your-access-token>"
```

Quick start
1) Build locally (no push):
```
./build_postgres_localed.sh onlylocal
```
This creates a local image tagged like `xomoxcc/postgreslocaled:18-trixie-pgvector-<x.y.z>` and also adds `:latest`.

2) Multi‑arch build and push (amd64 + arm64):
```
./build_postgres_localed.sh
```
The script will:
- Perform `docker login` if credentials are available via the include file.
- Ensure Buildx builder `mbuilder` exists (and install binfmt if needed).
- Build a local image (logging to `docker_build_psql_local.log`) and then build+push the multi‑arch variant.

3) Run a container locally
```
docker run --rm -it \
  -e POSTGRES_PASSWORD=arley \
  -e POSTGRES_USER=arley \
  -e POSTGRES_DB=arley \
  -p 5432:5432 \
  xomoxcc/postgreslocaled:latest
```
The image inherits Postgres defaults; adjust `POSTGRES_*` env vars as needed.

Enabling pgvector inside the database
After the server is up, connect and enable the extension in the target database:
```
psql postgresql://arley:arley@127.0.0.1:5432/arley -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Basic usage example (768‑dim vectors, cosine distance)
```
-- Enable extension once per database
CREATE EXTENSION IF NOT EXISTS vector;

-- Example table for documents with 768‑dimensional embeddings
CREATE TABLE IF NOT EXISTS documents (
  id bigserial PRIMARY KEY,
  content text NOT NULL,
  embedding vector(768)
);

-- Optional ANN index (IVFFLAT) for cosine distance
CREATE INDEX IF NOT EXISTS documents_embedding_idx
  ON documents USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Parameter to improve recall with IVFFLAT
SET ivfflat.probes = 10;

-- KNN search example (top 5)
SELECT id, content
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;
```

Locale and environment
- The image sets up `de_DE.UTF-8` and exports `LANG=de_DE.utf8` to match broader project expectations.
- Build metadata is injected as `BUILDTIME` via `--build-arg`.

Integration in the Arley project
- Purpose: Provide a Postgres+pgvector option for vector storage and similarity search in RAG flows.
- Where it fits: The main project refers to Postgres in `dbaccess/` and mentions pgvector in the root `README.md`. Use this image if you prefer Postgres over ChromaDB.
- Connection: Configure host/port/user/password/db in Arley’s config files (see project root README, section “Configuration and Environment Variables”).
- Operational tip: Ensure the `vector` extension is enabled in the application database before running vector queries.

Troubleshooting
- Missing credentials for push: Create `scripts/include.local.sh` with `DOCKER_TOKENUSER` and `DOCKER_TOKEN`; the build script will attempt `docker login` if `docker-config/config.json` is absent.
- Buildx/builder errors: Ensure Docker Buildx is available and that you can install binfmt (may require privileged Docker on the host).
- Extension not found: Run `CREATE EXTENSION vector;` in the target DB; verify the image tag includes `pgvector` and that `pgvector` shared objects are installed in the container (`CREATE EXTENSION` is the supported way).
- Performance: Tune `ivfflat.probes`, index `lists`, and consider `vector_l2_ops`/`vector_ip_ops` depending on your distance metric.

License
See the repository’s license files in the project root (`LICENSE.md`, `LICENSEMIT.md`, `LICENSEGPL.md`, `LICENSELGPL.md`).

## ⚠️ Note

This is a development/experimental project. For production use, review security settings, customize configurations, and test thoroughly in your environment. Provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software. Use at your own risk.