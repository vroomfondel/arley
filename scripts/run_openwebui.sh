#!/bin/bash

SCRIPTDIR=$(dirname "$0")
source "${SCRIPTDIR}/include.sh"

docker run \
	--rm \
	-it \
	-p 3000:8080 \
	-e OLLAMA_BASE_URL="${REMOTE_OLLAMA_BASE_URL}" \
	-e CHROMADB_HOST="${REMOTE_CHROMADB_HOST}" \
	-e CHROMADB_PORT="${REMOTE_CHROMADB_PORT}" \
	-e WEBUI_SECRET_KEY="${OLLAMA_WEBUI_SECRET_KEY}" \
	--add-host="${DOCKER_OLLAMA_ADD_REMOTE_HOST}" \
	--add-host="${DOCKER_CHROMADB_ADD_REMOTE_HOST}" \
	-v "${OLLAMA_WEBUI_DATA_DIR}":/app/backend/data \
	--name open-webui \
	ghcr.io/open-webui/open-webui:main

