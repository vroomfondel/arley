DOCKER_USERNAME="arley@somewhere.com"
DOCKER_PASSWORD="someweirdpassword"

DOCKER_TOKENUSER="dockerhubtokenuser"
DOCKER_TOKEN="dockerhubtokenforthatuser"

KUBECTL_CONTEXT="arley@local"

REMOTE_OLLAMA_BASE_URL="http://ollama.intra.somewhere.com"
DOCKER_OLLAMA_ADD_REMOTE_HOST="ollama.intra.somewhere.com:10.0.0.1"

OLLAMA_WEBUI_SECRET_KEY="webuisecretkey"

OLLAMA_WEBUI_DATA_DIR="${HOME}/open-webui/data"

REMOTE_CHROMADB_HOST="chromadb.intra.somewhere.com"
REMOTE_CHROMADB_PORT="80"
DOCKER_CHROMADB_ADD_REMOTE_HOST="chromadb.intra.somewhere.com:10.0.0.1"

declare -a DOCUMENT_DIRS=(/home/arley/Documents/{NDA,Bauver*}/*)

declare -a include_local_sh
include_local_sh[0]="$(dirname "$0")/include.local.sh"
include_local_sh[1]="$(dirname "$0")/scripts/include.local.sh"
include_local_sh[2]="$(dirname "$0")/../scripts/include.local.sh"
found=false

for path in "${include_local_sh[@]}"; do
  if [ -e "${path}" ]; then
    echo "${path} will be read..."
    source "${path}"
    found=true
    break
  fi
done

if [ "$found" = false ]; then
  echo "No include.local.sh file[s] found."
fi
