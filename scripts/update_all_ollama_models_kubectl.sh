#!/bin/bash

SCRIPTDIR=$(realpath $(dirname "$0"))
cd "${SCRIPTDIR}" || exit 123

source "${SCRIPTDIR}/include.sh"

# TODO put namespace etc. into ENV

models=$(kubectl --context="${KUBECTL_CONTEXT}" exec -n ollama -it deploy/ollama-deployment -- ollama list | awk 'NR>1 {print $1}')

for model in $models ; do
	echo "Updating model: $model"
	kubectl --context="${KUBECTL_CONTEXT}" exec -n ollama -it deploy/ollama-deployment -- ollama pull "$model"
	echo "-----"
done

