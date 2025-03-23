#!/bin/bash

SCRIPTDIR=$(dirname "$0")
OLLAMA_START_ENABLED=1
OLLAMA_MAX_START_RETRIES=2


check_ollama_running() {
  curl -o /dev/null -s http://127.0.0.1:11434
  local ollama_health=$?

  if [[ $ollama_health != 0 ]] ; then
    return 0
  fi
  return 1
}

ollama_check_running_or_start() {
  local ollama_running=-1
  check_ollama_running
  ollama_running=$?


  if [[ $ollama_running != 1 ]] ; then
    if [[ $OLLAMA_START_ENABLED == 0 ]] ; then
      echo "ollama is NOT running and I should not start it. blargh. bye."
      exit 123
    fi
    if ! [ -f "${SCRIPTDIR}"/startollama.sh ] ; then
      echo "ollama is NOT running, I should start \"${SCRIPTDIR}/startollama.sh\" - but that does not exist. blargh. bye."
      exit 127
    fi
  fi

  local loopcount=0
  while [[ $ollama_running != 1 && $OLLAMA_MAX_START_RETRIES > $loopcount ]] ; do
    # docker run --gpus=all -it --rm -v $(pwd)/ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    if [[ $loopcount == 0 ]] ; then
      echo STARTING OLLAMA...
      "${SCRIPTDIR}"/startollama.sh &
      local spawnedpid=$!
      # echo spawnedpid: $spawnedpid
      # wait  # for process to finish ?!
    fi

    echo sleeping 5s
    sleep 5

    (( loopcount++ ))
    check_ollama_running
    ollama_running=$?
  done
}



echo "first ensuring ollama (docker) is up2date"
docker pull ollama/ollama
echo

ollama_check_running_or_start

models=$(docker exec -it ollama ollama list | awk 'NR>1 {print $1}')
# models=$(docker run --network host -it --name ollama_cmd --rm ollama/ollama:latest list | awk 'NR>1 {print $1}')


for model in $models ; do
	echo "Updating model: $model"
	docker exec -it ollama ollama pull "$model"
	# docker run --network host -it --name ollama_cmd --rm ollama/ollama:latest pull "$model"
	echo "-----"
done

