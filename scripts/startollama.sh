#!/bin/bash

cd $(dirname $0) || exit 1

# set OLLAMA_KV_CACHE_TYPE   # https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-set-the-quantization-type-for-the-kv-cache
OLLAMA_KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE:-q8_0}
# f16 q8_0 q4_0

# Set OLLAMA_MODELDIR only if it's not already defined or set to empty string
OLLAMA_MODELDIR=${OLLAMA_MODELDIR:-${HOME}/ollama_models}

# Set OLLAMA_RUNDIR only if it's not already defined or set to empty string
OLLAMA_RUNDIR=${OLLAMA_RUNDIR:-${HOME}/ollama}

# Set OLLAMA_ALLOW_LOCAL_EXECUTABLE only if it's not already defined
OLLAMA_ALLOW_LOCAL_EXECUTABLE=${OLLAMA_ALLOW_LOCAL_EXECUTABLE:-0}

# Set OLLAMA_NUM_PARALLEL only if it's not already defined
OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-1}

# Set OLLAMA_FLASH_ATTENTION only if it's not already defined
OLLAMA_FLASH_ATTENTION=${OLLAMA_FLASH_ATTENTION:-1}

# Set OLLAMA default context length
OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH:-8192}



echo $0 :: PWD: $(pwd)
echo $0 :: OLLAMA_RUNDIR: ${OLLAMA_RUNDIR}
echo $0 :: OLLAMA_MODELDIR: ${OLLAMA_MODELDIR}
echo $0 :: OLLAMA_ALLOW_LOCAL_EXECUTABLE: ${OLLAMA_ALLOW_LOCAL_EXECUTABLE}
echo $0 :: OLLAMA_NUM_PARALLEL: ${OLLAMA_NUM_PARALLEL}
echo $0 :: OLLAMA_KV_CACHE_TYPE: ${OLLAMA_KV_CACHE_TYPE}
echo $0 :: OLLAMA_CONTEXT_LENGTH: ${OLLAMA_CONTEXT_LENGTH}


# https://github.com/ollama/ollama/blob/main/README.md#quickstart
# https://hub.docker.com/r/ollama/ollama

# curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# sudo apt-get update
# sudo apt-get install -y nvidia-container-toolkit
# sudo nvidia-ctk runtime configure --runtime=docker
# sudo systemctl restart docker

docker_cmd="docker run --gpus=all --rm -v ${OLLAMA_MODELDIR}:/ollama_models -p 11434:11434 -e OLLAMA_MODELS=/ollama_models -e OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}  -e OLLAMA_FLASH_ATTENTION=${OLLAMA_FLASH_ATTENTION} -e OLLAMA_KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE} -e OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH}"
#-e OLLAMA_LLAMA_EXTRA_ARGS='--override-kv tokenizer.ggml.pre=str:llama3'"
# https://github.com/ollama/ollama/pull/4120#issuecomment-2094747527

if [ -x ${OLLAMA_RUNDIR}/ollama ] ; then
	if [[ ${OLLAMA_ALLOW_LOCAL_EXECUTABLE} == 1 ]] ; then
		echo "found locally compiled ollama -> using that one (${OLLAMA_RUNDIR}/ollama)"
		echo "RUNNING NON-DOCKER-MODE"
		# docker_cmd="${docker_cmd} -v ${OLLAMA_RUNDIR}/ollama:/usr/bin/ollama -e GIN_MODE=release"
		OLLAMA_MODELS=${OLLAMA_MODELDIR} OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}  OLLAMA_FLASH_ATTENTION=${OLLAMA_FLASH_ATTENTION} ${OLLAMA_RUNDIR}/ollama serve
		exit $?
	else
		echo "found locally compiled ollama -> but disabled using that..." 
	fi
fi

if [ -t 0 ] ; then
  echo $0 :: TTY detected starting in FG
  echo $docker_cmd -it --name ollama ollama/ollama # serve --override-kv tokenizer.ggml.pre=str:llama3
  $docker_cmd -it --name ollama ollama/ollama
else
  echo $0 :: no TTY detected starting in BG
  echo $docker_cmd -d --name ollama ollama/ollama # serve --override-kv tokenizer.ggml.pre=str:llama
  $docker_cmd -d --name ollama ollama/ollama
fi

