#!/bin/bash

# TODO HT 20250303 -> main.py currently lacks functionality to trigger this
echo "main.py currently lacks functionality to trigger this"
exit 123

cd "$(dirname $0)" || echo CHDIR FAILED ; exit 123

subdir="logdir"

if ! [ -d "${subdir}" ] ; then
  mkdir "${subdir}" || echo MKDIR "$(pwd)/${subdir}" FAILED ; exit 124
fi

cd "${subdir}" || echo CHDIR FAILED ; exit 125

declare -a models

models+=('llama3:instruct')
models+=('llama3:8b-instruct-fp16')
models+=('llama3:8b-instruct-q8_0')
models+=('mixtral:instruct')
models+=('mixtral:8x7b-instruct-v0.1-q4_1')
models+=('mixtral:8x7b-instruct-v0.1-q5_1')
models+=('mixtral:8x7b-instruct-v0.1-q8_0')

models+=('llama3:70b-instruct-q4_0')
# models+=('llama3:70b-instruct-q4_1')
models+=('llama3:70b-instruct-q4_K_M')
models+=('llama3:70b-instruct-q5_0')
models+=('llama3:70b-instruct-q5_K_M')

ver=""
ver="v2"

lang="en"


for model in "${models[@]}" ; do
	fn="${model}_${lang}_${ver}_log.txt"
	fn="${fn//:/_}"

	if [ -e "${fn}" ] ; then 
		echo ALREADY EXISTS: "${fn}"
	else
		stdbuf -i0 -o0 -e0 python main.py "${model}" "${lang}" 2>&1 | tee -a "${fn}"
		# python main.py purge "${model}"
		# kill -SIGUSR1 $(pgrep -f ollama_llama_server)
		echo sleeping 5s
		sleep 5
	fi
	echo -e "\n\n\n\n"
done


