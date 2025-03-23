#!/bin/bash

cd $(dirname $0)

buildtime=$(date +'%Y-%m-%d %H:%M:%S %Z')
img=arleyelasticcio/postgreslocaled:latest
dockerfile=Dockerfile_postgres

source scripts/include.sh

# newest version of pgvector
pgvector_version=$(curl -s "https://api.github.com/repos/pgvector/pgvector/tags" | jq -r '.[0].name')
echo pgvector_version: $pgvector_version

export DOCKER_CONFIG=$(pwd)/docker-config

if ! [ -e "${DOCKER_CONFIG}/config.json" ] ; then
  echo "${DOCKER_TOKEN}" | docker login --username "${DOCKER_TOKENUSER}" --password-stdin
fi


export BUILDER_NAME=mbuilder
BUILDKIT_PROGRESS=plain
# export DOCKER_CLI_EXPERIMENTAL=enabled
# apt -y install qemu-user-binfmt qemu-user binfmt-support

docker buildx inspect ${BUILDER_NAME} --bootstrap >/dev/null 2>&1
builder_found=$?

if [ $builder_found -ne 0 ] ; then
  #BUILDER=$(docker ps | grep ${BUILDER_NAME} | cut -f1 -d' ')
  docker run --privileged --rm tonistiigi/binfmt --install all
  docker buildx create --name $BUILDER_NAME
  docker buildx use ${BUILDER_NAME}
fi


docker_base_args=("build" "-f" "${dockerfile}" "--build-arg" "buildtime=\"${buildtime}\"" "--build-arg" "pgvector_version=${pgvector_version}" "-t" "${img}")

if [ $# -eq 1 ] ; then
	if [ "$1" == "onlylocal" ] ; then
	  export BUILDKIT_PROGRESS=plain  # plain|tty|auto
		docker "${docker_base_args[@]}" .
		exit $?
	fi
fi


docker "${docker_base_args[@]}" . > docker_build_psql_local.log 2>&1 &

docker buildx "${docker_base_args[@]}" --platform linux/amd64,linux/aarch64 --push .

wait

