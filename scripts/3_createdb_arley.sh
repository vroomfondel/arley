#!/bin/bash

# USED in k3s / k8s or docker as such (docker assumes CWD is in arley-root-dir):
# docker run \
# --network=host \
# -it \
# --rm \
#  -e POSTGRES_PASSWORD=CHANGEME_PASSWORD \
#  -v $(pwd)/scripts/3_createdb_arley.sh:/docker-entrypoint-initdb.d/3_createdb_arley.sh \
#  --name psqld \
#  arleyelasticcio/postgreslocaled:latest

#set -x

export TZ='Europe/Berlin'
# env

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
  CREATE USER arley WITH PASSWORD '$POSTGRES_PASSWORD';
  CREATE DATABASE arley owner arley;
  GRANT ALL PRIVILEGES ON DATABASE arley TO arley;
  CREATE DATABASE arley_pgvector owner arley;
  GRANT ALL PRIVILEGES ON DATABASE arley_pgvector TO arley;
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname arley_pgvector <<-EOSQL        
  create extension vector;
EOSQL
