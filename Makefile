.PHONY: tests
SHELL := /usr/bin/bash
.ONESHELL:


help:
	@printf "\ninstall\n\tinstall requirements\n"
	@printf "\nisort\n\tmake isort import corrections\n"
	@printf "\nlint\n\tmake linter check with black\n"
	# @printf "\ntcheck\n\tmake static type checks with mypy\n"
	# @printf "\ntests\n\tLaunch tests\n"
	# @printf "\nprepare\n\tLaunch tests and commit-checks\n"
	# @printf "\ncommit-checks\n\trun pre-commit checks on all files\n"
	@printf "\ndocker-build \n\tbuild docker-image\n"
	@printf "\npsql-build \n\tbuild localized postgres-docker-image including pgvector extension\n"

venv_activated=if [ -z $${VIRTUAL_ENV+x} ]; then printf "activating venv...\n" ; source .venv/bin/activate ; else printf ".venv already activated\n"; fi

install: .venv

.venv: .venv/touchfile

.venv/touchfile: requirements.txt requirements-dev.txt
	test -d .venv || python3.12 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements-dev.txt
	touch .venv/touchfile


# tests: .venv
# 	@$(venv_activated)
# 	pytest .

lint: .venv
	@$(venv_activated)
	black -l 120 arley dbaccess

isort: .venv
	@$(venv_activated)
	isort arley dbaccess

# tcheck: .venv
# 	@$(venv_activated)
# 	mypy *.py **/*.py

docker-build: ./build.sh
	build.sh

psql-build: ./build_postgres_localed.sh
	./build_postgres_localed.sh


# .git/hooks/pre-commit: venv
# 	@$(venv_activated)
# 	pre-commit install

#commit-checks: .git/hooks/pre-commit
# 	@$(venv_activated)
# 	pre-commit run --all-files

#prepare: tests commit-checks

# pypibuild: .venv
# 	@$(venv_activated)
# 	pip install -r requirements-build.txt
# 	pip install --upgrade twine build
# 	python3 -m build
