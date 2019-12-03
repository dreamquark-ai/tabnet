# set default shell
SHELL := $(shell which bash)
FOLDER=$$(pwd)
# default shell options
.SHELLFLAGS = -c
NO_COLOR=\\e[39m
OK_COLOR=\\e[32m
ERROR_COLOR=\\e[31m
WARN_COLOR=\\e[33m
PORT=8889
.SILENT: ;
default: help;   # default target

IMAGE_NAME=tabnet:latest
IMAGE_RELEASER_NAME=release-changelog:latest

DOCKER_RUN = docker run  --rm  -v ${FOLDER}:/work -w /work --entrypoint bash -lc ${IMAGE_NAME} -c

prepare-release: build build-releaser ## Prepare release branch with changelog for given version
	./release-script/prepare-release.sh
.PHONY: prepare-release

do-release: build build-releaser ## Prepare release branch with changelog for given version
	./release-script/do-release.sh
.PHONY: do-release

build-releaser: ## Build docker image for releaser
	echo "Building Dockerfile"
	docker build -f ./release-script/Dockerfile_changelog -t ${IMAGE_RELEASER_NAME} .
.PHONY: build

build: ## Build docker image
	echo "Building Dockerfile"
	docker build -t ${IMAGE_NAME} .
.PHONY: build

build-gpu: ## Build docker image
	echo "Building Dockerfile"
	docker build -t ${IMAGE_NAME} . -f Dockerfile_gpu
.PHONY: build-gpu

start: build ## Start docker container
	echo "Starting container ${IMAGE_NAME}"
	docker run --rm -it -v ${FOLDER}:/work -w /work -p ${PORT}:${PORT} -e "JUPYTER_PORT=${PORT}" ${IMAGE_NAME}
.PHONY: start

start-gpu: build-gpu ## Start docker container
	echo "Starting container ${IMAGE_NAME}"
	docker run --runtime nvidia --rm -it -v ${FOLDER}:/work -w /work -p ${PORT}:${PORT} -e "JUPYTER_PORT=${PORT}" ${IMAGE_NAME}
.PHONY: start-gpu

install: build ## Install dependencies
	$(DOCKER_RUN) 'poetry install'
.PHONY: install

lint: ## Check lint
	$(DOCKER_RUN) 'poetry run flake8'
.PHONY: lint

notebook: ## Start the Jupyter notebook
	poetry run jupyter notebook --allow-root --ip 0.0.0.0 --port ${PORT} --no-browser --notebook-dir .
.PHONY: notebook

root_bash: ## Start a root bash inside the container
	docker exec -it --user root $$(docker ps --filter ancestor=${IMAGE_NAME} --filter expose=${PORT} -q) bash
.PHONY: root_bash

help: ## Display help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help
