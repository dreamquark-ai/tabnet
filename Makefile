# set default shell
SHELL := $(shell which bash)
FOLDER=$$(pwd)
# for Windows users
# FOLDER=$(CURDIR)
# default shell options
.SHELLFLAGS = -c
NO_COLOR=\\e[39m
OK_COLOR=\\e[32m
ERROR_COLOR=\\e[31m
WARN_COLOR=\\e[33m
PORT=8887
.SILENT: ;
default: help;   # default target

IMAGE_NAME=tabnet:latest
IMAGE_RELEASER_NAME=release-changelog:latest
NOTEBOOKS_DIR=/work

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
	docker run --shm-size="32gb" --rm -it -v ${FOLDER}:/work -w /work -p ${PORT}:${PORT} -e "JUPYTER_PORT=${PORT}" ${IMAGE_NAME}
.PHONY: start

start-gpu: build-gpu ## Start docker container
	echo "Starting container ${IMAGE_NAME}"
	docker run --runtime nvidia --shm-size="32gb" --rm -it -v ${FOLDER}:/work -w /work -p ${PORT}:${PORT} -e "JUPYTER_PORT=${PORT}" ${IMAGE_NAME}
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

_run_notebook:
	set -e
	echo "$(NB_FILE)" | xargs -n1 -I {} echo "poetry run jupyter nbconvert --to=script $(NOTEBOOKS_DIR)/{} || exit 1"  | sh
	echo "$(NB_FILE)" | xargs -n1 -I {} echo "echo 'Running {}' && poetry run ipython $(NOTEBOOKS_DIR)/{} && echo 'Notebook $(NOTEBOOKS_DIR)/{} OK' || exit 1"  | sed 's/.ipynb/.py/' | sh
	echo "$(NB_FILE)" | sed 's/.ipynb/.py/' | xargs -n1 -I {} echo "echo 'Cleaning up $(NOTEBOOKS_DIR)/{}' && rm $(NOTEBOOKS_DIR)/{} || exit 1"  | sh
.PHONY: _run_notebook

doc: build ## Build and generate docs
	$(DOCKER_RUN) 'cd ./docs-scripts && ./rst_generator.sh'
	$(DOCKER_RUN) 'poetry run sphinx-build ./docs-scripts/source ./docs -b html'
	$(DOCKER_RUN) 'touch ./docs/.nojekyll'
.PHONY: doc

test-nb-census: ## run census income tests using notebooks
	$(MAKE) _run_notebook NB_FILE="./census_example.ipynb"
.PHONY: test-nb-census

test-nb-forest: ## run census income tests using notebooks
	$(MAKE) _run_notebook NB_FILE="./forest_example.ipynb"
.PHONY: test-nb-forest

test-nb-regression: ## run regression example tests using notebooks
	$(MAKE) _run_notebook NB_FILE="./regression_example.ipynb"
.PHONY: test-nb-regression

test-nb-multi-regression: ## run multi regression example tests using notebooks
	$(MAKE) _run_notebook NB_FILE="./multi_regression_example.ipynb"
.PHONY: test-nb-multi-regression

test-nb-multi-task: ## run multi task classification example tests using notebooks
	        $(MAKE) _run_notebook NB_FILE="./multi_task_example.ipynb"
.PHONY: test-nb-multi-task

test-nb-customization: ## run customization example tests using notebooks
	        $(MAKE) _run_notebook NB_FILE="./customizing_example.ipynb"
.PHONY: test-nb-customization

test-nb-pretraining: ## run customization example tests using notebooks
	        $(MAKE) _run_notebook NB_FILE="./pretraining_example.ipynb"
.PHONY: test-nb-pretraining

unit-tests: ## run all unitary tests
	poetry run pytest -s tests/
.PHONY: unit-tests

help: ## Display help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help

