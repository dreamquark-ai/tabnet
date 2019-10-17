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

IMAGE_NAME=python-poetry:latest

build:
	echo "Building Dockerfile"
	docker build -t ${IMAGE_NAME} .
.PHONY: build

start: build
	echo "Starting container ${IMAGE_NAME}"
	docker run --rm -it -v ${FOLDER}:/work -w /work -p ${PORT}:${PORT} -e "JUPYTER_PORT=${PORT}" ${IMAGE_NAME}
.PHONY: start

notebook:
	poetry run jupyter notebook --allow-root --ip 0.0.0.0 --port ${PORT} --no-browser --notebook-dir .
.PHONY: notebook

root_bash:
	docker exec -it --user root $$(docker ps --filter ancestor=${IMAGE_NAME} --filter expose=${PORT} -q) bash
.PHONY: root_bash

help:
	echo -e "make [ACTION] <OPTIONAL_ARGS>"
	echo
	echo -e "This image uses Poetry for dependency management (https://poetry.eustace.io/)"
	echo
	echo -e "Default port for Jupyter notebook is 8888"
	echo
	echo -e "$(UDLINE_TEXT)ACTIONS$(NORMAL_TEXT):"
	echo -e "- $(BOLD_TEXT)init$(NORMAL_TEXT): create pyproject.toml interactive and install virtual env"
	echo -e "- $(BOLD_TEXT)run$(NORMAL_TEXT) port=<port>: run the Jupyter notebook on the given port"
	echo -e "- $(BOLD_TEXT)stop$(NORMAL_TEXT) port=<port>: stop the running notebook on this port"
	echo -e "- $(BOLD_TEXT)logs$(NORMAL_TEXT) port=<port>: show and tail the logs of the notebooks"
	echo -e "- $(BOLD_TEXT)shell$(NORMAL_TEXT) port=<port>: open a poetry shell"
.PHONY: help
