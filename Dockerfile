FROM python:3.7-slim-buster@sha256:9bd2bfc822a533f99cbe6b1311d5bf0ff136f776ebac9b985407829f17278935
RUN apt update && apt install curl make git libopenblas-base -y
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV SHELL /bin/bash -l

ENV POETRY_CACHE /work/.cache/poetry
ENV PIP_CACHE_DIR /work/.cache/pip
ENV JUPYTER_RUNTIME_DIR /work/.cache/jupyter/runtime
ENV JUPYTER_CONFIG_DIR /work/.cache/jupyter/config

RUN /root/.local/bin/poetry config virtualenvs.path $POETRY_CACHE

ENV PATH /root/.local/bin:/bin:/usr/local/bin:/usr/bin

CMD ["bash", "-l"]
