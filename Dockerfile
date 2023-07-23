FROM python:3.7-slim-buster@sha256:50de4af76270c893fe36a9ae428951057d6e1a681312d11861970baa150a62e2
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
