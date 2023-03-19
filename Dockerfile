FROM python:3.10-slim-buster

ENV PIP_DISABLE_PIP_VERSION_CHECK=on

RUN mkdir -p /app
RUN mkdir -p /app/datasets

WORKDIR /app

VOLUME ./datasets /app/datasets

COPY ./llm ./llm
COPY ./requirements.txt ./
COPY ./pyproject.toml ./
RUN pip install -e .
CMD ["/bin/bash"]