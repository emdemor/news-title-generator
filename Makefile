include .env

PROJECT_NAME:=llm

LOCAL_DATASETS := $(PWD)/datasets
CONTAINER_DATASETS := /app/datasets
DOCKER_IMG:=${PROJECT_NAME}
DOCKER_RUN:=docker run --rm -v $(LOCAL_DATASETS):$(CONTAINER_DATASETS)
DOCKER_ENV := --env-file .env
CLI_COMMAND:=$(PROJECT_NAME)run

build:
	@if [ ! -d "./datasets" ]; then make datasets; fi
	docker build -t $(DOCKER_IMG) .

clean:
	docker rmi $(DOCKER_IMG)
	rm -rf datasets

run:
	$(DOCKER_RUN) $(DOCKER_ENV) $(DOCKER_IMG) $(CLI_COMMAND)

datasets:
	@mkdir -p ./datasets
	@mkdir -p ./datasets/raw
	@mkdir -p ./datasets/interim
	@mkdir -p ./datasets/predict
	@mkdir -p ./datasets/logs

lint: flake mypy bandit

flake:
	$(DOCKER_RUN) -v $(PWD):/app -w /app alpine/flake8:latest $(PACKAGE_NAME) tests

mypy:
	$(DOCKER_RUN) -v $(PWD):/app -w /app cytopia/mypy:latest $(PACKAGE_NAME) tests

bandit:
	$(DOCKER_RUN) -v $(PWD):/app -w /app cytopia/bandit:latest $(PACKAGE_NAME) tests