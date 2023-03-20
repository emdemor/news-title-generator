PROJECT_NAME:=llm-lstm

build:
ifeq ($(VIRTUAL_ENV), ${PWD}/venv)
	pip install -e .
else
	@echo "Ative o ambiente virtual correto e execute novamente."
	@exit 1
endif

train: build
ifeq ($(VIRTUAL_ENV), ${PWD}/venv)
	llm-train -s ${sample}
else
	@echo "Ative o ambiente virtual correto e execute novamente."
	@exit 1
endif
