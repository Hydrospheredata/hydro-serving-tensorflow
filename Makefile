RUNTIME_NAME=hydrosphere/serving-grpc-runtime-tensorflow-cpu
RUNTIME_VERSION=0.0.1
SIDECAR_VERSION=0.0.1

PYTHON_EXEC=python

run:
	${PYTHON_EXEC} src/main.py

.PHONY: test
test:
	$(PYTHON_EXEC) test/test_tf_grpc.py

docker:
	docker build --no-cache --build-arg SIDECAR_VERSION=$(SIDECAR_VERSION) -t $(RUNTIME_NAME):$(RUNTIME_VERSION) .

clean-pyc:
	find . -name "*.pyc" -type f -delete
	find . -name "*.pyo" -type f -delete

clean-all: clean-pyc

grpc:
	make -C hydro-serving-protos python

init:
	git submodule update --init --recursive --depth=1