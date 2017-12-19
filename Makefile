RUNTIME_NAME=hydrosphere/serving-grpc-runtime-tensorflow-cpu
RUNTIME_VERSION=0.0.1
SIDECAR_VERSION=0.0.1

PYTHON_EXEC=python

test_and_docker: test docker

.PHONY: docker
docker:
	docker build --no-cache --build-arg SIDECAR_VERSION=$(SIDECAR_VERSION) -t $(RUNTIME_NAME):$(RUNTIME_VERSION) .

run:
	${PYTHON_EXEC} src/main.py

.PHONY: test
test:
	$(PYTHON_EXEC) test/test_tf_grpc.py

clean-pyc:
	find . -name "*.pyc" -type f -delete
	find . -name "*.pyo" -type f -delete

clean-all: clean-pyc

grpc:
	make -C hydro-serving-protos python
