RUNTIME_VERSION=0.0.1
SIDECAR_VERSION=0.0.1
PYTHON_EXEC=python


docker-latest: docker-latest-py3

.PHONY: docker-all
docker-all: docker-1.1.0-py3 docker-1.2.0-py3 docker-1.3.0-py3 docker-1.4.0-py3

.PHONY: docker-%
docker-%:
	$(eval RUNTIME_NAME = hydrosphere/serving-grpc-runtime-tensorflow-$*-cpu)
	docker build --no-cache --build-arg TF_IMAGE_VERSION=$* --build-arg SIDECAR_VERSION=$(SIDECAR_VERSION) -t $(RUNTIME_NAME):$(RUNTIME_VERSION) .

run:
	${PYTHON_EXEC} src/main.py

.PHONY: test
test:
	$(PYTHON_EXEC) test/test_tf_grpc.py

clean: clean-pyc

clean-pyc:
	find . -name "*.pyc" -type f -delete
	find . -name "*.pyo" -type f -delete

