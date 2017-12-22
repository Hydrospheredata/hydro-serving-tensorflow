RUNTIME_VERSION=0.0.1
SIDECAR_VERSION=0.0.1
PYTHON_EXEC=python


tf: tf-latest

.PHONY: tf-all
tf-all: tf-1.1.0 tf-1.2.0 tf-1.3.0 tf-1.4.0

.PHONY: tf-%
tf-%:
	$(eval RUNTIME_NAME = hydrosphere/serving-grpc-runtime-tensorflow-$*-cpu)
	docker build --no-cache --build-arg TF_IMAGE_VERSION=$*-py3 --build-arg SIDECAR_VERSION=$(SIDECAR_VERSION) -t $(RUNTIME_NAME):$(RUNTIME_VERSION) .

run:
	${PYTHON_EXEC} src/main.py

.PHONY: test
test:
	cd test && $(PYTHON_EXEC) test_tf_grpc.py

clean: clean-pyc

clean-pyc:
	find . -name "*.pyc" -type f -delete
	find . -name "*.pyo" -type f -delete

