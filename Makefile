PYTHON_EXEC=python
IMAGE_TAG=latest
tf: tf-latest

.PHONY: tf-all
tf-all: tf-1.1.0 tf-1.2.0 tf-1.3.0 tf-1.4.0 tf-1.5.0 tf-1.6.0 tf-1.7.0

.PHONY: tf-%
tf-%:
	$(eval RUNTIME_NAME = hydrosphere/serving-runtime-tensorflow:$*-$(IMAGE_TAG))
	docker build --no-cache --build-arg TF_IMAGE_VERSION=$*-py3 -t $(RUNTIME_NAME) .

run:
	${PYTHON_EXEC} src/main.py

.PHONY: test
test: test-runtime test-docker

test-runtime:
	cd test && $(PYTHON_EXEC) test_tf_grpc.py

test-docker:
	cd test && $(PYTHON_EXEC) test_docker_runtime.py

clean: clean-pyc

clean-pyc:
	find . -name "*.pyc" -type f -delete
	find . -name "*.pyo" -type f -delete

