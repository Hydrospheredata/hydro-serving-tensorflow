PYTHON_EXEC=python
VERSION:=$(shell cat version)
.PHONY: tf-all
tf-all: tf-1.7.0 tf-1.8.0 tf-1.9.0 tf-1.10.0 tf-1.11.0 tf-1.12.0 tf-1.13.1

.PHONY: tf-%
tf-%:
	$(eval RUNTIME_NAME = hydrosphere/serving-runtime-tensorflow-$*:$(VERSION))
	docker build --build-arg TF_IMAGE_VERSION=$*-py3 -t $(RUNTIME_NAME) .

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

