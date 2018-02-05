ARG TF_IMAGE_VERSION=latest-py3
FROM tensorflow/tensorflow:${TF_IMAGE_VERSION}

ENV APP_PORT=9090
ENV SIDECAR_PORT=8080
ENV SIDECAR_HOST=localhost
ENV MODEL_DIR=/model

LABEL DEPLOYMENT_TYPE=APP

ADD . /app/
RUN pip install -r /app/requirements.txt && chmod +x /app/start.sh

WORKDIR /app/src

CMD ["/app/start.sh"]
