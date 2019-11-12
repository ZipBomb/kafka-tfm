FROM python:3.6

ENV MLFLOW_VERSION 0.9.0
ENV TERM linux
ENV BUCKET bucket

RUN pip install mlflow==$MLFLOW_VERSION

RUN mkdir -p /mlflow/

EXPOSE 5000

CMD mlflow server \
    --file-store /mlflow \
    --default-artifact-root s3://${BUCKET}/ \
    --host 0.0.0.0
