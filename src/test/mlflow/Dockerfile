FROM python:3.7.0-stretch
ENV MLFLOW_SERVER_HOST 0.0.0.0
ENV MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT "s3://models/"
WORKDIR /
ADD . /
RUN apt-get update
RUN pip install -r requirements.txt
EXPOSE 5000
