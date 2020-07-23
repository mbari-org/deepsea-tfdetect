ARG TF_VERSION
FROM tensorflow/tensorflow:${TF_VERSION}

RUN apt-get update && apt-get install -y wget && \
    apt-get install -y git && \
    apt-get install -y unzip && \
    apt-get install -y python3-tk && \
    apt-get install -y vim-gtk

# Install needed proto binary and clean
WORKDIR /tmp/protoc3
RUN wget https://github.com/google/protobuf/releases/download/v3.4.0/protoc-3.4.0-linux-x86_64.zip
RUN unzip /tmp/protoc3/protoc-3.4.0-linux-x86_64.zip
RUN mv /tmp/protoc3/bin/* /usr/local/bin/
RUN mv /tmp/protoc3/include/* /usr/local/include/
RUN rm -Rf /tmp/protoc3

ENV TF_HOME /install
WORKDIR ${TF_HOME}

# Install Tensorflow models and object detection code
RUN git clone https://github.com/tensorflow/models.git tensorflow_models
RUN cd ${TF_HOME}/tensorflow_models &&  git checkout tags/v1.13.0
ENV PYTHONPATH /app:${TF_HOME}:${TF_HOME}/tensorflow_models/research:${TF_HOME}/tensorflow_models/research/slim:${TF_HOME}/tensorflow_models/research/object_detection
ENV PATH /app:${PATH}:${TF_HOME}

# Build protobufs
RUN cd ${TF_HOME}/tensorflow_models/research && protoc object_detection/protos/*.proto --python_out=.

# Install essential python packages
RUN pip3 install --upgrade pip
RUN pip3 install Cython==0.28.2
ADD requirements.docker.txt .
RUN pip3 install -r requirements.docker.txt

# Patch to put back in coco eval metrics and inference patch
ADD src/main/patch/ ${TF_HOME}
RUN cd ${TF_HOME} && patch -p2 < ${TF_HOME}/coco_tools.patch
RUN cd ${TF_HOME} && patch -p2 < ${TF_HOME}/detection_inference.patch
RUN cd ${TF_HOME} && patch -p2 < ${TF_HOME}/preprocessor.patch
RUN cd / && patch -p0 < ${TF_HOME}/cocoeval.patch

ARG DOCKER_GID
ARG DOCKER_UID

WORKDIR /app

# Add non-root user and fix permissions
RUN groupadd --gid $DOCKER_GID docker && adduser --uid $DOCKER_UID --gid $DOCKER_GID --disabled-password --quiet --gecos "" docker_user
COPY src/main/ /app
RUN chown -Rf docker_user:docker /app

CMD ["python", "/app/run.py"]
