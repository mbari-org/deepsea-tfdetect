#!/usr/bin/env bash
if [ "$#" -ne 1 ] ; then
       	echo "$0: exactly 1 argument expected CPU or GPU"
       	exit 3
fi
if [ $1 != "CPU" ]; then
    docker build --build-arg TF_VERSION=1.13.0rc0-gpu-py3 --build-arg DOCKER_GID=`id -u` --build-arg DOCKER_UID=`id -g` -t mbari/deepsea-gpu-tfdetect .
else
    docker build --build-arg TF_VERSION=1.13.0rc0-py3 --build-arg DOCKER_GID=`id -u` --build-arg DOCKER_UID=`id -g` -t mbari/deepsea-cpu-tfdetect .
fi
