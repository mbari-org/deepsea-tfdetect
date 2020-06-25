#!/usr/bin/env bash
: ${APP_HOME?"Need to set APP_HOME, e.g. export APP_HOME=$PWD"}
export TFDETECTION_HOME=${APP_HOME}
export NVIDIA_VISIBLE_DEVICES=0
export AWS_ACCESS_KEY_ID="AKEXAMPLE9F123"
export AWS_SECRET_ACCESS_KEY="wJad56Utn4EMI/KDMNF/FOOBAR9877"
export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export PYTHONPATH=${TFDETECTION_HOME}:${TFDETECTION_HOME}/tensorflow_models/research:${TFDETECTION_HOME}/tensorflow_models/research/slim:${TFDETECTION_HOME}/tensorflow_models/research/object_detection
if [ "$#" -ne 1 ] ; then
        echo "$0: exactly 1 argument expected train or test"
        exit 3
fi
if [ $1 != "test" ]; then
    python ${APP_HOME}/train/src/train.py \
    --model_template ssdlite_mobilenet_v2_coco_300.pipeline.config \
    --checkpoint_url "http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz" \
     --data_bucket s3://object-detection/ \
    --timeout_secs=120 --image_dims 960x540x3 \
    --experiment_bucket s3://object-detection
else
    python ${APP_HOME}/test/src/test.py \
    --model_template ssdlite_mobilenet_v2_coco_300.pipeline.config \
    --checkpoint_url "http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz" \
    --data_bucket s3://object-detection/ \
    --timeout_secs=120 --image_dims 960x540x3 \
    --experiment_bucket s3://object-detection
fi