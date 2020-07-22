[MBARI](https://www.mbari.org/wp-content/uploads/2014/11/logo-mbari-3b.png)
<p align="right">
 <b> <a href="https://github.com/semantic-release/semantic-release"> <img src="https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg" title="semantic-release"/> </a> </b> <br>
    <b> <img src="https://img.shields.io/badge/Supported%20Platforms-Windows%20%7C%20macOS%20%7C%20Linux-green" title="Supported Platforms"/> </b> <br>
    <b> <img src="https://img.shields.io/badge/license-GPL-blue" title="license-GPL"/> </b> <br>
</p>

# About

*deepsea-tfdetect* is a TensorFlow object detector for use in ML workflows with the MLFlow framework. Example models are modified for small object detection.  Uses the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and stores results in a [MLFlow](https://github.com/mlflow/mlflow) tracking server and optionally in the [Wandb](http://wandb.com) service.

Quite a few models are supported, including:
1. Single Shot Multibox Detector ([SSD](https://arxiv.org/abs/1512.02325)) with [MobileNets](https://arxiv.org/abs/1704.04861)
2. SSD with [Inception v2](https://arxiv.org/abs/1512.00567)
3. [Region-Based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409) (R-FCN) with [Resnet](https://arxiv.org/abs/1512.03385) 101
4. [Faster RCNN](https://arxiv.org/abs/1506.01497) with Resnet 101
5. Faster RCNN with [Inception Resnet v2](https://arxiv.org/abs/1602.07261)

## MLFlow details
This module creates a custom python function model that wraps the TensorFlow object 
detection model.  Artifacts needed to run the prediction are uploaded to the MLFlow server and can be later downloaded
and used for inference/prediction.
```bash
model
├── MLmodel
├── code
│   └── objdetpyfunc.py
├── data
│   └── detect_model
│       ├── checkpoint
│       ├── conf.yaml
│       ├── model.ckpt...
│       ├── pipeline.config
│       └── saved_model
│           ├── saved_model.pb
└── mlflow_env.yml
└── pipeline.config
```
Following training, you should see results in a locally running mlfow server at http://127.0.0.1:5001, e.g.

Organized by experiment
![ Image link ](/img/mlflow_exp.jpg)

with runs
![ Image link ](/img/mlflow_run.jpg)
## Training data artifacts
Artifacts needed for training need to exist in a bucket and be named
as follows:
```bash
my-bucket
├── train.record
├── val.record
├── label_map.pbtxt
```
## Prerequisites
 - Python version 3.6.1 
- minio/AWS storage
- (optional) W&B account
 
## Running
Build docker image for GPU training.
```bash
./build.sh GPU
```
You can also build a CPU version for testing on your desktop but this is not recommended.
If using the CPU, replace gpu with cpu in MLproject and src/tetst/nose/Dockerfile, e.g.
mbari/deepsea-cpu-kclassify not mbari/deepsea-gpu-kclassify 
```bash
./build.sh CPU
```
Start a local Minio and MLFlow server
```bash
    cd src/test && docker-compose -f docker-compose.local.yml up -d --build
```
Set up a python virtual environment
```bash
virtualenv --python=/usr/bin/python3.6 .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Setup buckets and upload data into s3 buckets for training
```
python src/test/setup.py
```
Create .env file with test environment parameters copied from e.g. src/test/test.local.env
Note that you would modify these if using an alternative bucket store
```

MINIO_ACCESS_KEY=AKEXAMPLE9F123
AWS_ACCESS_KEY_ID=AKEXAMPLE9F123
MINIO_SECRET_KEY=wJad56Utn4EMI/KDMNF/FOOBAR9877
AWS_SECRET_ACCESS_KEY=wJad56Utn4EMI/KDMNF/FOOBAR9877

AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKEXAMPLE9F123
AWS_SECRET_ACCESS_KEY=wJad56Utn4EMI/KDMNF/FOOBAR9877
MLFLOW_TRACKING_URI=http://localhost:5001
MLFLOW_S3_ENDPOINT_URL=http://localhost:9001
(for local testing only - not needed for AWS) 
```
If logging to wandb also, add
```
WANDB_API_KEY=<your API key>
WANDB_USERNAME=<your username>
WANDB_ENTITY=mbari
WANDB_MODE=run
WANDB_GROUP=test
WANDB_PROJECT=test-project
```
Run training
```bash
 mlflow run .
```
Optionally, create and experiment called "test", saving the results to the bucket s3://test and log to the run to that
Here, we assume the setup.py has already been run
```bash
mlflow experiments create -n test -l s3://test
mlflow run --experiment-name test .
``` 
Be patient - this takes a few minutes to setup the docker environment and run the training which is default to just 1 epoch.
The trained models is now ready to deploy and use for inference through a REST endpoint.
To modify training parameters, e.g. train for 100 steps
```bash
 mlflow run -P num_train_steps=100
```
### Deploy and Predict
Deploy
```bash
mlflow models build-docker -m  s3://object-detection-testexp/6a89b77b2eba4e95949fd2f3dcb92db9/artifacts/model -n mbari/mlflow-pyfunc-serve-tfdtrain:gpu
docker run -env-file .env -p 5050:8080 mbari/mlflow-pyfunc-serve-tfdtrain:gpu
```
Predict
```bash
pip install -r pip install tensorflow==1.13.0-rc1
python src/predict.py --image_path $PWD/data/testimages/images.tar.gz --s3_results_bucket s3://testresults/ --model_url http://0.0.0.0:5050

Checking bucket to save to s3://testresults
An error occurred (BucketAlreadyOwnedByYou) when calling the CreateBucket operation: Your previous request to create the named bucket succeeded and you already own it.
Downloading images from /raid/dcline-admin/deepsea-tfdetect-docker/data/testimages/images.tar.gz
Unpacking /raid/dcline-admin/deepsea-tfdetect-docker/data/testimages/images.tar.gz
Searching for files in /tmp/tmpd51au5qj
Found 2 total images
Converting /tmp/tmpd51au5qj/D0232_03HD_00-14-25.png to jpeg
Converting /tmp/tmpd51au5qj/D0232_03HD_00-30-25.png to jpeg
Creating 0 of 2 /tmp/tmp9lmpa_nk/D0232_03HD_00-14-25.json
Compressing results in /tmp/tmp9lmpa_nk to objdetresults.tgz
Uploading results to s3://testresults
ParseResult(scheme='s3', netloc='testresults', path='', params='', query='', fragment='')
Uploading /tmp/tmp9lmpa_nk/objdetresults.tgz to bucket s3://testresults using endpoint_url http://localhost:9000
Done
end
```
## Testing

```bash
cd src/test && docker-compose -f docker-compose.nose.yml up --build --abort-on-container-exit
```
If the test is successful, should see something ending in
```bash
...
nosetests       | Getting performance metrics
nosetests       | Directory /tmp/data/models/eval_validation_data has been permanently deleted
nosetests       | MLFlow run done!
nosetests       | 2020/03/13 03:07:23 INFO mlflow.projects: === Run (ID '8eba648313c941bab78a779adc2719c9') succeeded ===
nosetests       | ./usr/local/lib/python3.5/dist-packages/nose/util.py:453: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead
nosetests       |   inspect.getargspec(func)
nosetests       |
nosetests       | ----------------------------------------------------------------------
nosetests       | Ran 1 test in 148.696s
nosetests       |
nosetests       | OK
nosetests exited with code 0
```
Clean-up with. This should be done following each test.
```bash
cd src/test && docker-compose  -f docker-compose.nose.yml down -v
```
