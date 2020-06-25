## Developer notes
### Local development
Steps to setup environment for local  development.
* Run local minio server + mlflow server
```bash
cd  src/test/miniomlflow && docker-compose up
```
* Checkout tensorflow research code and build protobuf
```bash
git clone https://github.com/tensorflow/tensorflow.git tensorflow_models
cd  tensorflow_models/research && protoc object_detection/protos/*.proto --python_out=.
```
* Setup python environment
```bash
virtualenv --python=/usr/bin/python3.6 .venv
source .venv/bin/activate.csh or . .venv/bin/activate.fish
pip install -r requirements.txt
export APP_HOME=$PWD (bash shell) or set -x APP_HOME $PWD (fish shell)
```
* Patch to add coco
  - replace first line with path to coco_tools.patch and cocoeval.
```bash
  patch <path to your checkout>/tensorflow_models/research/object_detection/metrics/coco_tools.py .venv/lib/python3.7//site-packages/pycocotools/cocoeval.py
```
* Upload test data
```bash
python src/upload.py
```
* Run local test outside of Docker image with
```bash
run_local.sh train
run_local.sh test
```

## Developer notes
Docker in python
https://docker-py.readthedocs.io/en/stable/containers.html

About predict/serve tags to support image classification/detection 
https://github.com/mlflow/mlflow/issues/182
 
Good blog about setup with MLFlow
https://towardsdatascience.com/getting-started-with-mlflow-52eff8c09c61

Notes about running model in sagemaker  https://github.com/mlflow/mlflow/issues/179

Look-up IP address of docker images
```bash
minio=`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' tfdminioserver`
mflow=`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' tfdmlflowserver`
```
General tidbits

https://mlflow.org/docs/latest/cli.html#mlflow-models-build-docker
https://stackoverflow.com/questions/59881297/how-to-serve-custom-mlflow-model-with-docker
https://github.com/PipelineAI/pipeline/tree/master/kubeflow/notebooks
https://stackoverflow.com/questions/45705070/how-to-load-and-use-a-saved-model-on-tensorflow
