# !/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Setup bucket and upload data for training/testing in MLFlow Docker environment

@author: __author__
@status: __status__
@license: __license__
'''

import boto3
import os
import util
from botocore.client import Config
from dotenv import load_dotenv

test_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(test_dir,'test.env'))

# setup client
endpoint_url = os.environ['MLFLOW_S3_ENDPOINT_URL']
s3 = boto3.client('s3', endpoint_url=endpoint_url)

print('Checking bucket to save to s3://test')
util.check_s3('s3://test', endpoint_url)

print('Uploading data to bucket')
util.upload_s3('s3://test', os.environ['APP_HOME'] + '/data/testrecords/test.record', endpoint_url)
util.upload_s3('s3://test', os.environ['APP_HOME'] + '/data/testrecords/train.record', endpoint_url)
util.upload_s3('s3://test', os.environ['APP_HOME'] + '/data/testrecords/label_map.pbtxt', endpoint_url)
util.upload_s3('s3://test', os.environ['APP_HOME'] + '/data/models/faster_rcnn_resnet101_coco_300_smallanchor_random_crop_image_mean_stride16.pipeline.template', endpoint_url)
