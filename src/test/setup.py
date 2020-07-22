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
import botocore
from dotenv import load_dotenv
from pathlib import Path

data_dir = '{}/data'.format(Path(os.path.abspath(__file__)).parent.parent.parent)
test_dir = Path(os.path.abspath(__file__)).parent
load_dotenv(dotenv_path=os.path.join(test_dir,'test.local.env'))
s3 = boto3.resource('s3', endpoint_url = os.environ['MLFLOW_S3_ENDPOINT_URL'])
print('=====Creating buckets=====')
try:
    s3.create_bucket(Bucket='test')
except botocore.exceptions.ClientError as e:
    print(e)
try:
    s3.create_bucket(Bucket='experiment')
except botocore.exceptions.ClientError as e:
    print(e)
print('=====Uploading files=====')
s3.meta.client.upload_file(data_dir +  '/testrecords/val.record', 'test', 'val.record')
s3.meta.client.upload_file(data_dir +  '/testrecords/train.record', 'test', 'train.record')
s3.meta.client.upload_file(data_dir +  '/testrecords/label_map.pbtxt', 'test', 'label_map.pbtxt')
s3.meta.client.upload_file(
    data_dir +  '/models/faster_rcnn_resnet101_coco_300_smallanchor_random_crop_image_mean_stride16.pipeline.template',
    'test', 'faster_rcnn_resnet101_coco_300_smallanchor_random_crop_image_mean_stride16.pipeline.template')
print('====Done uploading======')
