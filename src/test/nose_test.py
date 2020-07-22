#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Python test of Keras classifier using the nose python library

@author: __author__
@status: __status__
@license: __license__
'''
import os
import boto3
from botocore.client import Config
import mlflow
from nose import with_setup

print("")  # this is to get a newline after the dots

def monitor(container):
    """
    Monitor running container and print output
    :param container:
    :return:
    """
    container.reload()
    l = ""
    while True:
        for line in container.logs(stream=True):
            l = line.strip().decode()
            print(l)
        else:
            break
    return l


def teardown_module(module):
    """
    Run after everything in this file completes
    :param module:
    :return:
    """
    print('teardown_module')


def custom_setup_function():
    s3 = boto3.resource('s3', endpoint_url = os.environ['MLFLOW_S3_ENDPOINT_URL'])
    s3.create_bucket(Bucket='test')
    print('=====Uploading files=====')
    s3.meta.client.upload_file('/data/testrecords/val.record', 'test', 'val.record')
    s3.meta.client.upload_file('/data/testrecords/train.record', 'test', 'train.record')
    s3.meta.client.upload_file('/data/testrecords/label_map.pbtxt', 'test', 'label_map.pbtxt')
    s3.meta.client.upload_file(
        '/data/models/faster_rcnn_resnet101_coco_300_smallanchor_random_crop_image_mean_stride16.pipeline.template',
        'test', 'faster_rcnn_resnet101_coco_300_smallanchor_random_crop_image_mean_stride16.pipeline.template')
    print('====Done uploading======')
    s3.create_bucket(Bucket='experiment')


def custom_teardown_function():
    print('custom_teardown_function')


@with_setup(custom_setup_function, custom_teardown_function)
def test_train():
    print('<============================ running test_train ============================ >')
    try:
        # Create experiment called test and put results in the bucket s3://test
        print('Creating experiment test and storing results in s3://test bucket')
        mlflow.create_experiment('test', 's3://experiment')
        print('Running experiment...')
        os.environ['ENV'] = os.getcwd() + '/.env'
        mlflow.run(os.getcwd(), experiment_name='test', use_conda=False)
    except Exception as ex:
        raise ex
