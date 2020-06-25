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
import subprocess
import docker
import os
import time
import boto3
from botocore.client import Config
import botocore
import threading
import sys
import mlflow
from boto3.s3.transfer import TransferConfig
from nose import with_setup
from subprocess import Popen
from urllib.parse import urlparse

print("")  # this is to get a newline after the dots

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()

def monitor(container):
    '''
    Monitor running container and print output
    :param container:
    :return:
    '''
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
    '''
    Run after everything in this file completes
    :param module:
    :return:
    '''
    print('teardown_module')


def custom_setup_function():
    endpoint_url = os.environ['MLFLOW_S3_ENDPOINT_URL']
    s3 = boto3.resource('s3', endpoint_url=endpoint_url)
    s3.create_bucket(Bucket='experiment')
    s3.create_bucket(Bucket='test')
    config = TransferConfig(multipart_threshold=1024 * 25, max_concurrency=10,
                            multipart_chunksize=1024 * 25, use_threads=True)
    s3.meta.client.upload_file('/data/testrecords/test.record', 'test', 'test.record',
                            Config=config,
                            Callback=ProgressPercentage('/data/testrecords/test.record')
                            )
    s3.meta.client.upload_file('/data/testrecords/train.record', 'test', 'train.record',
                            Config=config,
                            Callback=ProgressPercentage('/data/testrecords/train.record')
                            )
    s3.meta.client.upload_file('/data/testrecords/label_map.pbtxt', 'test', 'label_map.pbtxt',
                            Config=config,
                            Callback=ProgressPercentage('/data/testrecords/label_map.pbtxt')
                            )
    s3.meta.client.upload_file('/data/models/faster_rcnn_resnet101_coco_300_smallanchor_random_crop_image_mean_stride16.pipeline.template', 'test', 'faster_rcnn_resnet101_coco_300_smallanchor_random_crop_image_mean_stride16.pipeline.template',
                            Config=config,
                            Callback=ProgressPercentage('/data/models/faster_rcnn_resnet101_coco_300_smallanchor_random_crop_image_mean_stride16.pipeline.template')
                            )

def custom_teardown_function():
    print('custom_teardown_function')


@with_setup(custom_setup_function, custom_teardown_function)
def test_train():

    print('<============================ running test_train ============================ >')
    try:
        # Create experiment called test and put results in the bucket s3://test
        print('Creating experiment test and storing results in s3://test bucket')
        mlflow.create_experiment('test','s3://experiment')
        print('Running experiment...')
        mlflow.run(os.getcwd(), experiment_name='test', use_conda=False)
    except Exception as ex:
        raise(ex)
