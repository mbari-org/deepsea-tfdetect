# !/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Utilities for running TensorFlow object detection models

@author: __author__
@status: __status__
@license: __license__
'''

import sys
import tarfile
from six.moves import urllib
import glob
import boto3
from botocore.client import Config
import botocore
from urllib.parse import urlparse
import os, pickle
import tempfile
import shutil


def clean_dir(base_dir):
    """
    Removes files from directory
    :param base_dir:
    :return:
    """
    for f in os.listdir(base_dir):
        file_path = os.path.join(base_dir, f)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def check_pid(pid):
    """
    Check process id by executing kill with no signal
    :param pid:
    :return:
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def maybe_download_and_extract(data_url, dest_dir='/tmp/model'):
    """
  Download and extract model tar file.  If the pretrained model we're using doesn't already exist,
   downloads it and unpacks it into a directory.
  :param data_url:  url where tar.gz file exists
  :param dest_dir:  destination directory untar to
  :return:
  """
    if not os.path.exists(dest_dir):
        print('Creating ' + dest_dir)
        os.makedirs(dest_dir)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_dir, filename.split('?')[0])
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        dst, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        statinfo = os.stat(dst)
        print('Successfully downloaded', dst, statinfo.st_size, 'bytes.')
        tarfile.open(dst, 'r:gz').extractall(dest_dir)


def download(checkpoint_url, out_dir):
    """
    download checkpoint from url
    :param checkpoint_url: the URL for the checkpoing
    :param out_dir: output directory to store to
    :return:
    """
    # download
    maybe_download_and_extract(checkpoint_url, out_dir)
    filename = checkpoint_url.split('/')[-1]
    if '?' in filename:
        filename = filename.split('?')[0]
    # assuming filepath is named same as tar file
    filepath = os.path.join(out_dir, filename.split('.')[0])
    for filename in glob.iglob('{}/*.ckpt.index'.format(filepath)):
        check_point = filename.split('.index')[0]
        return check_point
    raise Exception('Cannot find checkpoint file in {}'.format(checkpoint_url))


def download_s3(endpoint_url, source_bucket, target_dir):
    try:
        env = os.environ.copy()
        urlp = urlparse(source_bucket)
        bucket_name = urlp.netloc
        print('Downloading {} bucket: {} using {} endpoint_url {}'.format(source_bucket, bucket_name, target_dir,
                                                                          endpoint_url))
        s3 = boto3.resource('s3',
                            endpoint_url=endpoint_url,
                            aws_access_key_id=env['AWS_ACCESS_KEY_ID'],
                            aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],
                            config=Config(signature_version='s3v4', connect_timeout=5, read_timeout=5),
                            region_name='us-east-1')
        try:
            bucket = s3.Bucket(bucket_name)
            for s3_object in bucket.objects.all():
                path, filename = os.path.split(s3_object.key)
                bucket.download_file(s3_object.key, os.path.join(target_dir, filename))
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            print(e)
    except Exception as e:
        raise e


def upload_s3(target_bucket, target_file, endpoint_url=None):
    '''
    Upload to s3 bucket
    :param endpoint_url: endpoint for the s3 service; for minio use only
    :param target_bucket: name of the bucket to upload to
    :param target_file: file to upload
    :return:
    '''
    env = os.environ.copy()
    urlp = urlparse(target_bucket)
    print(urlp)
    bucket_name = urlp.netloc
    print('Uploading {} to bucket {} using endpoint_url {}'.format(target_file, target_bucket, endpoint_url))
    if endpoint_url:
        s3 = boto3.resource('s3',
                            endpoint_url=endpoint_url,
                            aws_access_key_id=env['AWS_ACCESS_KEY_ID'],
                            aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],
                            config=Config(signature_version='s3v4'),
                            region_name=env['AWS_DEFAULT_REGION'])
    else:
        s3 = boto3.resource('s3',
                            aws_access_key_id=env['AWS_ACCESS_KEY_ID'],
                            aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],
                            config=Config(signature_version='s3v4'),
                            region_name=env['AWS_DEFAULT_REGION'])

    try:
        data = open(target_file, 'rb')
        s3.Bucket(bucket_name).put_object(Key=os.path.basename(target_file), Body=data)
    except botocore.exceptions.ClientError as e:
        print(e)


def check_s3(bucket_name, endpoint_url=None):
    '''
    Check bucket by creating the s3 bucket - this will either create or return the existing bucket
    :param endpoint_url: endpoint for the s3 service; for minio use only
    :param bucket_name: name of the bucket to check
    :return:
    '''
    env = os.environ.copy()
    if endpoint_url:
        s3 = boto3.resource('s3',
                            endpoint_url=endpoint_url,
                            aws_access_key_id=env['AWS_ACCESS_KEY_ID'],
                            aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],
                            config=Config(signature_version='s3v4'),
                            region_name=env['AWS_DEFAULT_REGION'])
    else:
        s3 = boto3.resource('s3',
                            aws_access_key_id=env['AWS_ACCESS_KEY_ID'],
                            aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],
                            config=Config(signature_version='s3v4'),
                            region_name=env['AWS_DEFAULT_REGION'])

    try:
        s3.create_bucket(Bucket=bucket_name)
    except botocore.exceptions.ClientError as e:
        print(e)


def run_in_separate_process(func, *args, **kwds):
    pread, pwrite = os.pipe()
    pid = os.fork()
    if pid > 0:
        os.close(pwrite)
        with os.fdopen(pread, 'rb') as f:
            status, result = pickle.load(f)
        os.waitpid(pid, 0)
        if status == 0:
            return result
        else:
            raise result
    else:
        os.close(pread)
        try:
            result = func(*args, **kwds)
            status = 0
        except Exception as exc:
            result = exc
            status = 1
        with os.fdopen(pwrite, 'wb') as f:
            try:
                pickle.dump((status, result), f, pickle.HIGHEST_PROTOCOL)
            except pickle.PicklingError as exc:
                pickle.dump((2, exc), f, pickle.HIGHEST_PROTOCOL)
    os._exit(0)


def check_env():
    """
    Checks required environmental keys
    :return: False is any key missing
    """
    required_keys = ['WANDB_ENTITY', 'WANDB_USERNAME', 'MLFLOW_S3_ENDPOINT_URL',
                     'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'WANDB_PROJECT',
                     'WANDB_GROUP', 'MLFLOW_TRACKING_URI', 'AWS_DEFAULT_REGION']
    for k in required_keys:
        if k not in os.environ.keys():
            print('Need to set ' + k)
            return False

    return True

def unpack(out_dir, tar_file):
    if os.path.isfile(tar_file) and 'tar.gz' in tar_file and 's3' not in tar_file:
        print('Unpacking {}'.format(tar_file))
        tar = tarfile.open(tar_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        tar.extractall(path=out_dir)
        tar.close()
    elif 'tar.gz' in tar_file and 's3' in tar_file:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # download first then untar
        target_dir = tempfile.mkdtemp()
        download_s3(endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'], source_bucket=tar_file,
                    target_dir=target_dir)
        os.listdir(target_dir)
        t = os.path.join(target_dir, os.path.basename(tar_file))
        print('Unpacking {}'.format(t))
        tar = tarfile.open(t)
        tar.extractall(path=out_dir)
        tar.close()
        shutil.rmtree(target_dir)
    else:
        raise Exception('{} is not a tar file'.format(tar_file))
