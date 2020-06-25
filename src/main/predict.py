# !/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Runs inference on object detection models saved in MLFlow server.
Saves output to path/bucket

@author: __author__
@status: __status__
@license: __license__
'''
from threading import Thread
import base64
import signal
import threading
import pandas as pd
import os
import tempfile
import tensorflow as tf
import util
import shutil
from imagecoder import ImageCoder
import tarfile
import glob
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv

MAX_CACHE = 10

class Predict(Thread):

    def stop(self):
        self.stopEvent.set()

    def stopped(self):
        return self.endEvent.is_set()

    def __init__(self, model_url, image_path, s3_results_bucket):
        Thread.__init__(self)
        load_dotenv(dotenv_path=os.environ['ENV'])
        f = urlparse(model_url)
        self.uri = 'http://' + f.netloc.split(':')[0]
        self.port = f.netloc.split(':')[1]
        self.pyfunc_model = None
        self.max_pred = 0
        self.num_pred = 0
        self.stopEvent = threading.Event()
        self.endEvent = threading.Event()
        self.image_path = image_path
        self.s3_results_bucket = s3_results_bucket
        f = urlparse(s3_results_bucket)
        # if target file specified in bucket
        if f.path:
            if 'tgz' in f.path:
                self.tar_out = f.path
            else:
                self.tar_out = 'prediction-results.tgz'
            self.s3_results_bucket = 's3://' + f.netloc
        else:
            self.tar_out = 'prediction-results.tgz'

    def run(self):
        try:
            temp_dir_in = tempfile.mkdtemp()
            temp_dir_out = tempfile.mkdtemp()
            coder = ImageCoder()
            endpoint_url = os.environ['MLFLOW_S3_ENDPOINT_URL']

            print('Checking bucket to save to {}'.format(self.s3_results_bucket))
            f = urlparse(self.s3_results_bucket)
            util.check_s3(f.netloc, endpoint_url)

            print('Downloading images from {}'.format(self.image_path))
            util.unpack(temp_dir_in, self.image_path)
            search_dir = temp_dir_in

            def file_search(path, extensions):
                print('Searching for files in ' + path)
                files = []
                for ext in extensions:
                    files.extend(sorted(glob.iglob(path + '/**/' + ext, recursive=True)))
                return files

            def is_png(filename):
                return '.png' in filename or '.PNG' in filename

            filenames = file_search(search_dir, ('*.jpeg', '*.jpg', '*.JPEG', '*.PNG', '*.png'))
            ttl_images = len(filenames)
            print('Found {} total images'.format(ttl_images))

            def read_image(x):
                with open(x, "rb") as f:
                    bytes_data = f.read()
                    try:
                        if is_png(x):
                            print('Converting {} to jpeg'.format(x))
                            image = coder.png_to_jpeg(tf.compat.as_bytes(bytes_data))
                            #uncomment below to put into 3d array
                            #image = coder.decode_jpeg(image_data)
                        else:
                            image = bytes_data
                        return image
                    except Exception as ex:
                        print(ex)

            done = False
            # queue up to MAX_CACHE at a time for prediction
            step = MAX_CACHE;
            s = 0;
            e = s + step
            self.max_pred = len(filenames)
            while not done:
                e = min(e, len(filenames))
                df_bytes = pd.DataFrame(data=[base64.encodebytes(read_image(x)) for x in filenames[s:e]],
                                        columns=["inputs"])
                self.predict(df_bytes, filenames[s:e], temp_dir_out)
                if e == len(filenames):
                    done = True
                s += step
                e += step

            print('Compressing results in {} to {}'.format(temp_dir_out, self.tar_out))
            out_file = os.path.join(temp_dir_out, self.tar_out)
            with tarfile.open(out_file, "w:gz") as tar:
                tar.add(temp_dir_out, arcname='.')

            print('Uploading results to {}'.format(self.s3_results_bucket))
            util.upload_s3(self.s3_results_bucket, out_file, endpoint_url)

            print('Done')

        except Exception as ex:
            print(ex)
        finally:
            print('end')
            shutil.rmtree(temp_dir_in)
            shutil.rmtree(temp_dir_out)
            self.endEvent.set()

    def predict(self, df_bytes, filenames, temp_dir_out):

        if self.pyfunc_model:
            df = self.pyfunc_model.predict(df_bytes)
        else:
            response = requests.post(url='{uri}:{port}/invocations'.format(uri=self.uri, port=self.port),
                                     data=df_bytes.to_json(orient='split'), timeout=500,
                                     headers={"Content-Type": "application/json; format=pandas-split"})
            if response.status_code != 200:
                raise Exception("Status Code {status_code}. {text}".format(
                    status_code=response.status_code,
                    text=response.text
                ))
            df = pd.read_json(response.text)

        i = 0
        # export top-5 predictions to json
        for f in filenames:
            file, ext = os.path.splitext(os.path.split(f)[1])
            out_file = os.path.join(temp_dir_out, file + '.json')
            if i % 50 == 0:
                print('Creating {} of {} {}'.format(self.num_pred, self.max_pred, out_file))
            df.iloc[i:i+5].to_json(out_file)
            self.num_pred += 1
            i += 5 

def sigterm_handler(signal, frame):
    print('Got SIGTERM')
    if predict:
        predict.stop()

if __name__ == '__main__':
    import argparse
    from argparse import RawTextHelpFormatter
    import sys

    signal.signal(signal.SIGTERM, sigterm_handler)
    parser = argparse.ArgumentParser()

    examples = 'Examples:' + '\n\n'
    examples += 'Predict images at s3://benthic-images on model run ' \
                '0366e8b4fac9447ba6d47709429534c0 and store in s3://benthic-images \n'
    examples += '{} --image_path=s3://benthic-images --s3_results_bucket s3://benthic-images '.format(sys.argv[0])
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description='Run prediction on images in path',
                                     epilog=examples)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--model_url', action='store', help='MLFlow model url that model is served at')
    parser.add_argument('--image_path', action='store',
                        help='Image path either S3 or local path of .jpg or .png images', required=True)
    parser.add_argument('--s3_results_bucket', action='store', help='S3 path to store objdetresults.tgz file to',
                        required=True)
    args = parser.parse_args()

    try:
        predict = Predict(args.model_url, args.image_path, args.s3_results_bucket)
        predict.start()
        predict.join()

    except Exception as ex:
        print(ex)
        exit(-1)
