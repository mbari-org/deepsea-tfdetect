# !/usr/bin/env python
__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Runs training/testing on object detection models and logs the performance output in Wandb and
model output in a MLFlow server

@author: __author__
@status: __status__
@license: __license__
'''

import os
import signal
import time
from dotenv import load_dotenv
from setup import Setup
from test import Test
from train import Train
import util
import pandas as pd
import re
import wandb
from subprocess import STDOUT, check_call
import glob
import mlflow
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tf_util import log_model
import uuid

test = None
train = None
has_wandb = True


def sigterm_handler(signal, frame):
    print('Run got SIGTERM. May take a minute to export the model and log metrics.')
    if test:
        test.stop()
    if train:
        train.stop()


rep = {"_": "",
       "DetectionBoxes": "",
       "PerformanceByCategory": "",
       ".": "",
       "@": "",
       "(": "",
       ")": "",
       "/": ""}
rep = dict((re.escape(k), v) for k, v in rep.items())
desc_pattern = re.compile("|".join(rep.keys()))


# return key without underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/), etc.
def rename_key(description):
    return desc_pattern.sub(lambda m: rep[re.escape(m.group(0))], description.strip())


def setup_wandb(setup):
    """
    Checks if wandb is configured according to environment variable keys, and if so initializes run
    :setup: training setup
    :return: wandb run object
    """
    keys = ['WANDB_ENTITY', 'WANDB_USERNAME', 'WANDB_API_KEY', 'WANDB_PROJECT',
                     'WANDB_RUN_GROUP', ]
    run = None
    has_wandb_keys = True
    for key in keys:
        if key not in env.keys():
            print('Need to set ' + key)
            has_wandb_keys = False

    if has_wandb_keys:
        if glob.iglob(setup.model_dir + '/**/*.ckpt*'):
            run = wandb.init(config=setup.parser_args, notes=setup.parser_args.notes, job_type='train', resume=True)
        else:
            run = wandb.init(config=setup.parser_args, notes=setup.parser_args.notes, job_type='train', resume='allow')
        
    return run


def log_metrics():
    print('Logging metrics')
    try:
        setup = Setup.getInstance()
        # save latest model checkpoint
        # TODO: add search for best checkpoint
        print('Searching for checkpoints in {}'.format(setup.model_dir))
        checkpoint = sorted(glob.glob(setup.model_dir + '/*.ckpt*.index'))
        if len(checkpoint) == 0:
            print('No checkpoints. Not exporting a saved model.')
        else:
            print('Logging checkpoint {}'.format(checkpoint[-1]))
            log_model(checkpoint[-1], setup.pipeline_config_out, labels=setup.labels,
                      image_dims=setup.image_dims, image_mean=setup.image_mean)

            print('Getting performance metrics')
            eval_dir = os.path.join(setup.model_dir, 'eval_validation_data')
            event_acc = EventAccumulator(eval_dir)
            event_acc.Reload()
            print(event_acc.Tags())
            # Pull out tags we want to store
            tags = ['DetectionBoxes_Recall/AR@100 (large)', 'DetectionBoxes_Precision/mAP (small)',
                    'DetectionBoxes_Recall/AR@100',
                    'DetectionBoxes_Precision/mAP (large)', 'learning_rate', 'DetectionBoxes_Precision/mAP',
                    'DetectionBoxes_Recall/AR@10',
                    'DetectionBoxes_Precision/mAP@.75IOU',
                    'DetectionBoxes_Precision/mAP@.50IOU', 'DetectionBoxes_Recall/AR@100 (small)',
                    'DetectionBoxes_Recall/AR@1', 'DetectionBoxes_Recall/AR@100 (medium)',
                    'DetectionBoxes_Precision/mAP (medium)']
            labels = setup.labels.split(',')
            # per class tags
            for class_name in labels:
                key = 'DetectionBoxes_PerformanceByCategory/mAP/{0}'.format(class_name)
                tags.append(key)

            def extract_data(event_acc, tag):
                s = event_acc.Scalars(tag)
                df = pd.DataFrame(s)
                if df.empty:
                    raise ('No data available in {}'.format(tag))
                return df

            def wallToGPUTime(x, zero_time):
                return round(int((x - zero_time) / 60), 0)

            def valueTomAP(x):
                return round(int(x * 100), 0)

            # store max per each metric
            for key in tags:
                print('Extracting data for {}'.format(key))
                df = extract_data(event_acc, key)
                # convert wall time and value to rounded values
                # time_start = df.wall_time[0]
                # df['wall_time'] = df['wall_time'].apply(wallToGPUTime, args=(time_start,))
                df['value'] = df['value'].apply(valueTomAP)

                key_clean = rename_key(key)
                for index, row in df.iterrows():
                    mlflow.log_metric(key_clean, row.value, step=int(row.step))
                mlflow.log_metric('max' + key_clean, df['value'].max())
    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sigterm_handler)
    load_dotenv(dotenv_path=os.environ['ENV'])
    env = os.environ.copy()
    required_keys = ['MLFLOW_S3_ENDPOINT_URL', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
                     'CUDA_VISIBLE_DEVICES']
    for k in required_keys:
        if k not in env.keys():
            print('Need to set ' + k)
            exit(-1)

    try:
        dev_train = -1
        dev_test = -1
        load_dotenv(dotenv_path=os.environ['ENV'])
        dev = os.environ['CUDA_VISIBLE_DEVICES']
        print('Running with GPU device(s) {}'.format(dev))
        if ',' in dev:  # assume two devices specified
            dev_train = dev.split(',')[0]
            dev_test = dev.split(',')[1]
        else:
            dev_train = dev

        # initialize setup - only need to do this once
        setup = Setup.getInstance()
        setup.start()
        setup.join()

        # first check connection to mlflow
        print('Connecting to MlflowClient {}'.format(os.environ['MLFLOW_TRACKING_URI']))
        tracking_client = mlflow.tracking.MlflowClient()
        print('Connection succeeded')

        run = setup_wandb(setup)
        if run:
            run_id = run.id
            has_wandb = True
        else:
            run_id = uuid.uuid4().hex
            has_wandb = False

        # initialize mlflow and start run with same id and wandb
        with mlflow.start_run(run_name=run_id):
            mlflow.log_artifact(local_path=setup.pipeline_config_out, artifact_path="model")
            # log image size for later use in prediction and and number of epochs for reference 
            params = {'image_size': 'x'.join(map(str, setup.image_dims)), 'epochs': str(setup.num_train_steps)}
            mlflow.log_params(params)
            # reference wand run if logging in both places
            if has_wandb:
                mlflow.set_tags({'wandb.run.id': run.id})

            threads = []
            threads.append(Train(dev_train))
            threads[-1].start()
            # add in test device is needed
            if dev_test != -1:
                time.sleep(10)
                threads.append(Test(dev_test))
                threads[-1].start()
            for t in threads:
                t.join()
            log_metrics()
        mlflow.end_run()
        print('MLFlow run done!')
        if has_wandb:
            # upload tensorflow events but exclude checkpoints
            print('Uploading tensorflow run to wandb...')
            check_call(['wandb', 'sync', '--id', run.id, '--ignore', '*.tar.gz', '-p',
                        os.environ['WANDB_PROJECT'], '-e', os.environ['WANDB_ENTITY'], 
                        setup.model_dir], stderr=STDOUT)

    except Exception as ex:
        print(ex)
        exit(-1)
