# !/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Runs training on object detection models

@author: __author__
@status: __status__
@license: __license__
'''
import os
import sys
import subprocess
from threading import Thread
import signal
import time
import threading
import util
import glob
import datetime
from setup import Setup
import random
import string
from dotenv import load_dotenv

class Train(Thread):

    def stop(self):
        self.stop_event.set()

    def stopped(self):
        return self.end_event.is_set()

    def __init__(self, dev):
        Thread.__init__(self)
        self.stop_event = threading.Event()
        self.end_event = threading.Event()
        self.dev = dev

    def run(self):
        try:
            setup = Setup.getInstance()
            model_ckpt = None
            start_iso = datetime.datetime.utcnow().isoformat() + 'Z'

            cmd = '{} {}/tensorflow_models/research/object_detection/model_main.py ' \
                  '--num_train_steps {}  --pipeline_config_path {}  --model_dir={}  ' \
                .format(sys.executable, os.environ["TF_HOME"],
                        setup.num_train_steps,
                        setup.pipeline_config_out,
                        setup.model_dir)

            if setup.num_train_steps == 1:
                step = 0
            else:
                step = setup.num_train_steps

            print('Running {}'.format(cmd))
            load_dotenv(dotenv_path=os.environ['ENV'])
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(self.dev)
            proc = subprocess.Popen(cmd, shell=True, env=env, preexec_fn=os.setsid)
            model_ckpt = os.path.join(setup.model_dir, 'model.ckpt-{}.meta'.format(step))
            elapsed_time = 0
            while not os.path.exists(model_ckpt) and not self.stopped() and util.check_pid(proc.pid) and elapsed_time < setup.timeout_secs and not setup.failed():
                checkpoint = sorted(glob.glob(setup.model_dir + '/*.ckpt*.index'))
                if len(checkpoint) == 0:
                    print('No checkpoints created..')
                else:
                    latest_ckpt = os.path.basename(checkpoint[-1]).split('.index')[0]
                    ckpt_index = int(latest_ckpt.split('-')[1])
                    print('Elapsed training time {} max time {} latest checkpoint {}'.format(elapsed_time, setup.timeout_secs, latest_ckpt))
                    if ckpt_index == setup.num_train_steps:
                        print('Last checkpoint {} created'.format(ckpt_index))
                        break
                time.sleep(30)
                elapsed_time += 30
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as ex:
                print(ex)

        except Exception as ex:
            print(ex)
        finally:
            print('Done')
            if setup.failed():
                print('Setup failure')
            if model_ckpt and os.path.exists(model_ckpt):
                print('Successfully created checkpoint {}'.format(model_ckpt))
            end_iso = datetime.datetime.utcnow().isoformat() + 'Z'
            setup.save(start_iso, end_iso, training_complete=True)
            self.end_event.set()

def sigterm_handler(signal, frame):
    print('Train got SIGTERM')
    if train:
        train.stop()

if __name__ == '__main__':

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:

        if not util.check_env():
            exit(-1)

        s = Setup.getInstance()
        s.start()
        s.join()

        if s.failed():
            raise Exception('Setup failure')

        # train
        train = Train(0)
        train.start()
        train.join()

    except Exception as ex:
      print(ex)
      exit(-1)
