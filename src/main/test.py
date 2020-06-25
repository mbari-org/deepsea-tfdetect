# !/usr/bin/env python
__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Runs testing on object detection models

@author: __author__
@status: __status__
@license: __license__
'''
import os
import sys
import subprocess
import signal
import time
import threading
from threading import Thread
import util
from setup import Setup
from dotenv import load_dotenv

class Test(Thread):

    def stop(self):
        self.stop_event.set()

    def stopped(self):
        return self.stop_event.is_set()

    def __init__(self, dev):
        Thread.__init__(self)
        self.stop_event = threading.Event()
        self.end_event = threading.Event()
        self.dev = dev

    def run(self):
        setup = Setup.getInstance()
        load_dotenv(dotenv_path=os.environ['ENV'])
        env = os.environ.copy()
        print('Starting testing with {} {}'.format(setup.pipeline_config_out, setup.scratch_dir))

        # run continuous testing; will test on each checkpoint
        cmd = '{} {}/tensorflow_models/research/object_detection/model_main.py \
                    --num_train_steps {}  --pipeline_config_path {} --checkpoint_dir={} --model_dir={} '\
                                                .format(sys.executable, os.environ["TF_HOME"],
                                                        setup.num_train_steps,
                                                        setup.pipeline_config_out,
                                                        setup.checkpoint_dir,
                                                        setup.model_dir)
        try:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(self.dev)
            proc = subprocess.Popen(cmd, env=env, shell=True, preexec_fn=os.setsid)

            while not self.stopped() and util.check_pid(proc.pid) and not setup.training_done() and not setup.failed():
                time.sleep(60)
            if setup.training_done(): 
                elapsed_time = 0
                print('Waiting for events to complete...')
                while elapsed_time < 30:
                    time.sleep(30)
                    elapsed_time += 30
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as ex:
                print(ex)
        except Exception as ex:
            print(ex)
        finally:
            self.end_event.set()

def sigterm_handler(signal, frame):
    print('Test got SIGTERM. May take a minute to export the model and log metrics.')
    if test:
        test.stop()

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

        test = Test(1)
        test.start()
        test.join()
        if test.stopped():
            raise Exception('Testing stopped prematurely')

    except Exception as ex:
        print(ex)


