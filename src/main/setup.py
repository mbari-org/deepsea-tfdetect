# !/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Setup object detection models training

@author: __author__
@status: __status__
@license: __license__
'''
import os
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import threading
from threading import Thread
from object_detection.utils import label_map_util
import util
import argparser
import json
import numpy as np
import datetime
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

DEFAULT_SCRATCH_DIR = '/tmp/data'

class Setup(Thread):

    _instance = None
    _clean = False
    @staticmethod
    def getInstance():
        load_dotenv(dotenv_path=os.environ['ENV'])
        if Setup._instance == None:
            if '_CONDOR_SCRATCH_DIR' in os.environ.keys():
                Setup(os.environ['_CONDOR_SCRATCH_DIR'])
            else:
                Setup('/tmp/data')
        return Setup._instance

    def failed(self):
        return self.failure_event.is_set()

    def __del__(self):
        print('Finished')

    def training_done(self):
        training_complete = False
        if os.path.exists(self.setup_json):
            with open(self.setup_json) as json_file:
                data = json.load(json_file)
                if data['training_complete'] == 'True':
                    training_complete = True
        return training_complete

    def save(self, start_iso=datetime.datetime.utcnow().isoformat() + 'Z',
             end_iso=datetime.datetime.utcnow().isoformat() + 'Z',
             training_complete=False):
        setup_dict = {
            'val_record': self.val_record,
            'train_record': self.train_record,
            'num_train_steps': self.num_train_steps,
            'pipeline_config_out': self.pipeline_config_out,
            'image_dims': 'x'.join(map(str, self.image_dims)),
            'labels': str(self.labels),
            'start_iso': start_iso,
            'end_iso': end_iso,
            'training_complete': str(training_complete),
            'notes': self.notes
        }
        if os.path.exists(self.setup_json):
            os.remove(self.setup_json)
        with open(self.setup_json, 'w') as json_file:
            json.dump(setup_dict, json_file)

    def __init__(self, scratch_dir=DEFAULT_SCRATCH_DIR):
        Thread.__init__(self)
        if Setup._instance != None:
            raise Exception('Setup is a singleton')
        else:
            print('Setup')
            self.failure_event = threading.Event()
            parser = argparser.ArgParser()
            self.parser_args = parser.parse_args()
            parser.summary()
            self.scratch_dir = scratch_dir
            self.checkpoint_dir = os.path.join(self.scratch_dir, 'models')
            self.model_dir = self.checkpoint_dir
            self.identifier = self.parser_args.model_template
            self.timeout_secs = self.parser_args.timeout_secs
            self.notes = self.parser_args.notes
            self.setup_json = os.path.join(self.scratch_dir, 'setup.json')
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            Setup._instance = self

    @property
    def args(self):
        return self.parser_args


    def run(self):
        try:
            pipeline_config_in = '{}/{}'.format(self.scratch_dir, self.parser_args.model_template)
            util.clean_dir(self.scratch_dir)
            util.download_s3(os.environ['MLFLOW_S3_ENDPOINT_URL'], self.parser_args.data_bucket, self.scratch_dir)
            if not os.path.exists(pipeline_config_in):
                raise Exception('Missing {}'.format(pipeline_config_in))
            self.pipeline_config_out = '{}/pipeline.config'.format(self.model_dir)
            self.val_record = '{}/val.record'.format(self.scratch_dir)
            self.train_record = '{}/train.record'.format(self.scratch_dir)
            label_map_path = '{}/label_map.pbtxt'.format(self.scratch_dir)
            self.conda_yaml = os.path.join(os.path.dirname(__file__), 'conda.yaml')
            self.num_train_steps = self.parser_args.num_train_steps
            self.image_dims = np.array([int(x) for x in self.parser_args.image_dims.split("x")], dtype=np.int32)
            self.image_mean = np.array([np.float32(x) for x in self.parser_args.image_mean.split(" ")])
            label_map_dict = label_map_util.get_label_map_dict(label_map_path)
            num_classes = len(label_map_dict.keys())
            sorted_labels = sorted(label_map_dict.items(), key=lambda kv: kv[1])
            self.labels = ','.join([l[0] for l in sorted_labels])


            # calculate total number of test examples and width/height of examples in separate process to free GPU resources
            num_eval_examples, height, width = util.run_in_separate_process(self.find_record_meta, self.val_record)

            # width and height of training data should match that specified for training the model
            if self.image_dims[0] != width or self.image_dims[1] != height:
                raise Exception('tensorflow record {} {} dims {}x{} should match that specified in argument'
                                ' as {}x{}'.format(self.val_record, self.train_record, width, height, self.image_dims[0],
                                                   self.image_dims[1]))

            print('Downloading {} to {}'.format(self.parser_args.checkpoint_url, self.checkpoint_dir))
            checkpoint_path = util.download(self.parser_args.checkpoint_url, self.checkpoint_dir)

            # export inference graph
            # exporting is not needed, but put here for reference. Sometimes models are old in the zoo
            # and need to be reexported. This should be put into a separate thread!!
            '''base_dir = os.path.dirname(my_check_point)
            input_type = 'image_tensor'
            cfg = os.path.join(base_dir, 'pipeline.config')
            ckpt_prefix = os.path.join(base_dir, 'model.ckpt')
            export_dir = os.path.join(base_dir, 'saved_model')
            if os.path.exists(export_dir):
                shutil.rmtree(export_dir)

            cmd = '{} {}/tensorflow_models/research/object_detection/export_inference_graph.py \
                --input_type={} \
                --pipeline_config_path={} \
                --trained_checkpoint_prefix={} \
                --output_directory={}'.format(sys.executable, os.environ["TF_HOME"], input_type, cfg,
                                               ckpt_prefix, base_dir) # creates saved_model dir automagically
            print('{}'.format(cmd))
            proc = Popen(cmd, shell=True, preexec_fn=os.setsid)
            saved_model = os.path.join(export_dir, 'saved_model.pb')
            print('Exporting inference graph to {} '.format(saved_model))
            while (util.check_pid(proc.pid) and not os.path.exists(saved_model)):
                time.sleep(3)

            try:
                #Kill all child processes of the group and wait a few seconds for GPU memory to release
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(5)
            except Exception as ex:
                print(ex)'''

            # replace template config
            pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

            with tf.gfile.GFile(pipeline_config_in, "r") as f:
                proto_str = f.read()
                text_format.Merge(proto_str, pipeline_config)

            pipeline_config.train_input_reader.label_map_path = label_map_path
            pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = self.train_record
            pipeline_config.eval_input_reader[0].label_map_path = label_map_path
            pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = self.val_record
            pipeline_config.eval_config.num_examples = num_eval_examples
            pipeline_config.eval_config.include_metrics_per_category = True
            if pipeline_config.eval_config.metrics_set is None:
                raise Exception('Need to set pipeline_config.eval_config.metrics_set="coco_detection_metrics" in {}'.format(pipeline_config_in)) 
            if 'coco_detection_metrics' not in pipeline_config.eval_config.metrics_set:
                raise Exception('Need to set pipeline_config.eval_config.metrics_set="coco_detection_metrics" in {}'.format(pipeline_config_in)) 
            pipeline_config.train_config.num_steps = self.parser_args.num_train_steps
            pipeline_config.train_config.fine_tune_checkpoint = checkpoint_path
            if self.parser_args.model_arch == 'ssd':
                pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width=width
                pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height=height
                pipeline_config.model.ssd.num_classes=num_classes
            elif self.parser_args.model_arch == 'faster_rcnn' or self.parser_args.model_arch == 'frcnn':
                pipeline_config.model.faster_rcnn.image_resizer.keep_aspect_ratio_resizer.min_dimension=min(height, width)
                pipeline_config.model.faster_rcnn.image_resizer.keep_aspect_ratio_resizer.max_dimension=max(height, width)
                pipeline_config.model.faster_rcnn.num_classes=num_classes
            elif self.parser_args.model_arch == 'rfcn':
                pipeline_config.model.rfcn.image_resizer.keep_aspect_ratio_resizer.min_dimension=min(height, width)
                pipeline_config.model.rfcn.image_resizer.keep_aspect_ratio_resizer.max_dimension=max(height, width)
                pipeline_config.model.rfcn.num_classes=num_classes

            means = [(float(i)/float(255)) for i in self.parser_args.image_mean.split(' ')]
            pipeline_config.train_config.data_augmentation_options[0].subtract_channel_mean.means[:]=means

            config_text = text_format.MessageToString(pipeline_config)
            with tf.gfile.Open(self.pipeline_config_out, "wb") as f:
                f.write(config_text)
            if not os.path.exists(self.pipeline_config_out):
                raise Exception('Cannot create {}'.format(self.pipeline_config_out))
            self.save()
            print('Done')
            return
        except Exception as ex:
            print(ex)
            self.failure_event.set()

    @staticmethod
    def find_record_meta(record):
        i = 0
        for _ in tf.python_io.tf_record_iterator(record):
            i += 1

        # Get image dimensions from first file in the tensorflow record
        # convert filenames to a queue for an input pipeline.
        fname_queue = tf.train.string_input_producer([record], num_epochs=None)

        # object to read records
        record_reader = tf.TFRecordReader()

        # read the full set of features for a single example
        _, example = record_reader.read(fname_queue)

        # parse the full example into its component features.
        features = tf.parse_single_example(
            example,
            features={
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64)
            })

        height = tf.cast(features['image/height'], tf.int64)
        width = tf.cast(features['image/width'], tf.int64)

        with tf.Session()  as sess:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord)
            h, w = sess.run([height, width])
            return i, h, w
