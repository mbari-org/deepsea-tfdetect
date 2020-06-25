# !/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Custom MLFlow python function wrapper around a TensorFlow object detection model.

@author: __author__
@status: __status__
@license: __license__
'''
import yaml
import pandas as pd
import tensorflow as tf
import os
import mlflow
import mlflow.keras
from mlflow.utils import PYTHON_VERSION
import PIL
import numpy as np
import matplotlib
import base64

MAX_BATCH = 1000

class ObjectDetPyfunc(object):
    """
    Object detection model with image resizing

    Custom MLFlow python function wrapper around a TensorFlow object detection model.
    The wrapper provides image resizing per model requirements.txt.
    The input to the model is base64 encoded image binary data
    The output is a Pandas data frame with the predicted class label, id, and probabilities for each
    class.
    """

    def __init__(self, model, graph, session, signature_def, image_dims, image_mean, labels):
        self.model = model
        self.graph = graph
        self.session = session
        self.image_dims = image_dims
        self.image_mean = image_mean
        self.labels = labels
        self.input_tensor_mapping = {
            tensor_column_name: graph.get_tensor_by_name(tensor_info.name)
            for tensor_column_name, tensor_info in signature_def.inputs.items()
        }
        self.output_tensors = {
            sigdef_output: graph.get_tensor_by_name(tnsr_info.name)
            for sigdef_output, tnsr_info in signature_def.outputs.items()
        }

    def predict(self, input):
        """
        Generate predictions for the data.

        :param input: pandas.DataFrame with one column containing images to be scored. The image
                     column must contain base64 encoded binary content of the image files.

        :return: pandas.DataFrame containing predictions with the following schema:
                     Predicted class: string,
                     Predicted class index: int,
                     Probability(class==0): float,
                     ...,
                     Probability(class==N): float,
        """
        return self.predict_images(input)

    def predict_images(self, input):
        """
        Generate predictions for input images.
        :param input: binary image data
        :return: predicted probabilities for each class
        """
        def label_map(index):
            return self.labels[int(index) - 1]  # label map files are indexed from 1

        def decode(x):
            p = tf.placeholder(tf.string, [self.image_dims[0]*self.image_dims[1]*3])
            try:
                p = base64.decodebytes(bytearray(x[0]))
            except Exception:
                p = base64.decodebytes(bytearray(x[0], encoding="utf8"))
            finally:
                return p

        images = input.apply(axis=1, func=decode)
        max_batch = min(images.shape[0], MAX_BATCH)

        df_final = pd.DataFrame()
        with self.graph.as_default():
            data = tf.constant(images.values)
            dataset = tf.data.Dataset.from_tensor_slices((data))
            dataset = dataset.batch(max_batch)
            next_batch = dataset.make_one_shot_iterator().get_next()

            print('Running prediction in batches')
            ttl_images = 0
            with self.session.as_default():
                print('initializing iterator')
                self.session.run(tf.global_variables_initializer())
                while(True):
                    try:
                        data_batch = self.session.run(next_batch)
                        feed_dict = {
                            self.input_tensor_mapping[tensor_column_name]: data_batch
                            for tensor_column_name in self.input_tensor_mapping.keys()
                        }
                        num_images = len(data_batch)
                        print('Predicting ' + str(num_images))
                        raw_preds = self.session.run(self.output_tensors, feed_dict=feed_dict)
                        print('Session run complete ')

                        # create dictionary of predictions
                        pred_dict = {}
                        for column_name, values in raw_preds.items():
                            if column_name == 'detection_boxes':
                                pred_dict['ymn'] = values[:, :, 0].reshape(-1)
                                pred_dict['xmn'] = values[:, :, 1].reshape(-1)
                                pred_dict['ymx'] = values[:, :, 2].reshape(-1)
                                pred_dict['xmx'] = values[:, :, 3].reshape(-1)
                            else:
                                pred_dict[column_name] = values.ravel()

                        # add an index to use in sorting
                        print('Getting top 5')
                        step = int(len(pred_dict['xmx'])/num_images)
                        indexes = []
                        for i in range(len(data_batch)):
                            indexes = indexes + [ttl_images] * step
                            ttl_images += 1
                        pred_dict['index'] = indexes

                        # put into frame and pull out top 5 predictions by index
                        df = pd.DataFrame.from_dict(pred_dict)
                        df_top5 = df.groupby(["index"]).apply(
                            lambda x: x.sort_values(["detection_scores"], ascending=False)[0:5]).reset_index(drop=True)


                        df_final = df_final.append(df_top5)
                        print('Done with batch!')

                    except tf.errors.OutOfRangeError:
                        print('No more data')
                        break
                    except Exception as ex:
                        print(ex)
                        break

        # add in class labels
        def class_name(detection_class):
            return self.labels[int(detection_class)]
        df_final['detection_label'] = df_final.apply(lambda x: class_name(x['detection_classes']), axis=1)
        return df_final

def log_model(tmp, data_path, tags, keys, labels, image_dims, image_mean, artifact_path='model'):
    '''
    Creates a custom Pyfunc object detection  model for running inference

    :param tmp: temporary directory to use for logging artifacts
    :param data_path: temporary directory with model artifacts
    :param labels: array of class labels
    :param image_dims: image dimensions model expects (300,300)
    :param image_mean: numpy array with RGB image mean
    :param artifact_path: run-relative artifact path to which to log the Python model
    :return:
    '''
    conf = {
        "image_dims": 'x'.join(map(str, image_dims)),
        "image_mean": ",".join(map(str, image_mean)),  # Image mean of training images
        "labels": str(labels),
        "meta_graph_tags": str(tags),
        "signature_def_key": str(keys)
    }
    with open(os.path.join(data_path, "conf.yaml"), "w") as f:
        yaml.safe_dump(conf, stream=f)

    conda_env = tmp.path("conda_env.yaml")
    with open(conda_env, "w") as f:
        f.write(conda_env_template.format(python_version=PYTHON_VERSION,
                                          tf_name=tf.__name__,  # can have optional -gpu suffix
                                          tf_version=tf.__version__,
                                          pillow_version=PIL.__version__,
                                          mlflow_version=mlflow.__version__,
                                          matplotlib_version=matplotlib.__version__))

    mlflow.pyfunc.log_model(artifact_path=artifact_path,
                            loader_module=__name__,
                            code_path=[__file__],
                            data_path=data_path,
                            conda_env=conda_env)

def _load_pyfunc(path):
    """
    Load the ObjectDetPyfunc model.
    """
    with open(os.path.join(path, "conf.yaml"), "r") as f:
        conf = yaml.safe_load(f)
    model_path = os.path.join(path, "saved_model")
    image_dims = [int(x) for x in conf["image_dims"].split("x")]
    image_mean = np.array([np.float32(x) for x in conf["image_mean"].split(",")])
    labels = [x for x in conf["labels"].split(',')]
    tags = conf["meta_graph_tags"]
    key = conf["signature_def_key"]
    print('dims {} labels {}'.format(image_dims, labels))

    with tf.Graph().as_default() as g:
        with tf.Session().as_default() as sess:
            print('tags {} model_path {}'.format(tags, model_path))
            model = tf.saved_model.loader.load(
                sess=sess,
                tags=[tags],
                export_dir=model_path)
            if 'serving_default' not in model.signature_def:
                raise Exception('Could not find signature def key serving_default')
            signature_def = model.signature_def['serving_default']

    return ObjectDetPyfunc(model, g, sess, signature_def, image_dims, image_mean, labels)

conda_env_template = """        
name: object_detector
channels:
  - defaults
  - anaconda
dependencies:
  - python=={python_version}
  - pip:    
    - tensorflow=={tf_version}
    - pillow=={pillow_version}
    - mlflow=={mlflow_version}
    - matplotlib=={matplotlib_version}
    - gevent
"""
