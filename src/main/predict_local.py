# !/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Run object detection model inference and output XML in PASCAL format.
Does not require MLFlow, only that model exist on a local drive, not a cloud bucket.
Included for reference.

@author: __author__
@status: __status__
@license: __license__
'''
from PIL import Image
import shutil
import os
import sys
import glob
from bs4 import BeautifulSoup, Tag
from subprocess import Popen
from object_detection.utils import visualization_utils
from object_detection.utils import label_map_util
import numpy as np
import cv2
import tensorflow as tf
from xml.dom.minidom import parse
from threading import Thread


def process_command_line():
    """
    Process command line
    :return: args object
    """

    import argparse
    from argparse import RawTextHelpFormatter

    examples = 'Examples:' + '\n\n'
    examples += 'Run inference with pretrained model checkpoint and output PASCAL annotation in r xml files in /data:\n'
    examples += '{} --model-path=/raid/models/detection/rfcn_resnet101_coco_100_smallanchor_random_crop_image'.format(
        sys.argv[0])
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description='Run inference with pretrained model checkpoint',
                                     epilog=examples)
    parser.add_argument('-i', '--image_path', action='store', help='Path with full resolution input images. '
                                                                   'If missing will extract from in_record_path',
                        required=False)
    parser.add_argument('--in_record_path', action='store', help='Path to the input data record', required=True)
    parser.add_argument('--out_xml_path', action='store', help='Path to store output in PASCAL formatted format',
                        required=True)
    parser.add_argument('--size', help='Images size in wxh', required=True, type=str)
    parser.add_argument('--model_path', action='store', help='Absolute path to model', required=True)
    parser.add_argument('--min_score_thresh', action='store', help='Minimum score threshold', required=False,
                        type=float, default=0.5)
    parser.add_argument('-l', '--label_map_path', action='store', help='Path to label map proto', required=True)
    args = parser.parse_args()
    return args


class Export(Thread):
    def __init__(self, env, inference_graph, log, model_path, last_checkpoint):
        Thread.__init__(self)
        self.env = env
        self.inference_graph = inference_graph
        self.log = log
        self.model_path = model_path
        self.last_checkpoint = last_checkpoint

    def run(self):
        with open(self.log, "w")  as out:
            print('Exporting graph')
            command = 'python {}/tensorflow_models/research/object_detection/export_inference_graph.py \
                        --input_type image_tensor \
                        --pipeline_config_path {}/pipeline.config \
                        --output_directory /tmp \
                        --trained_checkpoint_prefix {} \
                        '.format(env["APP_HOME"], self.model_path, self.last_checkpoint)
            print(command)
            out.write(command + '\n')
            p = Popen(command, env=env, shell=True)
            p.wait()
            if not os.path.exists(self.inference_graph):
                shutil.copyfile('/tmp/frozen_inference_graph.pb', self.inference_graph)
            print('Finished exporting model')


class Infer(Thread):
    def __init__(self, env, inference_graph, log, in_record_path, out_record_path):
        Thread.__init__(self)
        self.env = env
        self.inference_graph = inference_graph
        self.in_record_path = in_record_path
        self.out_record_path = out_record_path
        self.log = log

    def run(self):
        with open(self.log, "w")  as out:
            print('Running inference')
            command = 'python {}/tensorflow_models/research/object_detection/inference/infer_detections.py \
                        --inference_graph {} \
                        --discard_image_pixels=True \
                        --input_tfrecord_paths {} \
                        --output_tfrecord_path {}'.format(env["APP_HOME"],
                                                          self.inference_graph,
                                                          self.in_record_path,
                                                          self.out_record_path)
            print(command)
            out.write(command + '\n')
            p = Popen(command, env=env, stdout=out, stderr=out, shell=True)
            p.wait()
            print('Finished running inference')


def extract_images(tfrecords_filename, output_path):
    num_images = 0
    for record in tf.python_io.tf_record_iterator(tfrecords_filename):
        num_images += 1

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    tf.reset_default_graph()

    fq = tf.train.string_input_producer([tfrecords_filename], num_epochs=num_images)
    reader = tf.TFRecordReader()
    _, v = reader.read(fq)
    fk = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/height': tf.FixedLenFeature([], dtype=tf.int64),
        'image/width': tf.FixedLenFeature([], dtype=tf.int64)
    }

    ex = tf.parse_single_example(v, fk)
    image_decode = tf.decode_raw(ex['image/encoded'], tf.uint8)
    fname = tf.cast(ex['image/filename'], tf.string)
    height = tf.cast(ex['image/height'], tf.int64)
    width = tf.cast(ex['image/width'], tf.int64)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session()  as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("going to restore {} files from {}".format(num_images, tfrecords_filename))
        for i in range(num_images):
            im_, fname_, height_, width_ = sess.run([image_decode, fname, height, width])
            fname_f = fname_.decode("utf-8")
            image = Image.fromarray(im_)
            fname = os.path.join(output_path, fname_f)
            print('Writing to {}'.format(fname))
            cv2.imwrite(fname, image)

        coord.request_stop()
        coord.join(threads)


def record_to_xml(tfrecords_filename, label_map_path, xml_path, image_path, width, height, min_score_thresh):
    num_records = 0
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path)
    for record in tf.python_io.tf_record_iterator(tfrecords_filename):
        num_records += 1
    print('Found {} records'.format(num_records))
    tf.reset_default_graph()

    fq = tf.train.string_input_producer([tfrecords_filename], num_epochs=num_records)
    reader = tf.TFRecordReader()
    _, v = reader.read(fq)
    fk = {
        'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/detection/label': tf.VarLenFeature(tf.int64),
        'image/detection/score': tf.VarLenFeature(tf.float32),
        'image/detection/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/detection/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/detection/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/detection/bbox/xmax': tf.VarLenFeature(tf.float32)
    }

    ex = tf.parse_single_example(v, fk)

    print('Searching for images in {}'.format(image_path))
    images = sorted(glob.iglob(image_path + '/*.png'))

    fname = tf.cast(ex['image/filename'], tf.string)
    labels = tf.cast(ex['image/detection/label'], tf.int64)
    scores = tf.cast(ex['image/detection/score'], tf.float32)
    ymin = tf.cast(ex['image/detection/bbox/ymin'], tf.float32)
    ymax = tf.cast(ex['image/detection/bbox/ymax'], tf.float32)
    xmin = tf.cast(ex['image/detection/bbox/xmin'], tf.float32)
    xmax = tf.cast(ex['image/detection/bbox/xmax'], tf.float32)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session()  as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # set the number of images in your tfrecords file
        print("going to create {} annotations from {}".format(num_records, tfrecords_filename))
        for i in range(num_records):
            try:
                labels_, scores_, fname_, xmin_, xmax_, ymin_, ymax_ = sess.run(
                    [labels, scores, fname, xmin, xmax, ymin, ymax])

                im = Image.open(images[i])
                w, h = im.size

                scale_width = float(w) / float(width)
                scale_height = float(h) / float(height)

                boxes = np.vstack((scale_height * ymin_.values, scale_width * xmin_.values, scale_height * ymax_.values,
                                   scale_width * xmax_.values)).transpose()
                fname_f = fname_.decode("utf-8")

                infile = open('pascal_template.xml', "r")
                soup_dst = BeautifulSoup(infile.read(), 'xml')
                soup_dst.annotation.size.width.string = str(w)
                soup_dst.annotation.size.height.string = str(h)

                basename, _ = os.path.splitext(fname_f)
                xml_out = os.path.join(xml_path, basename) + '.xml'
                png_out = os.path.join(xml_path, basename) + '.png'
                print('Creating PASCAL file {} and annotated box for {}'.format(xml_out, png_out))
                scores_final = scores_.values
                labels_final = labels_.values
                max_boxes_to_draw = boxes.shape[0]
                image = cv2.imread(images[i])
                im_ = visualization_utils.visualize_boxes_and_labels_on_image_array(image,
                                                                                    boxes,
                                                                                    labels_final,
                                                                                    scores_final,
                                                                                    category_index,
                                                                                    instance_masks=None,
                                                                                    keypoints=None,
                                                                                    use_normalized_coordinates=True,
                                                                                    max_boxes_to_draw=max_boxes_to_draw,
                                                                                    min_score_thresh=min_score_thresh,
                                                                                    agnostic_mode=False,
                                                                                    line_thickness=2)
                cv2.imwrite(png_out, im_)
                for i in range(max_boxes_to_draw):
                    if scores_final is None or scores_final[i] > min_score_thresh:
                        box = tuple(boxes[i].tolist())
                        if labels_final[i] in category_index.keys():
                            class_name = category_index[labels_final[i]]['name']
                        else:
                            class_name = 'N/A'

                        ymn, xmn, ymx, xmx = box
                        print('File {} Class {} ymin {} xmin {} ymax {} xmax {}'.format(fname_f, class_name, ymn,
                                                                                        xmn, ymx, xmx))
                        # final_label = '{}_{}'.format(str(class_name), int(100 * scores_final[i]))
                        bndbox_tag = Tag(name='bndbox')
                        tag = Tag(name='xmin');
                        tag.string = str(int(xmn * width));
                        bndbox_tag.append(tag)
                        tag = Tag(name='ymin');
                        tag.string = str(int(ymn * height));
                        bndbox_tag.append(tag)
                        tag = Tag(name='xmax');
                        tag.string = str(int(xmx * width));
                        bndbox_tag.append(tag)
                        tag = Tag(name='ymax');
                        tag.string = str(int(ymx * height));
                        bndbox_tag.append(tag)

                        annotation_tag = Tag(name="object");
                        tag = Tag(name='name');
                        tag.string = str(class_name)
                        annotation_tag.append(tag)
                        tag = Tag(name='confidence');
                        tag.string = str(scores_final[i])
                        annotation_tag.append(tag)

                        tag = Tag(name='pose');
                        tag.string = 'Unspecified';
                        annotation_tag.append(tag)
                        tag = Tag(name='truncated');
                        tag.string = '0';
                        annotation_tag.append(tag)
                        tag = Tag(name='occluded');
                        tag.string = '0';
                        annotation_tag.append(tag)
                        tag = Tag(name='difficult');
                        tag.string = '0';
                        annotation_tag.append(tag)

                        annotation_tag.append(bndbox_tag)
                        soup_dst.annotation.append(annotation_tag)

                xml_temp = '/tmp/out.xml'
                print('Writing ' + xml_temp)
                f = open(xml_temp, "w")
                f.write(soup_dst.decode_contents())
                f.close()
                # a bit of hacky workaround to print a better looking xml than what beautifulsoup produces
                xmlf = parse(xml_temp)
                pretty_xml_as_string = xmlf.toprettyxml()
                # remove empty lines
                pretty_xml_as_string = os.linesep.join(
                    [s for s in pretty_xml_as_string.splitlines() if s.strip()])

                print('Writing ' + xml_out)
                with open(xml_out, 'w') as f:
                    f.write(pretty_xml_as_string)
                f.close()
            except Exception as ex:
                print(ex)
                continue

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    args = process_command_line()
    env = os.environ.copy()

    if not os.path.exists(args.out_xml_path):
        os.makedirs(args.out_xml_path)

    width = int(args.size.split('x')[0])
    height = int(args.size.split('x')[1])
    log = '{}/infer.log'.format(args.out_xml_path)

    checkpoints = sorted(glob.iglob(args.model_path + '/**/*.ckpt*'))
    if len(checkpoints) == 0:
        print('No checkpoints found')
        exit(-1)

    last_checkpoint = checkpoints[-1].split('.meta')[0]
    print('Using the checkpoint {}'.format(last_checkpoint))

    # export the graph
    inference_graph = '/tmp/frozen_inference_graph.pb'
    export = Export(env, inference_graph, log, args.model_path, last_checkpoint)
    export.run()

    # run inference
    out_record_path = '/tmp/out.record'
    infer = Infer(env, inference_graph, log, args.in_record_path, out_record_path)
    infer.run()

    if not args.image_path:
        image_path = '/tmp/imgs/'
        extract_images(args.in_record_path, image_path)
    else:
        image_path = args.image_path

    # export records to XML
    record_to_xml(out_record_path, args.label_map_path, args.out_xml_path, image_path, width, height,
                  args.min_score_thresh)
