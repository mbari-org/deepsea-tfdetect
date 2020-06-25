import argparse
from argparse import RawTextHelpFormatter
import sys



class ArgParser():

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        self.examples = 'Examples:' + '\n\n'
        self.examples += 'Train models on s3://data for 1000 epochs with experiment "object detect" and save to s3://objectdetect \n'
        self.examples += '{} --model_template=s3://objectdetectionmodeltemplates/faster_rcnn_resnet101_coco_300_smallanchor_random_crop_image_mean_stride8.pipeline.template' \
                    ' --data_bucket s3://test  --num_train_steps 1000 --image_mean "108.79285239 131.44784338 106.67269749"' \
                    ' --checkpoint_url=http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz' \
            .format(sys.argv[0])
        self.parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                         description='Train object detector',
                                         epilog=self.examples)
        self.parser.add_argument('--nvidia_dev', action='store', help='NVIDIA device numbers',  required=False, nargs='+')
        self.parser.add_argument('--data_bucket', action='store',
                            help='Path/bucket to the train/val data. Assumes records are '
                                 'named train.record and val.record in that bucket', required=True)
        self.parser.add_argument('--notes', action='store', help='Experiment notes', required=False)
        self.parser.add_argument('--image_mean', action='store', help='Image mean', required=True)
        self.parser.add_argument('--model_arch', action='store', help='Model architecture (ssd, frcnn, rfcn)', required=True)
        self.parser.add_argument('--image_dims', action='store', help='Image dimensions wxhxd e.g. 300x300x3', required=True)
        self.parser.add_argument('--model_template', action='store', help='Path/bucket to the model pipeline template',
                            required=True)
        self.parser.add_argument('--num_train_steps', action='store', help='Number of training steps', required=False,
                            default=1,  type=int)
        self.parser.add_argument('--timeout_secs', action='store', help='Timeout training in seconds', required=False,
                            type=int, default=120)
        self.parser.add_argument('--checkpoint_url', action='store', help='URL to model tar containing checkpoint',
                            required=True)

    def parse_args(self):
        self.args = self.parser.parse_args()
        return self.args

    def summary(self):
        print("data_bucket:", self.args.data_bucket)
        print("notes:", self.args.notes)
        print("model_template:", self.args.model_template)
        print("model_arch:", self.args.model_arch)
        print("timeout_secs:", self.args.timeout_secs)
        print("num_train_steps:", self.args.num_train_steps)
        print("checkpoint_url:", self.args.checkpoint_url)
        print("image_mean:", self.args.image_mean)
        print('image_dims', self.args.image_dims)

if __name__ == '__main__':
    parser = ArgParser()
    args = parser.parse_args()
    print(args)
