import pandas as pd
import tensorflow as tf
import base64
from imagecoder import ImageCoder

def decode(x):
    p = tf.placeholder(tf.string, [960*540*3])
    try:
        p =  base64.decodebytes(bytearray(x[0]))
    except Exception as ex:
        p = base64.decodebytes(bytearray(x[0], encoding="utf8"))
    finally:
        return p

def is_png(filename):
    return '.png' in filename or '.PNG' in filename

def read_image(x):
        with open(x, "rb") as f:
            bytes_data = f.read()
            try:
                if is_png(x):
                    print('Converting {} to jpeg'.format(x))
                    image_data = coder.png_to_jpeg(tf.compat.as_bytes(bytes_data))
                    image = coder.decode_jpeg(image_data)
                else:
                    image = bytes_data
                return image
            except Exception as ex:
                print(ex)


model_path = '/Users/dcline/Downloads/model/saved_model'
tags = "serve"
coder = ImageCoder()

print('Creating initializable iterator')
filenames = ['/Users/dcline/Sandbox/avedac-tfdetect-docker/data/testimages/jpg/D0232_03HD_00-14-25_r.jpg',
             "/Users/dcline/Sandbox/avedac-tfdetect-docker/data/testimages/jpg/D0232_03HD_00-14-25_r.jpg",
             "/Users/dcline/Sandbox/avedac-tfdetect-docker/data/testimages/jpg/D0232_03HD_00-14-25_r.jpg",
             "/Users/dcline/Sandbox/avedac-tfdetect-docker/data/testimages/jpg/D0232_03HD_00-14-25_r.jpg",
             "/Users/dcline/Sandbox/avedac-tfdetect-docker/data/testimages/jpg/D0232_03HD_00-14-25_r.jpg"]

input = pd.DataFrame(data=[base64.encodebytes(read_image(x)) for x in filenames[0:5]],
                        columns=["input"])
images = input.apply(axis=1, func=decode)
print('Predicting ' + str(len(images)) + ' images...')
max_batch = 2 #min(images.shape[0], 1)

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

input_tensor_mapping = {
    tensor_column_name: g.get_tensor_by_name(tensor_info.name)
    for tensor_column_name, tensor_info in signature_def.inputs.items()
}

# We assume that output keys in the signature definition correspond to output DataFrame
# column names
output_tensors = {
    sigdef_output: g.get_tensor_by_name(tnsr_info.name)
    for sigdef_output, tnsr_info in signature_def.outputs.items()
}

with g.as_default():
        data = tf.constant(images.values)
        dataset = tf.data.Dataset.from_tensor_slices((data))
        dataset = dataset.batch(max_batch)
        next_batch = dataset.make_one_shot_iterator().get_next()

        pred_dict = {}
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            df_final = pd.DataFrame()
            count = 0
            while (True):
                try:
                    data_batch = sess.run(next_batch)
                    feed_dict = {
                        input_tensor_mapping[tensor_column_name]: data_batch
                        for tensor_column_name in input_tensor_mapping.keys()
                    }
                    raw_preds = sess.run(output_tensors, feed_dict=feed_dict)
                    print('Predicting ' + str(len(data_batch)))
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
                    indexes = []
                    for i in range(len(data_batch)):
                        indexes = indexes + [count] * 300
                        count += 1
                    pred_dict['index'] = indexes
                    # put into frame and pull out top 5 predictions by index
                    df = pd.DataFrame.from_dict(pred_dict)
                    df_top5 = df.groupby(["index"]).apply(lambda x: x.sort_values(["detection_scores"], ascending=False)[0:5]).reset_index(drop=True)
                    df_final = df_final.append(df_top5)
                    print(df_final.shape)
                except tf.errors.OutOfRangeError:
                    print("no more data")
                    break
        print(len(df_final))