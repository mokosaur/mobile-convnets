import tensorflow as tf
import os
import cv2
import numpy as np
import random
import datasets.preprocessing as preprocessing


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(src, filters='RGB', preprocessor=None):
    im = cv2.imread(src, 1)
    im = cv2.resize(im, (448, 448), interpolation=cv2.INTER_CUBIC)
    if preprocessor:
        im = preprocessor.apply(im)
    if filters == 'LAB':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    elif filters == 'YUV':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    elif filters == 'RGB':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32)
    return im


def save_data(source_name, dataset_name=None, train=.8, seed=44, filters='RGB', preprocessor=None):
    if not dataset_name:
        dataset_name = source_name
    data = []
    for i, class_name in enumerate(os.listdir(source_name)):
        for j in os.listdir(source_name + '/' + class_name):
            data.append((source_name + '/%s/%s' % (class_name, j), i))
    if seed:
        data.sort(key=lambda x: x[0] + '/' + str(x[1]))
        random.seed(seed)
    random.shuffle(data)
    data = {
        'train': data[:int(train * len(data))],
        'val': data[int(train * len(data)):]
    }

    for mode in ['train', 'val']:
        filename = '%s-%s.tfrecords' % (dataset_name, mode)
        writer = tf.python_io.TFRecordWriter(filename)
        for address, label in data[mode]:
            img = load_image(address, filters, preprocessor)
            feature = {mode + '/label': _int64_feature(label),
                       mode + '/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()


# p = preprocessing.PCAPreprocessor()
# save_data('cladonia', 'cladoniapca', filters=None, preprocessor=p)
save_data('cladonia')
