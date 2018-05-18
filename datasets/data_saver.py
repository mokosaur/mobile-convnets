import tensorflow as tf
import os
import cv2
import numpy as np
import random


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(src):
    im = cv2.imread(src, 1)
    im = cv2.resize(im, (448, 448), interpolation=cv2.INTER_CUBIC)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32)
    return im


def save_data(dataset_name):
    data = []
    for i, class_name in enumerate(os.listdir(dataset_name)):
        for j in os.listdir(dataset_name + '/' + class_name):
            data.append((dataset_name + '/%s/%s' % (class_name, j), i))
    random.shuffle(data)
    data = {
        'train': data[:int(.8 * len(data))],
        'val': data[int(.8 * len(data)):]
    }

    for mode in ['train', 'val']:
        filename = '%s-%s.tfrecords' % (dataset_name, mode)
        writer = tf.python_io.TFRecordWriter(filename)
        for address, label in data[mode]:
            img = load_image(address)
            feature = {mode + '/label': _int64_feature(label),
                       mode + '/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()


save_data('cladonia')
