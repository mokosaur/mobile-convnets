import tensorflow as tf
import os
import cv2
import numpy as np
import random
import datasets.preprocessing as preprocessing

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(src, filters='RGB', preprocessor=None):
    im = cv2.imread(src, 1)
    if im is not None:
        height, width, channels = im.shape
        min_dim = min(height, width)
        im = im[height // 2 - min_dim // 2: height // 2 + min_dim // 2,
             width // 2 - min_dim // 2: width // 2 + min_dim // 2]
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
    else:
        print(src)
        return None


def save_data(source_name, dataset_name=None, train=.8, seed=44, filters='RGB',
              preprocessor=None, preprocess_all=False):
    source_name = os.path.join(__location__, source_name)
    if not dataset_name:
        dataset_name = source_name
    data = []
    with open(os.path.join(__location__, dataset_name + '.txt'), 'w') as f:
        for i, class_name in enumerate(os.listdir(source_name)):
            f.write('%s\n' % (class_name.replace('+', ' ')))
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
        filename = os.path.join(__location__, '%s-%s.tfrecords' % (dataset_name, mode))
        writer = tf.python_io.TFRecordWriter(filename)
        if not preprocess_all:
            example_index = 1
            for address, label in data[mode]:
                print('\rLoading %s/%s %s image...' % (example_index, len(data[mode]), mode), end='')
                example_index += 1
                img = load_image(address, filters, preprocessor)
                feature = {mode + '/label': _int64_feature(label),
                           mode + '/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        else:
            example_index = 1
            images = []
            labels = []
            for address, label in data[mode]:
                print('\rLoading %s/%s %s image...' % (example_index, len(data[mode]), mode), end='')
                example_index += 1
                img = load_image(address, filters)
                images.append(img)
                labels.append(label)
            images = np.array(images)
            images = preprocessor.apply(images)
            for img, label in zip(images, labels):
                feature = {mode + '/label': _int64_feature(label),
                           mode + '/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        writer.close()


# p = preprocessing.PCAPreprocessor(2)
# save_data('cladonia', 'cladoniapca2', filters=None, preprocessor=p)
# save_data('cladonia')
save_data('moss', 'moss')
