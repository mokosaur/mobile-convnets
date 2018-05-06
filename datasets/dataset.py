import os
import random
from skimage import io
import numpy as np


def split(dataset_name, train_size):
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_name)
    for class_name in os.listdir(dir):
        class_path = os.path.join(dir, class_name)
        if not os.path.exists(os.path.join(class_path, 'train')):
            os.makedirs(os.path.join(class_path, 'train'))
            os.makedirs(os.path.join(class_path, 'test'))
        files = [file for file in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, file))]
        random.shuffle(files)
        split_point = int(train_size * len(files))
        for f in files[:split_point]:
            os.rename(os.path.join(class_path, f), os.path.join(class_path, 'train', f))
        for f in files[split_point:]:
            os.rename(os.path.join(class_path, f), os.path.join(class_path, 'test', f))


def load_data(dataset_name, batch_size, type='train', shuffled=True):
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_name)
    labels = os.listdir(dir)
    examples = [(file, l) for l in labels for file in os.listdir(os.path.join(dir, l, type))]
    X = []
    y = []
    if not batch_size:
        batch_size = len(examples)
    while True:
        if shuffled:
            random.shuffle(examples)
        for file, label in examples:
            X.append(io.imread(os.path.join(dir, label, type, file)))
            y.append(labels.index(label))
            if len(y) == batch_size:
                yield (np.array(X), np.array(y))
                X.clear()
                y.clear()


if __name__ == '__main__':
    split('cladonia_preprocessed', 0.8)
    # for X, y in load_data('cladonia_preprocessed', 16):
    #     print(y)
    #     print(X)
    #     break
