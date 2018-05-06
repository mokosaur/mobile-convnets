import cv2
import os
import tensorflow as tf
import numpy as np


class Resizer:
    def __init__(self, shape, keep_ratio=True):
        self.shape = shape
        self.keep_ratio = keep_ratio

    def transform(self, img):
        cp = img.copy()
        if self.keep_ratio:
            dim = min(img.shape[0], img.shape[1])
            half_dim = dim // 2
            cp = img[img.shape[0] // 2 - half_dim:img.shape[0] // 2 - half_dim + dim,
                 img.shape[1] // 2 - half_dim:img.shape[1] // 2 - half_dim + dim, :]
        return [cv2.resize(cp, self.shape)]


class Rotator:
    def __init__(self, max_angle, n_samples=1, random=False):
        self.max_angle = max_angle
        self.n_samples = n_samples
        self.random = random

    def transform(self, img):
        rows, cols, _ = img.shape
        imgs = []
        for i in range(self.n_samples):
            if self.random:
                angle = np.random.randint(-self.max_angle, self.max_angle)
            else:
                angle = -self.max_angle + i * self.max_angle * 2 / self.n_samples
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            imgs.append(cv2.warpAffine(img, M, (cols, rows)))
        return imgs


def preprocess(dataset_name, pipeline=()):
    dir = os.path.dirname(os.path.abspath(__file__))
    preprocessing_dir = os.path.join(dir, dataset_name + '_preprocessed')
    if not os.path.exists(preprocessing_dir):
        os.makedirs(preprocessing_dir)
    for i, class_name in enumerate(os.listdir(dataset_name)):
        class_dir = os.path.join(preprocessing_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        idx = 0
        for j in os.listdir(dataset_name + '/' + class_name):
            im = cv2.imread(dataset_name + '/%s/%s' % (class_name, j), 1)
            im = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB)]
            for p in pipeline:
                tmp = []
                for image in im:
                    tmp += p.transform(image)
                im = tmp
            for image in im:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(class_dir, "img" + str(idx) + '.png'), image)
                idx += 1


if __name__ == '__main__':
    preprocess('cladonia', (Resizer((224, 224)),))
