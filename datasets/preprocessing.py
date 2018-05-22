from abc import ABC, abstractmethod
import cv2
import numpy as np
from sklearn.decomposition import PCA


class Preprocessor(ABC):
    @abstractmethod
    def apply(self, img):
        pass


class CLAHEPreprocessor(Preprocessor):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        planes = cv2.split(img)
        planes[0] = self.clahe.apply(planes[0])
        img = cv2.merge(planes)
        return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)


class PCAPreprocessor(Preprocessor):
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.pca = PCA(n_components)

    def apply(self, img):
        h, w, d = img.shape
        img = np.reshape(img, (w * h, d))
        img = self.pca.fit_transform(img)
        img = np.reshape(img, (w, h, self.n_components))
        return img
