

from sklearn.neighbors import KNeighborsClassifier
import cv2 as cv
import numpy as np
import os
import sys

import sift_extraction
# import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class Extract_BoW:
    def __init__(self, centroids, extractor):
        self.centroids = centroids
        self.extractor = extractor
        self.knn = KNeighborsClassifier(n_neighbors=1, weights="distance")
        self.knn.fit(centroids, range(len(centroids)))

    def extract(self, img):
        des = self.extractor.extract(img)
        try:
            index = []
            for i, arr in enumerate(des):
                if np.any(np.isnan(arr)):
                    index.append(i)
            des = np.delete(des, index, axis=0)
            pred = self.knn.predict(des)
        except:
            length = self.extractor.descriptorSize()
            des = np.zeros((1, length))
            pred = self.knn.predict(des)

        arr_count = np.zeros(len(self.centroids))
        for x in pred:
            arr_count[x] += 1
        return arr_count / len(des)


centroids = np.load(
    "/home/lmtruong1512/Codes/BTL_CSDLDPT/centroid_files/sift100_centroids128.npy")
extractor = sift_extraction.sift_extraction()
img = cv.imread(
    "/home/lmtruong1512/Codes/BTL_CSDLDPT/image_data/animals_test/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")
extract_BoW = Extract_BoW(centroids, extractor)
arr_bow = extract_BoW.extract(img)
print(arr_bow)
