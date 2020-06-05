
import time
import math
import random
import os
import sys
import numpy as np
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from feature_extraction import sift_extraction
from Extract_BOW import extract_bow

def split_data(dir_path, ratio):
    # create list of animal_paths
    image_paths = []
    animal_categories = os.listdir(dir_path)
    for animal_category in animal_categories:
        animal_category_path = os.path.join(dir_path, animal_category)
        animals = os.listdir(animal_category_path)
        for animal in animals:
            image_path = os.path.join(animal_category_path, animal)
            image_paths.append(image_path)
    # split train set and test set
    random.shuffle(image_paths)
    partition = int(len(image_paths)*ratio)

    train_set = image_paths[:partition]
    test_set = image_paths[partition:]
    return (train_set, test_set)


def extract_encode_label(dir_path, centroids, extractor):
    if not os.path.exists("file_encode_label.npy"):
        # find label(category) and encode(keypoints after BoW) and save to a dictionary
        file_encode_label = {}
        extract_BoW = extract_bow.extract_bow(centroids, extractor)
        num = 0
        animal_categories = os.listdir(dir_path)
        for animal_category in animal_categories:
            animal_category_path = os.path.join(dir_path, animal_category)
            animals = os.listdir(animal_category_path)
            for animal in animals:
                img_path = os.path.join(animal_category_path, animal)
                img = cv.imread(img_path)
                img_encode = extract_BoW.extract(img)
                img_label = animal_category
                file_encode_label[img_path] = (img_encode, img_label)
                num += 1
                print("encoding + labeling image:", num, "completed")
        np.save('file_encode_label.npy', file_encode_label)
    else:
        file_encode_label = np.load('file_encode_label.npy',allow_pickle='TRUE').item()
    return file_encode_label
def prepare_data(dir_path, ratio, centroids, extractor):
    # extract encode_label for each image in data
    print('extract encode_label for each image in dataset:')
    time1 = time.time()
    file_encode_label = extract_encode_label(dir_path, centroids, extractor)
    time2 = time.time()
    print("Time to encode_label data:", time2-time1)
    # split train and test set
    train_set, test_set = split_data(dir_path, ratio)
    print("splited dataset")
    return (train_set, test_set, file_encode_label)

def training_knn(train_set, file_encode_label):
    # Training begin:
    print("Begin training")
    time1 = time.time()
    num_neigbors = 1
    trainX = [file_encode_label[i][0] for i in train_set]
    trainY = [file_encode_label[i][1] for i in train_set]
    clf = KNeighborsClassifier(n_neighbors=num_neigbors, weights='distance')
    clf.fit(trainX, trainY)
    time2 = time.time()
    print("Training completed within", time2-time1)
    return clf
def test_result(test_set, clf):
    #Testing and validate
    right = 0
    testX = [file_encode_label[i][0] for i in test_set]
    testY = [file_encode_label[i][1] for i in test_set]
    for index, encode in enumerate(testX):
        pred = clf.predict([encode])
        if pred == testY[index]:
            right += 1

    accuracy = right/len(testX)
    return accuracy
def test_result_with_display(test_set, clf):
    #Testing and validate
    right = 0
    testX = [file_encode_label[i][0] for i in test_set]
    testY = [file_encode_label[i][1] for i in test_set]

    fig = plt.figure(figsize=(9, 9))
    size = int(math.sqrt(len(testX)))

    for index, encode in enumerate(testX):
        pred = clf.predict([encode])
        img = cv.imread(test_set[index])
        sub = fig.add_subplot(size+1, size+1, index+1)
        sub.set_xticks([])
        sub.set_yticks([])
        if pred != testY[index]:
            sub.set_title(pred, color="r")
        else:
            sub.set_title(pred)
        plt.imshow(img)
        if pred == testY[index]:
            right += 1
    accuracy = right/len(testX)
    fig.suptitle(f"Rate: {right}/{len(testX)}~{accurate}", fontsize=16)
    return accuracy

    


# extractor
# sift_extractor = sift_extract.sift_extract()


# dict_extractor = {"brief": brief_extractor, "brisk": brisk_extractor,
#                   "harrislaplace_CM": harrislaplace_CM_extractor, "harrislaplace_ICM": harrislaplace_ICM_extractor,
#                   "sift": sift_extractor, "sift100": sift100_extractor,
#                   "sift_CM": sift_CM_extractor, "sift_ICM": sift_ICM_extractor,
#                   "sift100_CM": sift100_CM_extractor, "sift100_ICM": sift100_ICM_extractor,
#                   "surf64": surf64_extractor, "surf128": surf128_extractor
#                   }


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('path_centroids')
#     parser.add_argument('extractor')
#     options = parser.parse_args()
#     dirpath = "/home/lmtruong/Pictures/data"
#     ratio = 0.8
#     centroids = np.load(options.path_centroids)
#     extractor = sift_extract.sift_extract()

#     accuracy = validate(dirpath, ratio, centroids, extractor)
#     plt.show()

    # sum = 0
    # for i in range(50):
    #     accuracy = validate(dirpath, ratio, centroids, extractor)
    #     sum += accuracy
    #     print(accuracy)
    # print("Average precise:", sum/50)

# use: python3 train_test.py /home/lmtruong/Documents/Work_Project/Data/Centroid_extract/sift100_centroids256.npy sift100


dir_path = "/home/lmtruong1512/Pictures/Data/animals10/img-test"
path_centroids = "/home/lmtruong1512/codes/BTL1/centroid_files/sift_centroids128.npy"
ratio = 0.8
centroids = np.load(path_centroids)
extractor = sift_extraction.sift_extraction()
print('begin:')
train_set, test_set, file_encode_label = prepare_data(dir_path, ratio, centroids, extractor)
count = 0
sum_accuracy = 0
while count < 1000:
    train_set, test_set = split_data(dir_path, ratio)
    clf = training_knn(train_set, file_encode_label)
    accuracy = test_result(test_set, clf)
    count += 1
    sum_accuracy += accuracy
    print("accuracy:", accuracy)
final_accuracy = sum_accuracy/count
print("final_accuracy:", final_accuracy)
# plt.show()
