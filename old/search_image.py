from image_match.goldberg import ImageSignature
import os
import time
import sys
from multiprocessing import Pool

TEST_DIR = 'val/'
TRAIN_DIR = '../../Imagenet'
gis = ImageSignature()
BATCH = 50

def generate_signature(train_image):
    return gis.generate_signature(train_image)

def main():
    sys.stdout = open("search.txt", "w")
    train_image_list = []
    all_train_image_list = []
    test_image_list = []
    all_test_image_list = []
    train_image_data = []
    test_image_data = []
    i = 0
    t = 0
    for dirpath, dirnames, filenames in os.walk(TRAIN_DIR):
        for file in filenames:
            if file.endswith(".JPEG"):
                train_image = os.path.join(dirpath, file)
                train_image_list.append(train_image)
                all_train_image_list.append(train_image)
                if len(train_image_list) == BATCH:
                    time_m = time.clock()
                    pool = Pool(BATCH)
                    results = []
                    for train_image in train_image_list:
                        results.append(pool.apply_async(generate_signature, (train_image,)))
                    pool.close()
                    pool.join()
                    for result in results:
                        train_image_data.append(result.get())
                    time_n = time.clock()
                    t = t + time_n - time_m
                    train_image_list = []
                i = i + 1
                if i % 1000 == 0 and i != 0:
                    print(i, " : ", t)
                    t = 0
    i = 0
    for file in os.listdir(TEST_DIR):
        if file.endswith(".JPEG"):
            test_image = os.path.join(TEST_DIR, file)
            test_image_list.append(test_image)
            all_test_image_list.append(test_image)
            if len(test_image_list) == BATCH:
                time_m = time.clock()
                pool = Pool(BATCH)
                results = []
                for test_image in test_image_list:
                    results.append(pool.apply_async(generate_signature, (test_image,)))
                pool.close()
                pool.join()
                for result in results:
                    test_image_data.append(result.get())
                time_n = time.clock()
                t = t + time_n - time_m
                test_image_list = []
            i = i + 1
            if i % 1000 == 0 and i != 0:
                print(i, " : ", t)
                t = 0

    i = 0
    for test_image in test_image_data:
        j = 0
        for train_image in train_image_data:
            if gis.normalized_distance(test_image, train_image) < 0.3:
                print(all_test_image_list[i])
                print(all_train_image_list[j])
            j = j + 1
        i = i + 1

if __name__ == '__main__':
    main()