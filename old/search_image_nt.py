from image_match.goldberg import ImageSignature
import os
import sys
import time
import tensorflow as tf

TEST_DIR = 'val/'
TRAIN_DIR = '../../Imagenet'
gis = ImageSignature()
BATCH = 100

def generate_signature(train_image):
    return gis.generate_signature(train_image)

def main():
    '''sys.stdout = open("searchnt.txt", "w")
    train_image_list = []
    test_image_list = []
    train_image_data = []
    test_image_data = []
    i = 0
    t = 0
    for dirpath, dirnames, filenames in os.walk(TRAIN_DIR):
        for file in filenames:
            if file.endswith(".JPEG"):
                time1 = time.clock()
                train_image = os.path.join(dirpath, file)
                train_image_list.append(train_image)
                train_image_data.append(gis.generate_signature(train_image))
                time2 = time.clock()
                i = i + 1
                t = t + time2 - time1
            if i % 1000 == 0 and i != 0:
                print(i, " ", t)
                t = 0

    for file in os.listdir(TEST_DIR):
        if file.endswith(".JPEG"):
            test_image = os.path.join(TEST_DIR, file)
            test_image_list.append(test_image)
            test_image_data.append(gis.generate_signature(test_image))
    i = 0
    for test_image in test_image_data:
        j = 0
        for train_image in train_image_data:
            if gis.normalized_distance(test_image, train_image) < 0.3:
                print(test_image_list[i])
                print(train_image_list[j])
            j = j + 1
        i = i + 1
'''
    correct_prediction = tf.equal(tf.argmax([[1,2,3], [4,5,6], [12,8,9]], 1), tf.argmax([[2,3,4],[6,4,2],[9,7,4]], 1))
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        print(sess.run(evaluation_step))
if __name__ == '__main__':
    main()