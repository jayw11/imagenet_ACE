#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# testing dateset -> val set

import glob
import os
import random
import numpy as np
import tensorflow as tf
import time
import sys
import pickle
from fbpca import pca
from tensorflow.python.platform import gfile


BOTTLENECK_TENSOR_SIZE = 2048

# tensor name of the output of Inception-v3
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# tensor name of the image input
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# Inception-v3 model
MODEL_DIR = 'model/'
MODEL_FILE = 'tensorflow_inception_graph.pb'

# testing dataset(validation dataset of imagenet 2012)
TEST_DIR = 'val/'
TEST_GROUND_TRUTH_FILE = 'ILSVRC2012_validation_ground_truth.txt'
TEST_SIZE = 50000

# use to store bottleneck value
CACHE_DIR = 'tmp/bottleneck/'

# training dataset(training dataset of imagenet 2012)
INPUT_DATA = 'tmp/bottleneck/'

# define the parameters
LEARNING_RATE = 0.01
STEPS = 20000
BATCH = 128

# how many training and validation images we want to use
TRAINING_SIZE = 1282167

bottlenecks = np.zeros((TRAINING_SIZE, 2048), dtype=float)
ground_truths = np.zeros((TRAINING_SIZE, 1000), dtype=float)


def get_bottlenecks():
    svd_bn_0 = pickle.load(open('svd_bn_0.pkl', 'rb'))
    svd_bn_1 = pickle.load(open('svd_bn_1.pkl', 'rb'))
    #svd_bn = np.concatenate((svd_bn_0, svd_bn_1), axis=0)

    cls_bn_0 = pickle.load(open('cls_bn_0.pkl', 'rb'))
    cls_bn_1 = pickle.load(open('cls_bn_1.pkl', 'rb'))
    cls_bn_2 = pickle.load(open('cls_bn_2.pkl', 'rb'))
    cls_bn_3 = pickle.load(open('cls_bn_3.pkl', 'rb'))
    cls_bn_4 = pickle.load(open('cls_bn_4.pkl', 'rb'))
    bottlenecks = np.concatenate((svd_bn_0, svd_bn_1, cls_bn_0, cls_bn_1, cls_bn_2, cls_bn_3, cls_bn_4), axis=0)
    print(bottlenecks.shape)
    return bottlenecks #svd_bn, cls_bn


def get_ground_truth():
    svd_gt_0 = pickle.load(open('svd_gt_0.pkl', 'rb'))
    svd_gt_1 = pickle.load(open('svd_gt_1.pkl', 'rb'))
    #svd_gt = np.concatenate([svd_gt_0, svd_gt_1], axis=0)

    cls_gt_0 = pickle.load(open('cls_gt_0.pkl', 'rb'))
    cls_gt_1 = pickle.load(open('cls_gt_1.pkl', 'rb'))
    cls_gt_2 = pickle.load(open('cls_gt_2.pkl', 'rb'))
    cls_gt_3 = pickle.load(open('cls_gt_3.pkl', 'rb'))
    cls_gt_4 = pickle.load(open('cls_gt_4.pkl', 'rb'))
    ground_truths = np.concatenate([cls_gt_0, cls_gt_1, cls_gt_0, cls_gt_1, cls_gt_2, cls_gt_3, cls_gt_4], axis=0)
    print(ground_truths.shape)
    return ground_truths #svd_gt, cls_gt


def get_random_cached_bottlenecks(MA, MB):
    bottlenecks = []
    ground_truths = []
    for i in range(BATCH):
        x = random.randint(1, len(MA)-1)
        bottlenecks.append(MA[x])
        ground_truths.append(MB[x])
    return bottlenecks, ground_truths


# 这个函数获取全部的测试数据。在最终测试的时候需要在所有的测试数据上计算正确率。
def get_test_bottlenecks():
    test_gt = pickle.load(open('test_gt.pkl', 'rb'))
    test_bn = pickle.load(open('test_bn.pkl', 'rb'))

    return test_bn, test_gt


def svd_test(singular_value_number, n_classes, bottlenecks_t2, ground_truth_t2, test_bottlenecks, test_ground_truth, v):
    bottleneck_input = tf.placeholder(tf.float32, [None, singular_value_number], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
    # 定义一层全连接层来解决新的图片分类问题。
    # 因为训练好的Inception-v3模型已经将原始的图片抽象为了更加容易分类的特征向量了，所以不需要再训练那么复杂的神经网络来完成这个新的分类任务。
    '''with tf.name_scope('second_last_training_ops'):
        weights_1 = tf.Variable(tf.truncated_normal([singular_value_number, singular_value_number], stddev=0.001))
        biases_1 = tf.Variable(tf.zeros([singular_value_number]))
        logits_1 = tf.matmul(bottleneck_input, weights_1) + biases_1
        final_tensor_1 = tf.nn.relu(logits_1)
    with tf.name_scope('final_training_ops'):
        weights_2 = tf.Variable(tf.truncated_normal([singular_value_number, n_classes], stddev=0.001))
        biases_2 = tf.Variable(tf.zeros([n_classes]))
        logits_2 = tf.matmul(final_tensor_1, weights_2) + biases_2
        final_tensor_2 = tf.nn.softmax(logits_2)'''
    with tf.name_scope('final_training_ops'):
        weights_2 = tf.Variable(tf.truncated_normal([singular_value_number, n_classes], stddev=0.001))
        biases_2 = tf.Variable(tf.zeros([n_classes]))
        logits_2 = tf.matmul(bottleneck_input, weights_2) + biases_2
        final_tensor_2 = tf.nn.softmax(logits_2)
    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_2, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor_2, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #bottlenecks_t1_array = np.array(bottlenecks_t1)
    #(u, sigma, vt) = pca(bottlenecks_t1_array, singular_value_number)
    #v = vt.transpose()
    v = v[:, 0:singular_value_number]
    bottlenecks_after_svd_t2 = np.dot(bottlenecks_t2, v)
    #bottlenecks_after_svd_v = np.dot(bottlenecks_t1, v)
    test_bottlenecks_after_svd = np.dot(test_bottlenecks, v)

    with tf.Session() as sess2:
        tf.global_variables_initializer().run()
        # 训练过程
        for i in range(STEPS):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(bottlenecks_after_svd_t2, ground_truth_t2)
            sess2.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
            # 在验证集上测试正确率。
            '''if i % 500 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(bottlenecks_after_svd_v, ground_truth_t1)
                validation_accuracy = sess2.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%'
                    % (i, BATCH, validation_accuracy * 100))'''

    # 在最后的测试数据上测试正确率
        test_accuracy = sess2.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks_after_svd,
                                                         ground_truth_input: test_ground_truth})
        #print(sess2.run(tf.argmax(final_tensor, 1), feed_dict={bottleneck_input: test_bottlenecks_after_svd}))
    print('Final test accuracy (svd %d) = %.1f%%' % (singular_value_number, test_accuracy * 100))


def main():
    # 读取所有图片。
    #sys.stdout = open("test.txt", "w")
    n_classes = 1000

    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

    # 定义一层全连接层来解决新的图片分类问题。
    # 因为训练好的Inception-v3模型已经将原始的图片抽象为了更加容易分类的特征向量了，所以不需要再训练那么复杂的神经网络来完成这个新的分类任务。
    '''with tf.name_scope('second_last_training_ops'):
        weights_1 = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, BOTTLENECK_TENSOR_SIZE], stddev=0.001))
        biases_1 = tf.Variable(tf.zeros([BOTTLENECK_TENSOR_SIZE]))
        logits_1 = tf.matmul(bottleneck_input, weights_1) + biases_1
        final_tensor_1 = tf.nn.relu(logits_1)
    with tf.name_scope('final_training_ops'):
        weights_2 = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases_2 = tf.Variable(tf.zeros([n_classes]))
        logits_2 = tf.matmul(final_tensor_1, weights_2) + biases_2
        final_tensor_2 = tf.nn.softmax(logits_2)'''
    with tf.name_scope('final_training_ops'):
        weights_2 = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases_2 = tf.Variable(tf.zeros([n_classes]))
        logits_2 = tf.matmul(bottleneck_input, weights_2) + biases_2
        final_tensor_2 = tf.nn.softmax(logits_2)

    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_2, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor_2, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()

        test_bottlenecks, test_ground_truth = get_test_bottlenecks()

        bottlenecks_t = get_bottlenecks()
        ground_truth_t = get_ground_truth()

        '''bottlenecks_t = np.concatenate((bottlenecks_t1, bottlenecks_t2))
        del bottlenecks_t1, bottlenecks_t2
        ground_truth_t = np.concatenate((ground_truth_t1, ground_truth_t2))
        del ground_truth_t1,ground_truth_t2'''

        #训练过程
        for i in range(STEPS):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(bottlenecks_t, ground_truth_t)
            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
            # 在验证集上测试正确率。
            if i%500 == 0 or i+1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(bottlenecks_t, ground_truth_t)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                #print(sess.run(tf.argmax(final_tensor, 1), feed_dict={bottleneck_input: bottlenecks_v}))
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%'
                      % (i, BATCH, validation_accuracy*100))

        # 在最后的测试数据上测试正确率
        test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                                             ground_truth_input: test_ground_truth})

        print('Final test accuracy (original) = %.1f%%' % (test_accuracy * 100))

    #bottlenecks_t1_array = np.array(bottlenecks[0:400000, :])
    #(u, sigma, vt) = pca(bottlenecks_t1_array, 1500)
    #v = vt.transpose()
    #pickle.dump(v, open('v.p:kl', 'wb'))

    v = pickle.load(open('v.pkl', 'rb'))

    #svd
    ground_truth_t2 = ground_truth_t[400000:, :]
    del ground_truth_t
    bottlenecks_t2 = bottlenecks_t[400000:, :]
    del bottlenecks_t

    svd_test(1024, n_classes, bottlenecks_t2, ground_truth_t2, test_bottlenecks, test_ground_truth, v)
    svd_test(512,  n_classes, bottlenecks_t2, ground_truth_t2, test_bottlenecks, test_ground_truth, v)
    svd_test(256, n_classes, bottlenecks_t2, ground_truth_t2, test_bottlenecks, test_ground_truth, v)
    svd_test(128,n_classes, bottlenecks_t2, ground_truth_t2, test_bottlenecks, test_ground_truth, v)
    svd_test(64, n_classes, bottlenecks_t2, ground_truth_t2, test_bottlenecks, test_ground_truth, v)
    svd_test(32, n_classes, bottlenecks_t2, ground_truth_t2, test_bottlenecks, test_ground_truth, v)

    '''svd_test(1024, n_classes, bottlenecks_t1, ground_truth_t1,
             test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2)
    svd_test(512,  n_classes, bottlenecks_t1, ground_truth_t1,
             test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2)
    svd_test(256, n_classes, bottlenecks_t1, ground_truth_t1,
             test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2)
    svd_test(128, n_classes, bottlenecks_t1, ground_truth_t1,
             test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2)
    svd_test(64, n_classes, bottlenecks_t1, ground_truth_t1,
             test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2)
    svd_test(32, n_classes, bottlenecks_t1, ground_truth_t1,
             test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2)'''


if __name__ == '__main__':
    main()
