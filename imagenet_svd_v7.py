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
STEPS = 10000
BATCH = 200

# how many training and validation images we want to use
TRAINING_SIZE = 1282167

bottlenecks = np.zeros((TRAINING_SIZE, 2048), dtype=float)
ground_truths = np.zeros((TRAINING_SIZE, 1000), dtype=float)

class NodeLookup(object):
  def __init__(self, label_lookup_path=None, uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = 'model2/imagenet_2012_challenge_label_map_proto.pbtxt'
    if not uid_lookup_path:
      uid_lookup_path = 'model2/imagenet_synset_to_human_label_map.txt'
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    return node_id_to_uid

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def dict2list(dic:dict):
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst

# 这个函数从数据文件夹中读取所有的图片列表并按训练、验证、测试数据分开。
# testing_percentage和validation_percentage参数指定了测试数据集和验证数据集的大小。
def create_bottleneck_lists():
    # 得到的所有图片都存在result这个字典(dictionary)里。
    # 这个字典的key为类别的名称，value也是一个字典，字典里存储了所有的图片名称。
    result = {}
    # 获取当前目录下所有的子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # 得到的第一个目录是当前目录，不需要考虑
    is_root_dir = True
    for sub_dir in sorted(sub_dirs):
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下所有的有效txt文件。
        file_list = []
        base_names = []
        dir_name = os.path.basename(sub_dir)
        file_glob = os.path.join(INPUT_DATA, dir_name, '*.txt')
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        # 通过目录名获取类别的名称。
        label_name = dir_name.lower()
        # 初始化当前类别的训练数据集、测试数据集和验证数据集
        i = 0
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            base_names.append(base_name)

        # 将当前类别的数据放入结果字典。
        result[label_name] = {
            #'dir': dir_name,
            'base_name': base_names
            }
    sorted(dict2list(result), key=lambda x:x[0], reverse=False)
    # 返回整理好的所有数据
    return result


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    # 这个过程实际上就是将当前图片作为输入计算瓶颈张量的值。这个瓶颈张量的值就是这张图片的新的特征向量。
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量（一维数组）
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def create_test_bottleneck(sess, image_data, jpeg_data_tensor, bottleneck_tensor):
    # 由于输入的图片大小不一致，此处得到的image_data大小也不一致（已验证），但却都能通过加载的inception-v3模型生成一个2048的特征向量。具体原理不详。
    # 通过Inception-v3模型计算特征向量
    bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    # 返回得到的特征向量
    return bottleneck_values


def get_bottlenecks(sess, n_classes, bottleneck_lists):
    global bottlenecks, ground_truths
    label_name_list = list(bottleneck_lists.keys())
    i = 0
    # 枚举所有的类别和每个类别中的测试图片。
    for label_index, label_name in enumerate(label_name_list):
        category = 'base_name'
        for index, unused_base_name in enumerate(bottleneck_lists[label_name][category]):
            #text_file = open(os.path.join(INPUT_DATA, label_name, unused_base_name), "r")
            #lines = text_file.read().split(',')
            #bottlenecks[i] = lines

            bottleneck_path = os.path.join(INPUT_DATA, label_name, unused_base_name)
            with open(bottleneck_path, 'r') as bottleneck_file:
                bottleneck_string = bottleneck_file.read()
            bottleneck = [float(x) for x in bottleneck_string.split(',')]
            bottlenecks[i] = bottleneck

            label_index = list(bottleneck_lists.keys()).index(label_name)
            ground_truths[i][label_index] = 1
            i = i + 1
            if i % 1000 == 0:
                print("load ",i," training bottlenecks.")
            if i == TRAINING_SIZE:
                break
    ordering = np.random.permutation(TRAINING_SIZE)
    bottlenecks = bottlenecks[ordering, :]
    ground_truths = ground_truths[ordering, :]

    svd_bn_0 = bottlenecks[0:200000, :]
    svd_bn_1 = bottlenecks[200000:400000, :]

    svd_bn = [] #np.stack([svd_bn_0, svd_bn_1], axis=0)

    pickle.dump(svd_bn_0, open('svd_bn_0.pkl', 'wb'))
    pickle.dump(svd_bn_1, open('svd_bn_1.pkl', 'wb'))

    cls_bn_0 = bottlenecks[400000:600000, :]
    cls_bn_1 = bottlenecks[600000:800000, :]
    cls_bn_2 = bottlenecks[800000:1000000, :]
    cls_bn_3 = bottlenecks[1000000:1200000, :]
    cls_bn_4 = bottlenecks[1200000:, :]

    cls_bn = [] #np.stack([cls_bn_0, cls_bn_1, cls_bn_2, cls_bn_3, cls_bn_4], axis=0)

    pickle.dump(cls_bn_0, open('cls_bn_0.pkl', 'wb'))
    pickle.dump(cls_bn_1, open('cls_bn_1.pkl', 'wb'))
    pickle.dump(cls_bn_2, open('cls_bn_2.pkl', 'wb'))
    pickle.dump(cls_bn_3, open('cls_bn_3.pkl', 'wb'))
    pickle.dump(cls_bn_4, open('cls_bn_4.pkl', 'wb'))

    svd_gt_0 = ground_truths[0:200000, :]
    svd_gt_1 = ground_truths[200000:400000, :]

    svd_gt = [] #np.stack([svd_gt_0, svd_gt_1], axis=0)

    pickle.dump(svd_gt_0, open('svd_gt_0.pkl', 'wb'))
    pickle.dump(svd_gt_1, open('svd_gt_1.pkl', 'wb'))

    cls_gt_0 = ground_truths[400000:600000, :]
    cls_gt_1 = ground_truths[600000:800000, :]
    cls_gt_2 = ground_truths[800000:1000000, :]
    cls_gt_3 = ground_truths[1000000:1200000, :]
    cls_gt_4 = ground_truths[1200000:, :]

    cls_gt = [] #np.stack([cls_gt_0, cls_gt_1, cls_gt_2, cls_gt_3, cls_gt_4], axis=0)

    pickle.dump(cls_gt_0, open('cls_gt_0.pkl', 'wb'))
    pickle.dump(cls_gt_1, open('cls_gt_1.pkl', 'wb'))
    pickle.dump(cls_gt_2, open('cls_gt_2.pkl', 'wb'))
    pickle.dump(cls_gt_3, open('cls_gt_3.pkl', 'wb'))
    pickle.dump(cls_gt_4, open('cls_gt_4.pkl', 'wb'))

    return svd_bn, svd_gt, cls_bn, cls_gt


def get_random_cached_bottlenecks(MA, MB):
    bottlenecks = []
    ground_truths = []
    for i in range(BATCH):
        x = random.randint(1, len(MA)-1)
        bottlenecks.append(MA[x])
        ground_truths.append(MB[x])
    return bottlenecks, ground_truths


# 这个函数获取全部的测试数据。在最终测试的时候需要在所有的测试数据上计算正确率。
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    inception_node_id = []
    image_path_list = []
    i = 0
    correct_classify_count = 0
    node_lookup = NodeLookup()
    bottlenecks = np.zeros((TEST_SIZE, 2048), dtype=float)
    ground_truths = np.zeros((TEST_SIZE, 1000), dtype=float)
    # 枚举所有的类别和每个类别中的测试图片。
    with open(TEST_GROUND_TRUTH_FILE) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    for file in os.listdir(TEST_DIR):
        if file.endswith(".JPEG"):
            image_path_list.append(os.path.join(TEST_DIR, file))
    image_path_list.sort()
    if not os.path.exists("tmp/val"):
        os.makedirs("tmp/val")
    for image_path in image_path_list:
        image_data = gfile.FastGFile(image_path, 'rb').read()
        #softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        #predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        #predictions = np.squeeze(predictions)
        #node_id = int(np.squeeze(np.where(predictions == np.max(predictions))))
        #inception_node_id.append(node_id)
        # 通过Inception-v3模型计算图片对应的特征向量，并将其加入最终数据的列表。
        bottleneck_name = image_path.split(".")
        new_name = bottleneck_name[0] + ".txt"
        bottleneck_path = os.path.join("tmp", new_name)

        #if not os.path.exists(bottleneck_path):
        '''bottleneck = create_test_bottleneck(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
            bottleneck_string = ','.join(str(x) for x in bottleneck)
            with open(bottleneck_path, 'w') as bottleneck_file:
                bottleneck_file.write(bottleneck_string)
        else:'''
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck = [float(x) for x in bottleneck_string.split(',')]
        bottlenecks[i] = bottleneck
        i = i + 1
        if i % 1000 == 0:
            print("load ",i," test bottlenecks.")

    for j in range(TEST_SIZE):
        string_id = node_lookup.id_to_string(int(content[j]))
        label_index = list(image_lists.keys()).index(string_id)
        ground_truths[j][label_index] = 1
        #if int(content[j]) == inception_node_id[j]:
        #    correct_classify_count = correct_classify_count + 1
    inception_accuracy = 0 #correct_classify_count / TEST_SIZE

    pickle.dump(bottlenecks, open('test_bn.pkl', 'wb'))
    pickle.dump(ground_truths, open('test_gt.pkl', 'wb'))

    return bottlenecks, ground_truths, inception_accuracy


def svd_test(singular_value_number, n_classes, bottlenecks_t1, ground_truth_t1,
            test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2):
    bottleneck_input = tf.placeholder(tf.float32, [None, singular_value_number], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
    # 定义一层全连接层来解决新的图片分类问题。
    # 因为训练好的Inception-v3模型已经将原始的图片抽象为了更加容易分类的特征向量了，所以不需要再训练那么复杂的神经网络来完成这个新的分类任务。
    with tf.name_scope('second_last_training_ops'):
        weights_1 = tf.Variable(tf.truncated_normal([singular_value_number, singular_value_number], stddev=0.001))
        biases_1 = tf.Variable(tf.zeros([singular_value_number]))
        logits_1 = tf.matmul(bottleneck_input, weights_1) + biases_1
        final_tensor_1 = tf.nn.relu(logits_1)
    with tf.name_scope('final_training_ops'):
        weights_2 = tf.Variable(tf.truncated_normal([singular_value_number, n_classes], stddev=0.001))
        biases_2 = tf.Variable(tf.zeros([n_classes]))
        logits_2 = tf.matmul(final_tensor_1, weights_2) + biases_2
        final_tensor_2 = tf.nn.softmax(logits_2)
    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_2, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor_2, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    bottlenecks_t1_array = np.array(bottlenecks_t1)
    (u, sigma, vt) = pca(bottlenecks_t1_array, singular_value_number)
    v = vt.transpose()

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


def create_graph():
    # TensorFlow模型持久化的问题在第5章中有详细的介绍。
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # 加载读取的Inception-v3模型，并返回数据输入所对应的张量以及计算瓶颈层结果所对应的张量。
        tf.import_graph_def(graph_def, name='')
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                              JPEG_DATA_TENSOR_NAME])
        return bottleneck_tensor, jpeg_data_tensor

def main():
    # 读取所有图片。
    #sys.stdout = open("test.txt", "w")
    bottleneck_lists = create_bottleneck_lists()
    n_classes = len(bottleneck_lists.keys())
    # 读取已经训练好的Inception-v3模型。
    # 谷歌训练好的模型保存在了GraphDef Protocol Buffer中，里面保存了每一个节点取值的计算方法以及变量的取值。
    bottleneck_tensor, jpeg_data_tensor = create_graph()
    # 定义新的神经网络输入，这个输入就是新的图片经过Inception-v3模型前向传播到达瓶颈层时的结点取值。
    # 可以将这个过程类似的理解为一种特征提取。
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    # 定义新的标准答案输入
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
    # 定义一层全连接层来解决新的图片分类问题。
    # 因为训练好的Inception-v3模型已经将原始的图片抽象为了更加容易分类的特征向量了，所以不需要再训练那么复杂的神经网络来完成这个新的分类任务。
    with tf.name_scope('second_last_training_ops'):
        weights_1 = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, BOTTLENECK_TENSOR_SIZE], stddev=0.001))
        biases_1 = tf.Variable(tf.zeros([BOTTLENECK_TENSOR_SIZE]))
        logits_1 = tf.matmul(bottleneck_input, weights_1) + biases_1
        final_tensor_1 = tf.nn.relu(logits_1)
    with tf.name_scope('final_training_ops'):
        weights_2 = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases_2 = tf.Variable(tf.zeros([n_classes]))
        logits_2 = tf.matmul(final_tensor_1, weights_2) + biases_2
        final_tensor_2 = tf.nn.softmax(logits_2)

    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_2, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor_2, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        test_bottlenecks, test_ground_truth, inception_accuracy = get_test_bottlenecks(sess, bottleneck_lists, n_classes,
                                                                                jpeg_data_tensor, bottleneck_tensor)

        print('Inception accuracy (original) = %.3f%%' % (inception_accuracy * 100))

        bottlenecks_t1, ground_truth_t1, bottlenecks_t2, ground_truth_t2 = get_bottlenecks(sess, n_classes, bottleneck_lists)

        bottlenecks_t = np.vstack((bottlenecks_t1, bottlenecks_t2))
        ground_truth_t = np.vstack((ground_truth_t1, ground_truth_t2))

        # 训练过程
        for i in range(STEPS):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks( bottlenecks_t, ground_truth_t)
            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
            # 在验证集上测试正确率。
            '''if i%500 == 0 or i+1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(bottlenecks_t, ground_truth_t)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                #print(sess.run(tf.argmax(final_tensor, 1), feed_dict={bottleneck_input: bottlenecks_v}))
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%'
                      % (i, BATCH, validation_accuracy*100))'''

        # 在最后的测试数据上测试正确率
        test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                                             ground_truth_input: test_ground_truth})

        print('Final test accuracy (original) = %.1f%%' % (test_accuracy * 100))


    #svd
    svd_test(1024, n_classes, bottlenecks_t1, ground_truth_t1,
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
             test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2)


if __name__ == '__main__':
    main()
