#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
import time
from fbpca import pca
from tensorflow.python.platform import gfile


# Inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# Inception-v3模型中代表瓶颈层结果的张量名称。
# 在谷歌提出的Inception-v3模型中，这个张量名称就是'pool_3/_reshape:0'。
# 在训练模型时，可以通过tensor.name来获取张量的名称。
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像输入张量所对应的名称。
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# 下载的谷歌训练好的Inception-v3模型文件目录
MODEL_DIR = 'model/'

# 下载的谷歌训练好的Inception-v3模型文件名
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 因为一个训练数据会被使用多次，所以可以将原始图像通过Inception-v3模型计算得到的特征向量保存在文件中，免去重复的计算。
# 下面的变量定义了这些文件的存放地址。
CACHE_DIR = 'tmp/bottleneck/'

# 图片数据文件夹。
# 在这个文件夹中每一个子文件夹代表一个需要区分的类别，每个子文件夹中存放了对应类别的图片。
INPUT_DATA = '../../Imagenet'

# 验证的数据百分比
VALIDATION_PERCENTAGE = 10
# 测试的数据百分比
TEST_PERCENTAGE = 10

# 定义神经网络的设置
LEARNING_RATE = 0.01
STEPS = 5000
BATCH = 100
TRAINING_SIZE = 50000
VALIDATION_SIZE = 50000


class NodeLookup(object):
  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
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


# 这个函数从数据文件夹中读取所有的图片列表并按训练、验证、测试数据分开。
# testing_percentage和validation_percentage参数指定了测试数据集和验证数据集的大小。
def create_image_lists(testing_percentage, validation_percentage):
    # 得到的所有图片都存在result这个字典(dictionary)里。
    # 这个字典的key为类别的名称，value也是一个字典，字典里存储了所有的图片名称。
    result = {}
    # 获取当前目录下所有的子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # 得到的第一个目录是当前目录，不需要考虑
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下所有的有效图片文件。
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        # 通过目录名获取类别的名称。
        label_name = dir_name.lower()
        # 初始化当前类别的训练数据集、测试数据集和验证数据集
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # 随机将数据分到训练数据集、测试数据集和验证数据集。
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据放入结果字典。
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
            }
    # 返回整理好的所有数据
    return result


# 这个函数通过类别名称、所属数据集和图片编号获取一张图片的地址。
# image_lists参数给出了所有图片信息。
# image_dir参数给出了根目录。存放图片数据的根目录和存放图片特征向量的根目录地址不同。
# label_name参数给定了类别的名称。
# index参数给定了需要获取的图片的编号。
# category参数指定了需要获取的图片是在训练数据集、测试数据集还是验证数据集。
def get_image_path(image_lists, image_dir, label_name, index, category):
    # 获取给定类别中所有图片的信息。
    label_lists = image_lists[label_name]
    # 根据所属数据集的名称获取集合中的全部图片信息。
    category_list = label_lists[category]
    if len(category_list) == 0:
        print(len(category_list))
    mod_index = index % len(category_list)
    # 获取图片的文件名。
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    # 最终的地址为数据根目录的地址 + 类别的文件夹 + 图片的名称
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


# 这个函数通过类别名称、所属数据集和图片编号获取经过Inception-v3模型处理之后的特征向量文件地址。
def get_bottlenect_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt';


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor, category):
    # 这个过程实际上就是将当前图片作为输入计算瓶颈张量的值。这个瓶颈张量的值就是这张图片的新的特征向量。
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量（一维数组）
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# 这个函数使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量。
def run_bottleneck_on_images(sess, image_datas, image_data_tensor, bottleneck_tensor):
    bottleneck_values = []
    # 这个过程实际上就是将当前图片作为输入计算瓶颈张量的值。这个瓶颈张量的值就是这张图片的新的特征向量。
    for image_data in image_datas:
        bottleneck_value = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
        # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量（一维数组）
        bottleneck_values.append(np.squeeze(bottleneck_value))
    return bottleneck_values


# 这个函数获取一张图片经过Inception-v3模型处理之后的特征向量。
# 这个函数会先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件。
def get_or_create_bottleneck(sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件的路径。
    label_lists = image_lists[label_name]
    #sub_dir = label_lists['dir']
    #sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    #if not os.path.exists(sub_dir_path):
    #    os.makedirs(sub_dir_path)
    bottleneck_path = get_bottlenect_path(image_lists, label_name, image_index, category)
    # 获取原始的图片路径
    image_path = get_image_path(image_lists, INPUT_DATA, label_name, image_index, category)
    # 获取图片内容。
    image_data = gfile.FastGFile(image_path, 'rb').read()
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)
    node_lookup = NodeLookup()
    node_id = int(np.squeeze(np.where(predictions == np.max(predictions))))
    stringid = node_lookup.id_to_string(node_id)
    if stringid == label_name:
        correct_classify = 1
    else:
        correct_classify = 0
    # 如果这个特征向量文件不存在，则通过Inception-v3模型来计算特征向量，并将计算的结果存入文件。
    if not os.path.exists(bottleneck_path):
        # 由于输入的图片大小不一致，此处得到的image_data大小也不一致（已验证），但却都能通过加载的inception-v3模型生成一个2048的特征向量。具体原理不详。
        # 通过Inception-v3模型计算特征向量
        time_0 = time.clock()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor, category)
        time_1 = time.clock()
        cal_time = time_1 - time_0
        # 将计算得到的特征向量存入文件
        #bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        #with open(bottleneck_path, 'w') as bottleneck_file:
        #    bottleneck_file.write(bottleneck_string)

    else:
        # 直接从文件中获取图片相应的特征向量。
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    # 返回得到的特征向量
    return bottleneck_values, correct_classify, cal_time


def get_fixed_size_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = np.zeros((TRAINING_SIZE, 2048), dtype=float)
    ground_truths = np.zeros((TRAINING_SIZE, 1000), dtype=float)
    image_datas = []
    i = 0
    time1 = 0
    while i < how_many:
        # 随机一个类别和图片的编号加入当前的训练数据。
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        label_lists = image_lists[label_name]
        #sub_dir = label_lists['dir']
        #sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
        #if not os.path.exists(sub_dir_path):
        #    os.makedirs(sub_dir_path)
        bottleneck_path = get_bottlenect_path(image_lists, label_name, image_index, category)
        # 如果这个特征向量文件不存在，则通过Inception-v3模型来计算特征向量，并将计算的结果存入文件。
        if not os.path.exists(bottleneck_path):
            image_path = get_image_path(image_lists, INPUT_DATA, label_name, image_index, category)
            image_data = gfile.FastGFile(image_path, 'rb').read()
            image_datas.append(image_data)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            ground_truths[i] = ground_truth
            i = i + 1
            if i % BATCH == 0:
                time_0 = time.clock()
                bottleneck_values = run_bottleneck_on_images(sess, image_datas, jpeg_data_tensor, bottleneck_tensor)
                image_datas = []
                time_1 = time.clock()
                time1 = time1 + time_1 - time_0
                j = BATCH
                for bottleneck_value in bottleneck_values:
                    #bottleneck_string = ','.join(str(x) for x in bottleneck_value)
                    #with open(bottleneck_path, 'w') as bottleneck_file:
                    #    bottleneck_file.write(bottleneck_string)
                    bottlenecks[i-j] = bottleneck_value
                    j = j - 1

        if i % 1000 == 0:
            print('load %d images. cal BN time: %.3fs. ' % (i, time1))
            time1 = 0

    return bottlenecks, ground_truths


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
    bottlenecks = []
    ground_truths = []
    correct_classify_count = []
    i = 0
    cal_bn_time = 0
    label_name_list = list(image_lists.keys())
    # 枚举所有的类别和每个类别中的测试图片。
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            # 通过Inception-v3模型计算图片对应的特征向量，并将其加入最终数据的列表。
            bottleneck, correct_classify, cal_time = get_or_create_bottleneck(sess, image_lists, label_name, index, category,
                                                  jpeg_data_tensor, bottleneck_tensor)
            cal_bn_time = cal_bn_time + cal_time
            correct_classify_count.append(correct_classify)
            bottlenecks.append(bottleneck)
            ground_truth = np.zeros(n_classes, dtype = np.float32)
            ground_truth[label_index] = 1.0
            ground_truths.append(ground_truth)
            i = i + 1
            if i % 1000 == 0:
                print('(test) load %d images. cal BN time: %.3fs. ' % (i, cal_bn_time))
                cal_bn_time = 0
    return bottlenecks, ground_truths, correct_classify_count


def svd_test(singular_value_number, n_classes, bottlenecks_t1, ground_truth_t1,
            test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2, bottlenecks_v, ground_truth_v):
    bottleneck_input = tf.placeholder(tf.float32, [None, singular_value_number], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
    # 定义一层全连接层来解决新的图片分类问题。
    # 因为训练好的Inception-v3模型已经将原始的图片抽象为了更加容易分类的特征向量了，所以不需要再训练那么复杂的神经网络来完成这个新的分类任务。
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([singular_value_number, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)
    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    bottlenecks_t1_array = np.array(bottlenecks_t1)
    (u, sigma, vt) = pca(bottlenecks_t1_array, singular_value_number)
    v = vt.transpose()

    bottlenecks_after_svd_t2 = np.dot(bottlenecks_t2, v)
    bottlenecks_after_svd_v = np.dot(bottlenecks_v, v)
    test_bottlenecks_after_svd = np.dot(test_bottlenecks, v)

    with tf.Session() as sess2:
        tf.global_variables_initializer().run()
        # 训练过程
        for i in range(STEPS):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(bottlenecks_after_svd_t2, ground_truth_t2)
            sess2.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
            # 在验证集上测试正确率。
            if i % 200 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(bottlenecks_after_svd_v, ground_truth_v)
                validation_accuracy = sess2.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%'
                    % (i, BATCH, validation_accuracy * 100))

    # 在最后的测试数据上测试正确率
        test_accuracy = sess2.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks_after_svd,
                                                         ground_truth_input: test_ground_truth})
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
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
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
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        test_bottlenecks, test_ground_truth, correct_classify_count = get_test_bottlenecks(sess, image_lists,
                                                        n_classes, jpeg_data_tensor, bottleneck_tensor)
        Inception_accuracy = float(np.sum(correct_classify_count) / len(correct_classify_count))
        print('Inception accuracy (original) = %.3f%%' % (Inception_accuracy * 100))

        bottlenecks_t1, ground_truth_t1 = get_fixed_size_bottlenecks(sess, n_classes, image_lists,
                                             TRAINING_SIZE, 'training', jpeg_data_tensor,bottleneck_tensor)
        bottlenecks_t2, ground_truth_t2 = get_fixed_size_bottlenecks(sess, n_classes, image_lists,
                                             TRAINING_SIZE, 'training', jpeg_data_tensor, bottleneck_tensor)
        bottlenecks_v, ground_truth_v = get_fixed_size_bottlenecks(sess, n_classes, image_lists,
                                             VALIDATION_SIZE, 'validation', jpeg_data_tensor,bottleneck_tensor)

        bottlenecks_t = np.vstack((bottlenecks_t1, bottlenecks_t2))
        ground_truth_t = np.vstack((ground_truth_t1, ground_truth_t2))

        # 训练过程
        for i in range(STEPS):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks( bottlenecks_t, ground_truth_t)
            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
            # 在验证集上测试正确率。
            if i%100 == 0 or i+1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(bottlenecks_v, ground_truth_v)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%'
                      % (i, BATCH, validation_accuracy*100))

        # 在最后的测试数据上测试正确率
        test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                                             ground_truth_input: test_ground_truth})

        print('Final test accuracy (original) = %.1f%%' % (test_accuracy * 100))

        #np.savetxt('matrix_ILSVRC2012_img_train_t3.csv', A, delimiter=',')
        #(u, sigma, vt) = pca(A, 200, True)
        #np.savetxt('value_ILSVRC2012_img_train_t3.csv', sigma, delimiter=',')


    #svd
    svd_test(1024, n_classes, bottlenecks_t1, ground_truth_t1,
             test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2, bottlenecks_v, ground_truth_v)
    svd_test(512,  n_classes, bottlenecks_t1, ground_truth_t1,
             test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2, bottlenecks_v, ground_truth_v)
    svd_test(256, n_classes, bottlenecks_t1, ground_truth_t1,
             test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2, bottlenecks_v, ground_truth_v)
    svd_test(128, n_classes, bottlenecks_t1, ground_truth_t1,
             test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2, bottlenecks_v, ground_truth_v)
    svd_test(64, n_classes, bottlenecks_t1, ground_truth_t1,
             test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2, bottlenecks_v, ground_truth_v)
    svd_test(32, n_classes, bottlenecks_t1, ground_truth_t1,
             test_bottlenecks, test_ground_truth, bottlenecks_t2, ground_truth_t2, bottlenecks_v, ground_truth_v)


if __name__ == '__main__':
    main()
