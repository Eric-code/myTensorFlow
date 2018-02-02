# 加载包
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# 数据集名称，数据集要放在工作目录下
TRAFFIC_TRAINING = "nonTor_120s_Layer2_tidy_train.csv"
TRAFFIC_TEST = "nonTor_120s_Layer2_tidy_test.csv"

# 数据集读取，训练集和测试集
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TRAFFIC_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TRAFFIC_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)


def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y


def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x, y


# 特征
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=77)]

# 构建DNN网络，3层，每层分别为100,200,100个节点
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[100, 200, 100],
                                            n_classes=8,
                                            dropout=0.5,
                                            optimizer=tf.train.GradientDescentOptimizer(
                                                learning_rate=0.1,
                                                # l1_regularization_strength=0.001
                                            ))

for i in range(1000):
# 拟合模型，迭代x步
    classifier.fit(input_fn=get_train_inputs, steps=200)
# 计算精度
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
    # print('Accuracy: {0:f}'.format(accuracy_score))
    print('Step: %d' % (i * 200), 'Accuracy: {0:f}'.format(accuracy_score))

# # 预测新样本的类别
# new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
# y = list(classifier.predict(new_samples, as_iterable=True))
# print('Predictions: {}'.format(str(y)))
