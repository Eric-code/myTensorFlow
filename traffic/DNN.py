#加载包
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import tensorflow as tf
import numpy as np

# 数据集名称，数据集要放在工作目录下
TRAFFIC_TRAINING = "Comnet-14_all_withoutIP+Port_train.csv"
TRAFFIC_TEST = "Comnet-14_all_withoutIP+Port_test.csv"

# 数据集读取，训练集和测试集
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TRAFFIC_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TRAFFIC_TEST,
    target_dtype=np.int,
    features_dtype=np.float)

# 特征
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=42)]

# 构建DNN网络，3层，每层分别为100,200,100个节点
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[50, 100, 50],
                                            n_classes=7,
                                            model_dir="/tmp/traffic_model_flow_withoutIP+Port")

for i in range(100):
# 拟合模型，迭代x步
    classifier.fit(x=training_set.data,
                   y=training_set.target,
                   steps=50)
    # 计算精度
    accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
    print('Step: %d' % i, 'Accuracy: {0:f}'.format(accuracy_score))

# # 预测新样本的类别
# new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
# y = list(classifier.predict(new_samples, as_iterable=True))
# print('Predictions: {}'.format(str(y)))

