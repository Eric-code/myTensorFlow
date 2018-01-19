# 头文件
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 数据占位
x = tf.placeholder(tf.float32, [None, 256*46])
y_actual = tf.placeholder(tf.float32, shape=[None, 6])


# 定义函数，用于初始化权值 W，初始化偏置b，定义卷积层，定义池化层
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_6(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 256, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 1, 1], padding='SAME')


def max_pool_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')


# 构建网络
# 获取数据+BN层1
x_image = tf.reshape(x, [-1, 46, 256, 1])
epsilon = 0.001
mean, var = tf.nn.moments(x_image, axes=[0],)
scale = tf.Variable(tf.ones([1]))
shift = tf.Variable(tf.zeros([1]))
x_image_normal = tf.nn.batch_normalization(x_image, mean, var, shift, scale, epsilon)
# 卷积层1+池化层1
W_conv1 = weight_variable([4, 256, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d_6(x_image_normal, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)
# BN层2
# 想要 normalize 的维度, [0] 代表 batch 维度
# 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
mean_pool1, var_pool1 = tf.nn.moments(h_pool1, axes=[0, 1, 2],)
scale_pool1 = tf.Variable(tf.ones([32]))
shift_pool1 = tf.Variable(tf.zeros([32]))
h_bn_pool1 = tf.nn.batch_normalization(h_pool1, mean_pool1, var_pool1, shift_pool1, scale_pool1, epsilon)
# 卷积层2+池化层2
W_conv2 = weight_variable([4, 4, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_bn_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2(h_conv2)
# BN层3
mean_pool2, var_pool2 = tf.nn.moments(h_pool2, axes=[0, 1, 2],)
scale_pool2 = tf.Variable(tf.ones([64]))
shift_pool2 = tf.Variable(tf.zeros([64]))
h_bn_pool2 = tf.nn.batch_normalization(h_pool2, mean_pool2, var_pool2, shift_pool2, scale_pool2, epsilon)
# 全连接层1
W_fc1 = weight_variable([23 * 1 * 64, 512])
b_fc1 = bias_variable([512])
h_pool2_flat = tf.reshape(h_bn_pool2, [-1, 23 * 1 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# BN层4
mean_fc1, var_fc1 = tf.nn.moments(h_fc1, axes=[0],)
scale_fc1 = tf.Variable(tf.ones([512]))
shift_fc1 = tf.Variable(tf.zeros([512]))
h_bn_fc1 = tf.nn.batch_normalization(h_fc1, mean_fc1, var_fc1, shift_fc1, scale_fc1, epsilon)
# 全连接层2
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_bn_fc1, keep_prob)
W_fc2 = weight_variable([512, 6])
b_fc2 = bias_variable([6])
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 模型优化
# cross_entropy = tf.reduce_mean((y_actual - y_predict) * (y_actual - y_predict))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(tf.clip_by_value(y_predict, 1e-15, 1.0))))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 读入真实数据
input_count = 5508
x_s = np.loadtxt('Comnet-14_flow_onehot_withoutIP_google_train.txt')
y_s = np.loadtxt('Comnet-14_raw_without_google_train_label.txt')
train_images = np.array([[0] * 256 for i in range(input_count*46)])
train_labels = np.array([[0] * 6 for i in range(input_count)])
for index in range(input_count*46):
    for j in range(256):
        train_images[index][:] = x_s[index, :]
train_images = tf.reshape(train_images, [-1, 256*46])
for index in range(input_count):
    for k in range(6):
        train_labels[index][:] = y_s[index, :]
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=input_count).batch(1377).repeat()
iterator = dataset.make_initializable_iterator()
one_element = iterator.get_next()

input_count1 = 3000
x_t = np.loadtxt('Comnet-14_raw_withoutIP_google_test_threehot.txt')
y_t = np.loadtxt('Comnet-14_raw_withoutIP+Port_test_three_label.txt')
test_images = np.array([[0] * 256 for i in range(input_count1*46)])
test_labels = np.array([[0] * 6 for i in range(input_count1)])
for index in range(input_count1*46):
    for j in range(256):
        test_images[index][:] = x_t[index, :]
test_images = tf.reshape(test_images, [-1, 256*46])
for index in range(input_count1):
    for k in range(6):
        test_labels[index][:] = y_t[index, :]
dataset1 = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(buffer_size=input_count1).batch(400).repeat()
iterator1 = dataset1.make_initializable_iterator()
one_element1 = iterator1.get_next()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
train_images = sess.run(train_images)
test_images = sess.run(test_images)
sess.run(iterator.initializer)
sess.run(iterator1.initializer)

for i in range(30000):
    batch_xs, batch_ys = sess.run(one_element)
    # batch_xs1, batch_ys1 = sess.run(one_element1)
    if i % 5 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_actual: batch_ys, keep_prob: 0.5})
        test_accuracy = accuracy.eval(feed_dict={x: test_images, y_actual: test_labels, keep_prob: 0.5})
        # train_accuracy = accuracy.eval(feed_dict={x: train_images, y_actual: train_labels, keep_prob: 0.5})
        # test_accuracy = accuracy.eval(feed_dict={x: test_images, y_actual: test_labels, keep_prob: 0.5})
        print("step %d, train accuracy %g" % (i, train_accuracy), ", test accuracy %g" % test_accuracy)
    train_step.run(feed_dict={x: batch_xs, y_actual: batch_ys, keep_prob: 0.5})
    # train_step.run(feed_dict={x: train_images, y_actual: train_labels, keep_prob: 0.5})
