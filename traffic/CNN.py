# 头文件
import tensorflow as tf
import numpy as np


def read_data(filename):
    data_list = []
    label_list = []
    with open(filename, 'r') as f:
        # line = f.readline()
        # line = list(line.split(','))
        # print(line)
        for line in f.readlines():
            line = list(line.split(','))
            line.pop(-1)
            line = [float(x) for x in line]
            data_list.append(line[:-8])
            label_list.append(line[-8:])
    return data_list, label_list


tf_x = tf.placeholder(tf.float32, [None, 1*19])
image = tf.reshape(tf_x, [-1, 1, 19, 1])              # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 8])            # input y
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing

# 构建网络
# 获取数据+BN层1
x_image = tf.reshape(tf_x, [-1, 1, 19, 1])
epsilon = 0.001
mean, var = tf.nn.moments(x_image, axes=[0],)
scale = tf.Variable(tf.ones([1]))
shift = tf.Variable(tf.zeros([1]))
x_image_normal = tf.nn.batch_normalization(x_image, mean, var, shift, scale, epsilon)


# CNN
conv1 = tf.layers.conv2d(   # shape (1, 19, 1)
    inputs=x_image_normal,
    filters=32,
    kernel_size=4,
    strides=1,
    padding='same',
    activation=tf.nn.relu,
)           # -> (28, 28, 16)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=1,
    strides=1,
)           # -> (1, 19, 32)
drop1 = tf.layers.dropout(pool1, rate=0.5, training=tf_is_training)   # drop out 50% of inputs
conv2 = tf.layers.conv2d(drop1, 64, 4, 1, 'same', activation=tf.nn.relu)    # -> (1, 19, 64)
pool2 = tf.layers.max_pooling2d(conv2, 1, 1)    # -> (1, 19, 64)
drop2 = tf.layers.dropout(pool2, rate=0.5, training=tf_is_training)   # drop out 50% of inputs
flat = tf.reshape(drop2, [-1, 1*19*64])          # -> (7*7*32, )
flat2 = tf.layers.dense(flat, 1024)
output = tf.layers.dense(flat2, 8)              # output layer
# output = tf.nn.softmax(output)
print(tf_y)
print(output)
# output = tf.to_int32(output)
tf_y = tf.to_float(tf_y)
# loss = tf.reduce_mean((tf_y - output) * (tf_y - output))
loss = tf.reduce_mean(-tf.reduce_sum(tf_y * tf.log(tf.clip_by_value(output, 1e-15, 1.0))))
# loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

# 读入真实数据
train_x, train_y = read_data("flow_15s_19_train_label.csv")
test_x, test_y = read_data("flow_15s_19_test_label.csv")


# input_count = 755200
# x_s = np.loadtxt('Comnet-14_feature5_train.txt')
# y_s = np.loadtxt('Comnet-14_feature6_train_label.txt')
# train_images = np.array([[0] * 5 for i in range(input_count)])
# train_labels = np.array([[0] * 7 for i in range(input_count)])
# for index in range(input_count):
#     for j in range(5):
#         train_images[index][:] = x_s[index, :]
#     for k in range(7):
#         train_labels[index][:] = y_s[index, :]

# input_count1 = 83600
# x_t = np.loadtxt('Comnet-14_feature6_test.txt')
# y_t = np.loadtxt('Comnet-14_feature6_test_label.txt')
# test_images = np.array([[0] * 6 for i in range(input_count1)])
# test_labels = np.array([[0] * 7 for i in range(input_count1)])
# for index in range(input_count1):
#     for j in range(6):
#         test_images[index][:] = x_t[index, :]
#     for k in range(7):
#         test_labels[index][:] = y_t[index, :]

for i in range(5000):
    if i % 5 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y, tf_is_training: False})
        # print("step %d, train accuracy %g" % (i, train_accuracy))
        # test_predict = ytest_predict.eval(feed_dict={x: test_images, y_actual: test_labels, keep_prob: 1})
        # test_accuracy = accuracy.eval(feed_dict={x: test_images, y_actual: test_labels, keep_prob: 0.5})
        print("step %d, test accuracy %g" % (i, accuracy_))
        # print(test_predict)
    # train_step.run(feed_dict={x: train_x, y_actual: train_y})
    _, loss_ = sess.run([train_op, loss], {tf_x: train_x, tf_y: train_y, tf_is_training: True})
# saver.save(sess, 'model/my-model-feature5')
