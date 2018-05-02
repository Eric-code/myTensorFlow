import tensorflow as tf
import numpy as np


# Hyper Parameters
BATCH_SIZE = 64
TIME_STEP = 1          # rnn time step / image height
INPUT_SIZE = 19         # rnn input size / image width
LR = 0.01               # learning rate


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


train_x, train_y = read_data("flow_15s_19_train_label.csv")
test_x, test_y = read_data("flow_15s_19_test_label.csv")

# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])       # shape(batch, 784)
image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])                   # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 8])                             # input y

# RNN
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    image,                      # input
    initial_state=None,         # the initial hidden state
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
output = tf.layers.dense(outputs[:, -1, :], 8)              # output based on the last output step

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(BATCH_SIZE).repeat()
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
dataset1 = tf.data.Dataset.from_tensor_slices((test_x, test_y)).shuffle(buffer_size=2000).batch(BATCH_SIZE).repeat()
iterator1 = dataset1.make_one_shot_iterator()
one_element1 = iterator1.get_next()
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

for step in range(60000):    # training
    batch_xs, batch_ys = sess.run(one_element)
    batch_xs1, batch_ys1 = sess.run(one_element1)
    batch_xs = batch_xs.reshape([BATCH_SIZE, TIME_STEP*INPUT_SIZE])
    batch_xs1 = batch_xs1.reshape([BATCH_SIZE, TIME_STEP*INPUT_SIZE])
    _, loss_ = sess.run([train_op, loss], {tf_x: batch_xs, tf_y: batch_ys})
    if step % 50 == 0:      # testing
        accuracy_ = sess.run(accuracy, {tf_x: batch_xs1, tf_y: batch_ys1})
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)


