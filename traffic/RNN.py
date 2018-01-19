import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
# tf.set_random_seed(1)
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
training_iters = 3000000
batch_size = 180
n_inputs = 6
n_steps = 10
n_hidden_units = 180
n_classes = 6
keep_prob = 0.5

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])  # [150,16,80]
y = tf.placeholder(tf.float32, [None, n_classes])  # [150,6]

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),   # [80,150]
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))  # [150,6]
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),  # [150,]
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))  # [6,]
}


def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, n_inputs])  # [150*16,80]
    X_in = tf.matmul(X, weights['in']) + biases['in']  # [150*16,150]
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])  # [150,16,150]

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # [150,150]

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

x_s = np.loadtxt('Comnet-14_feature6_RNN.txt')
y_s = np.loadtxt('Comnet-14_feature6_RNN_label.txt')
x_train = np.array([[0] * 60 for i in range(5580)])
y_train = np.array([[0] * 6 for i in range(5580)])
for index in range(5580):
    for j in range(60):
        x_train[index][:] = x_s[index, :]
    for k in range(6):
        y_train[index][:] = y_s[index, :]
x_train = tf.reshape(x_train, [-1, 60])
x_train = tf.cast(x_train, tf.float32)
y_train = tf.reshape(y_train, [-1, 6])
y_train = tf.cast(y_train, tf.float32)
epsilon = 0.001
x_mean, x_var = tf.nn.moments(x_train, axes=[0],)
x_scale = tf.Variable(tf.ones([1]))
x_shift = tf.Variable(tf.zeros([1]))
x_train_normal = tf.nn.batch_normalization(x_train, x_mean, x_var, x_shift, x_scale, epsilon)

y_mean, y_var = tf.nn.moments(y_train, axes=[0],)
y_scale = tf.Variable(tf.ones([1]))
y_shift = tf.Variable(tf.zeros([1]))
y_train_normal = tf.nn.batch_normalization(y_train, y_mean, y_var, y_shift, y_scale, epsilon)
# x_test = np.loadtxt('Comnet-14_feature6_test.txt')
# y_test = np.loadtxt('Comnet-14_feature6_test_label.txt')

dataset = tf.data.Dataset.from_tensor_slices((x_train_normal, y_train_normal)).shuffle(buffer_size=55800).batch(batch_size).repeat()
iterator = dataset.make_initializable_iterator()
one_element = iterator.get_next()
# dataset1 = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(buffer_size=6000).batch(batch_size).repeat()
# iterator1 = dataset1.make_one_shot_iterator()
# one_element1 = iterator1.get_next()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(iterator.initializer)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = sess.run(one_element)
        # batch_xs1, batch_ys1 = sess.run(one_element1)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        # batch_xs1 = batch_xs1.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
        step += 1


