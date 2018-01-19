
#头文件
import tensorflow as tf 
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data


#数据占位    
x = tf.placeholder(tf.float32, [None, 600])
y_actual = tf.placeholder(tf.float32, shape=[None, 7])
         


#定义函数，用于初始化权值 W，初始化偏置b，定义卷积层，定义池化层
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='SAME')

def max_pool_2(x):
  return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


#构建网络
x_image = tf.reshape(x, [-1, 10, 60, 1])
W_conv1 = weight_variable([1, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = conv2d(x_image, W_conv1) + b_conv1     
h_pool1 = max_pool(h_conv1)                                  

W_conv2 = weight_variable([1, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2     
h_pool2 = max_pool_2(h_conv2)                                   

W_fc1 = weight_variable([10*10*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 10*10*64])
h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1   

keep_prob = tf.placeholder("float") 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  
W_fc2 = weight_variable([1024, 7])
b_fc2 = bias_variable([7])
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   



#模型优化
cross_entropy =tf.reduce_mean((y_actual-y_predict)*(y_actual-y_predict))     
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)   
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                 
sess=tf.InteractiveSession()                          
sess.run(tf.global_variables_initializer())



#读入真实数据
input_count = 7080
x_s = np.loadtxt('Comnet_CRNN_train.txt')
y_s = np.loadtxt('Comnet_CRNN_train_label.txt')
train_images = np.array([[0]*600 for i in range(input_count)])
train_labels = np.array([[0]*7 for i in range(input_count)])

for index in range(input_count):
    for j in range(600):
        train_images[index][:] = x_s[index, :]
    for k in range(7):
        train_labels[index][:] = y_s[index, :]


print(x_s)

for i in range(100000):
    if i % 20 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: train_images, y_actual: train_labels, keep_prob: 0.5})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: train_images, y_actual: train_labels, keep_prob: 0.5})



