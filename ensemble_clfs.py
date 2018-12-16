import pandas as pd
import numpy as np
from random import shuffle
import tensorflow as tf
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelBinarizer

#parameters 
learning_rate = 0.0001
in_num = 300
#minmaxscaler for normalization
def MinMaxScaler(data):
    buj = data - np.min(data, 0)
    bum = np.max(data, 0) - np.min(data, 0)
    return buj / (bum + 1e-7)

#read cuisine datas and change them to one-hot vector
path = 'C:/Users/10User/Desktop/JH/all/train.json'
data = pd.read_json(path, encoding="utf-8")
cuis = pd.DataFrame(data['cuisine'])
one_cuis = np.array(cuis)
encoder1 = LabelBinarizer()
label_oh = encoder1.fit_transform(one_cuis)

#read vector from .csv file and do minmax scaler normalization for whole dataset
xy = np.loadtxt('C:/Users/10User/Desktop/JH/all/ft300_sg.csv', delimiter=',', dtype=np.float32)
x_data = xy[:,:]
y_data = label_oh
x_data = MinMaxScaler(x_data)

#data shuffle
c = list(zip(x_data, y_data))
shuffle(c)
x_data, y_data = zip(*c)

#dividing data to train:8 & test:2
train_data_x = np.array(x_data[0:int(0.8*len(x_data))])
train_data_y = np.array(y_data[0:int(0.8*len(y_data))])
test_data_x = np.array(x_data[int(0.8*len(x_data)):])
test_data_y = np.array(y_data[int(0.8*len(y_data)):])
# reshape input data for convolution
train_data = np.reshape(train_data_x, [-1,in_num,1,1])
test_data = np.reshape(test_data_x, [-1,in_num,1,1])

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.cook_net()
    #model building
    def cook_net(self):
        with tf.variable_scope(self.name):
            # set droupout rate 0.7
            self.training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [None, in_num, 1, 1])
            self.Y = tf.placeholder(tf.int32, [None, 20])

            W1 = tf.get_variable("W1", shape=[5, 1, 1, 3],
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
            conv2d1 = tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='VALID')
            conv2d1 = tf.nn.relu(conv2d1)

            W2 = tf.get_variable("W2", shape=[5, 1, 3, 5],
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
            conv2d2 = tf.nn.conv2d(conv2d1, W2, strides=[1, 1, 1, 1], padding='VALID')
            conv2d2 = tf.nn.relu(conv2d2)

            W3 = tf.get_variable("W3", shape=[5, 1, 5, 10],
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
            conv2d3 = tf.nn.conv2d(conv2d2, W3, strides=[1, 1, 1, 1], padding='VALID')
            conv2d3 = tf.nn.relu(conv2d3)

            #flattern to connect fulley connected layer
            conv2d3 = tf.reshape(conv2d3, [-1, 2880])

            W4 = tf.get_variable("W4", shape=[2880, 512],
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
            b1 = tf.Variable(tf.zeros([512]))
            L1 = tf.nn.relu(tf.matmul(conv2d3, W4) + b1)
            L1 = tf.layers.dropout(inputs=L1, rate=0.7, training=self.training)

            W5 = tf.get_variable("W5", shape=[512, 512],
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
            b2 = tf.Variable(tf.zeros([512]))
            L2 = tf.nn.relu(tf.matmul(L1, W5) + b2)
            L2 = tf.layers.dropout(inputs=L2, rate=0.7, training=self.training)

            W6 = tf.get_variable("W6", shape=[512, 20],
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
            b3 = tf.Variable(tf.zeros([20]))

            self.logits = tf.matmul(L2, W6) + b3

        #cost/optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        c_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(c_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def g_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

# initialize
sess = tf.Session()
#append new models list for ensemble
models = []
num_models = 5
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))
sess.run(tf.global_variables_initializer())

print('Learning Started!')
# train models
for epoch in range(4001):
    avg_cost_list = np.zeros(len(models))
    for m_idx, m in enumerate(models):
        c, _ = m.train(train_data, train_data_y)
        avg_cost_list[m_idx] += c
    if epoch % 200 == 0:
        print('E:', '%04d' % (epoch + 1),'/', 'loss =', avg_cost_list)
print('Learning Finished!')

# test models acc & ensemble acc
td_size = len(test_data)
predictions = np.zeros(td_size * 20).reshape(td_size, 20)
for m_idx, m in enumerate(models):
    print('model',m_idx+1, 'Accuracy:', m.g_accuracy(test_data, test_data_y))
    p = m.predict(test_data)
    predictions += p

ensemble_c_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_data_y, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_c_prediction, tf.float32))

print(num_models,'ensemble model accuracy:', sess.run(ensemble_accuracy))
