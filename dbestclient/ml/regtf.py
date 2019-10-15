# Created by Qingzhi Ma at 14/10/2019
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

import numpy as np
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha
from PIL import Image
import random
import sys


def test():
    m1 = tf.constant([[3, 3]])
    m2 = tf.constant([[2], [3]])
    product = tf.matmul(m1, m2)

    with tf.Session() as sess:
        result = sess.run(product)
        print(result)


def sub():
    x = tf.Variable([1, 2])
    a = tf.constant([3, 3])
    sub = tf.subtract(x, a)
    add = tf.add(x, sub)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        print(sess.run(sub))
        print(sess.run(add))

    state = tf.Variable(0, name='counter')
    new_value = tf.add(state, 1)
    update = tf.compat.v1.assign(state, new_value)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(state))
        for _ in range(5):
            print(sess.run(update))


def fetch():
    input1 = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)

    add = tf.add(input2, input3)
    mul = tf.multiply(input1, add)

    with tf.Session() as sess:
        result = sess.run([mul, add])
        print(result)


def feed():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        print(sess.run(output, feed_dict={input1: [7.0], input2: [2.0]}))


def tf_example():
    x_data = np.random.rand(100)
    y_data = x_data * 0.1 + 0.2

    b = tf.Variable(0.1)
    k = tf.Variable(0.1)
    y = k * x_data + b

    loss = tf.reduce_mean(tf.square(y_data - y))
    optimizer = tf.train.GradientDescentOptimizer(0.2)

    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for step in range(201):
            sess.run(train)
            if step % 20 == 0:
                print(step, sess.run([k, b]))


def reg3():
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    noise = np.random.normal(0, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise

    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    # intermediate layer
    Weights_L1 = tf.Variable(tf.random.normal([1, 10]))
    biases_L1 = tf.Variable(tf.zeros([10]))
    Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
    L1 = tf.nn.tanh(Wx_plus_b_L1)

    # output layer
    Weights_L2 = tf.Variable(tf.random.normal([10, 1]))
    biases_L2 = tf.Variable(tf.zeros([1]))
    Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
    prediction = tf.nn.tanh(Wx_plus_b_L2)

    # loss
    loss = tf.reduce_mean(tf.square(y - prediction))
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(2000):
            sess.run(train_step, feed_dict={x: x_data, y: y_data})

        # get predictions
        prediction_value = sess.run(prediction, feed_dict={x: x_data})

        plt.figure()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, prediction_value, 'r-', lw=5)
        plt.show()


def mnist3():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    max_steps = 1001

    image_num = 3000

    DIR = '/home/u1796377/Desktop/'

    sess = tf.Session()

    embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

    batch_size = 100
    n_batch = mnist.train.num_examples // batch_size

    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    with tf.name_scope('input'):
        # define placeholders
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # show images
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.zeros([784, 10]), name='W')
            variable_summaries(W)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.zeros([10]), name='b')
            variable_summaries(b)
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.matmul(x, W) + b
        with tf.name_scope('softmax'):
            prediction = tf.nn.softmax(wx_plus_b)
    # create NN

    # W_L1 = tf.Variable(tf.zeros([784, 100]))
    # b_L1 = tf.Variable(tf.zeros([100]))
    # Wx_plus_b_L1 = tf.matmul(x, W_L1) + b_L1
    # L1 = tf.nn.tanh(Wx_plus_b_L1)
    #
    # W_L2 = tf.Variable(tf.random.normal([100, 10], 1, 0.1))
    # # W_L2 = tf.Variable(tf.zeros([100, 10]))
    # b_L2 = tf.Variable(tf.zeros([10]))
    # Wx_plus_b_L2 = tf.matmul(L1, W_L2) + b_L2
    # prediction = tf.nn.softmax(Wx_plus_b_L2)

    # loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
        tf.summary.scalar('loss', loss)
    # loss = tf.reduce_mean(tf.square(y - prediction))

    # optimizer
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(2).minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    # create metadata
    if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
        tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
    with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
        labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
        for i in range(image_num):
            f.write(str(labels[i]) + '\n')

    merged = tf.summary.merge_all()

    # projector writer
    projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
    saver = tf.train.Saver()
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = embedding.name
    embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
    embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(projector_writer, config)

    for i in range(max_steps):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys}, options=run_options,
                              run_metadata=run_metadata)
        projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        projector_writer.add_summary(summary, i)

        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("Iter " + str(i) + ", Testing Accuracy " + str(acc))
    saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
    projector_writer.close()
    sess.close()

    # writer =tf.summary.FileWriter('/home/u1796377/Desktop/logs/', sess.graph)
    # for epoch in range(51):
    #     for batch in range(n_batch):
    #         batch_xs, batch_ys= mnist.train.next_batch(batch_size)
    #         summary, _ = sess.run([merged, train_step], feed_dict={x:batch_xs, y:batch_ys})
    #     writer.add_summary(summary, epoch)
    #
    #     acc = sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels})
    #     print("Iter "+ str(epoch) +", Testing Accuracy "+ str(acc))


def dropout4():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    batch_size = 100
    n_batch = mnist.train.num_examples // batch_size

    # define placeholders
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    lr = tf.Variable(0.001, dtype=tf.float32)

    # create NN
    W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
    b1 = tf.Variable(tf.zeros([500]) + 0.1)
    L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
    L1_drop = tf.nn.dropout(L1, keep_prob)

    W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
    b2 = tf.Variable(tf.zeros([300]) + 0.1)
    L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
    L2_drop = tf.nn.dropout(L2, keep_prob)

    # W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
    # b3 = tf.Variable(tf.zeros([1000]) + 0.1)
    # L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
    # L3_drop = tf.nn.dropout(L3, keep_prob)

    W4 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
    b4 = tf.Variable(tf.zeros([10]) + 0.1)
    prediction = tf.nn.softmax(tf.matmul(L2_drop, W4) + b4)

    # W_L1 = tf.Variable(tf.zeros([784, 100]))
    # b_L1 = tf.Variable(tf.zeros([100]))
    # Wx_plus_b_L1 = tf.matmul(x, W_L1) + b_L1
    # L1 = tf.nn.tanh(Wx_plus_b_L1)
    #
    # W_L2 = tf.Variable(tf.random.normal([100, 10], 1, 0.1))
    # # W_L2 = tf.Variable(tf.zeros([100, 10]))
    # b_L2 = tf.Variable(tf.zeros([10]))
    # Wx_plus_b_L2 = tf.matmul(L1, W_L2) + b_L2
    # prediction = tf.nn.softmax(Wx_plus_b_L2)

    # loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # loss = tf.reduce_mean(tf.square(y - prediction))
    # optimizer
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(51):
            sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
            print(
                "Iter " + str(epoch) + ", Testing Accuracy " + str(test_acc) + ", Training Accuracy " + str(train_acc))


def cnn6():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    batch_size = 100
    n_batch = mnist.train.num_examples // batch_size

    # initialize weight
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # initialize biases
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # conv layer
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # pool layer
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    # reshape the figure to 4D
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # initialize the first conv layer's weights and biases
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # initialize the second conv layer's weights and biases
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # initialize the second fully-conntected layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(21):
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))


def cnn6_summary():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    batch_size = 100
    n_batch = mnist.train.num_examples // batch_size

    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    # initialize weight
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    # initialize biases
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    # conv layer
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # pool layer
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])
        with tf.name_scope('x_image'):
            # reshape the figure to 4D
            x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('Conv1'):
        # initialize the first conv layer's weights and biases
        with tf.name_scope('W_Conv1'):
            W_conv1 = weight_variable([5, 5, 1, 32], name='W_Conv1')
        with tf.name_scope('b_Conv1'):
            b_conv1 = bias_variable([32], name='b_Conv1')
        with tf.name_scope('conv2d_1'):
            conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
        with tf.name_scope('relu'):
            h_conv1 = tf.nn.relu(conv2d_1)
        with tf.name_scope('h_pool1'):
            h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('Conv2'):
        # initialize the second conv layer's weights and biases
        with tf.name_scope('W_conv2'):
            W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')
        with tf.name_scope('b_conv2'):
            b_conv2 = bias_variable([64], name='b_conv2')
        with tf.name_scope('conv2d_2'):
            conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
        with tf.name_scope('relu'):
            h_conv2 = tf.nn.relu(conv2d_2)
        with tf.name_scope('h_pool2'):
            h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        with tf.name_scope('W_fc1'):
            W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')
        with tf.name_scope('b_fc1'):
            b_fc1 = bias_variable([1024], name='b_fc1')

        with tf.name_scope('h_pool2_flat'):
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='h_pool2_flat')
        with tf.name_scope('wx_plus_b1'):
            wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        with tf.name_scope('relu'):
            h_fc1 = tf.nn.relu(wx_plus_b1)

        with tf.name_scope('keep_prob'):
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        with tf.name_scope('h_fc1_drop'):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

    # initialize the second fully-conntected layer
    with tf.name_scope('fc2'):
        with tf.name_scope('W_fc2'):
            W_fc2 = weight_variable([1024, 10], name='W_fc2')
        with tf.name_scope('b_fc2'):
            b_fc2 = bias_variable([10], name='b_fc2')
        with tf.name_scope('wx_plus_b2'):
            wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        with tf.name_scope('prediction'):
            prediction = tf.nn.softmax(wx_plus_b2)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                       name='cross_entropy')
        tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('Accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        with tf.name_scope('Accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("Accuracy", accuracy)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('/home/u1796377/Desktop/logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('/home/u1796377/Desktop/logs/test', sess.graph)
        for i in range(1001):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})

            summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            train_writer.add_summary(summary, i)
            batch_xs, batch_ys = mnist.test.next_batch(batch_size)
            summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            test_writer.add_summary(summary, i)

            if i % 100 == 0:
                test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
                train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000],
                                                          keep_prob: 1.0})
                print("Iter " + str(i) + ", Testing Accuracy= " + str(test_acc) + ",  Training Accuracy= " + str(
                    train_acc))


def rnn7():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    n_inputs = 28
    max_time = 28
    lstm_size = 100
    n_classes = 10
    batch_size = 50
    n_batch = mnist.train.num_examples // batch_size

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

    # define RNN
    def RNN(X, weights, biases):
        inputs = tf.reshape(X, [-1, max_time, n_inputs])
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
        results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
        return results

    prediction = RNN(x, weights, biases)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(6):
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))


def image10():
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # randomly generate a sequence of words
    def random_captcha_text(char_set=number, captcha_size=4):
        captcha_text=[]
        for i in range(captcha_size):
            c= random.choice(char_set)
            captcha_text.append(c)
        return captcha_text

    # generate
    def gen_captcha_text_and_image():
        image = ImageCaptcha()
        captcha_text = random_captcha_text()
        captcha_text = ''.join(captcha_text)

        captcha = image.generate(captcha_text)
        image.write(captcha_text,'/home/u1796377/Desktop/captcha/'+ captcha_text+'.jpg')



    num = 10000
    for i in range(num):
        gen_captcha_text_and_image()
        print("Creating image %d/%d"%(i+1, num))
    print("finished.")

def tfrecord():
    _NUM_TEST=500
    _RANDOM_SEED = 0
    DATASET_DIR = "/home/u1796377/Desktop/captcha/"
    TFRECORD_DIR = "/home/u1796377/Desktop/tfrecord/"

    def _dataset_exists(dataset_dir):
        for split_name in ['train', 'test']:
            output_filename = os.path.join(dataset_dir, split_name+'.tfrecords')
            if not tf.gfile.Exists(output_filename):
                return False
        return True

    def _get_filenames_and_classes(dataset_dir):
        photo_filenames =[]
        for filename in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, filename)
            photo_filenames.append(path)
        return photo_filenames

    def int64_feature(values):
        if not isinstance(values, (tuple,list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def bytes_feature(values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

    def image_to_tfexample(image_data, label0, label1, label2, label3):
        return tf.train.Example(features=tf.train.Features(feature={
            'image':bytes_feature(image_data),
            'label0':int64_feature(label0),
            'label1': int64_feature(label1),
            'label2': int64_feature(label2),
            'label3': int64_feature(label3),
        }))

    def _convert_dataset(split_name, filenames, dataset_dir):
        assert split_name in ['train', 'test']

        with tf.Session() as sess:
            output_filename = os.path.join(TFRECORD_DIR,split_name+'.tfrecords')
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i, filename in enumerate(filenames):
                    try:
                        sys.stdout.write("\r Converting image %d/%d"%(i+1, len(filenames)))
                        sys.stdout.flush()

                        image_data = Image.open(filename)
                        image_data =image_data.resize((224,224))
                        image_data = np.array(image_data.convert('L'))
                        image_data =image_data.tobytes()

                        labels = filename.split('/')[-1][0:4]
                        num_labels =[]
                        for j in range(4):
                            num_labels.append(int(labels[j]))

                        example = image_to_tfexample(image_data,num_labels[0],num_labels[1],num_labels[2],num_labels[3],)
                        tfrecord_writer.write((example.SerializeToString()))
                    except IOError as e:
                        print("could not read:", filename)
                        print(e)
        sys.stdout.write('\n')
        sys.stdout.flush()

    if _dataset_exists(TFRECORD_DIR):
        print('tfrecords exist!')
    else:
        photo_filenames = _get_filenames_and_classes(DATASET_DIR)
        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)
        training_filenames = photo_filenames[_NUM_TEST:]
        testing_filenames = photo_filenames[:_NUM_TEST]

        _convert_dataset('train', training_filenames, DATASET_DIR)
        _convert_dataset('test', testing_filenames, DATASET_DIR)
        print("tfrecord files created!")





    # model = tf.keras.Sequential()
    # # Adds a densely-connected layer with 64 units to the model:
    # model.add(layers.Dense(64, activation='relu'))
    # # Add another:
    # model.add(layers.Dense(64, activation='relu'))
    # # Add a softmax layer with 10 output units:
    # model.add(layers.Dense(10, activation='softmax'))
    #
    # model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    #
    #
    # data = np.random.random((1000, 32))
    # labels = np.random.random((1000, 10))
    # val_data = np.random.random((1000, 32))
    # val_labels = np.random.random((1000, 10))
    #
    # # Instantiates a toy dataset instance:
    # dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    # dataset = dataset.batch(32)
    #
    # val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    # val_dataset = val_dataset.batch(32)
    #
    # model.fit(dataset, epochs=10,
    #           validation_data=val_dataset)
    #
    # result = model.predict(data, batch_size=32)
    # print(result.shape)
