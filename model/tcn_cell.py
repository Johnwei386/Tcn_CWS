#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from __future__ import division

import logging
import tensorflow as tf
import numpy as np

logger = logging.getLogger('ner_model')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class TemporalBlock(tf.keras.layers.Layer):
    def __init__(self, n_outputs, kernel_size, dilation_rate=1, strides=1, dropout=0.3, name=None):
        super(TemporalBlock, self).__init__()
        self.n_outputs = n_outputs
        self.dropout = dropout
        padding_size = (kernel_size - 1) * dilation_rate
        # the Future scope scheme
        self.padding = tf.constant([[0, 0], [0, padding_size], [0, 0]]) # padding after the last word of sentence
        # the Past scope scheme
        #self.padding = tf.constant([[0, 0], [padding_size, 0], [0, 0]])
        self.conv1 = tf.layers.Conv1D(n_outputs, 
                                      kernel_size, 
                                      strides,
                                      dilation_rate = dilation_rate,
                                      padding = 'valid',
                                      kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                      activation = tf.nn.relu,
                                      name = 'conv1')
        self.conv2 = tf.layers.Conv1D(n_outputs, 
                                      kernel_size, 
                                      strides,
                                      dilation_rate = dilation_rate,
                                      padding = 'valid',
                                      kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                      activation = tf.nn.relu,
                                      name = 'conv2')
        self.dropout1 = tf.layers.Dropout(self.dropout, noise_shape = [1, 1, self.n_outputs])
        self.dropout2 = tf.layers.Dropout(self.dropout, noise_shape = [1, 1, self.n_outputs])
        self.down_sample = None

    def build(self, input_shape):
        channel_dim = 2
        # print(input_shape[channel_dim])
        if input_shape[channel_dim] != self.n_outputs:
            self.down_sample = tf.layers.Dense(self.n_outputs, activation=None)

    def __call__(self, inputs, training=True):
        input_shape = inputs.shape
        self.build(input_shape)
        #print(input_shape)
        x = tf.pad(inputs, self.padding)
        # return x
        #print(x.shape)
        x = self.conv1(x)        
        x = tf.contrib.layers.layer_norm(x)
        if training:
            x = self.dropout1(x, training=training)
        #x = self.dropout1(x, training=training)
        # print(x.shape)
        x = tf.pad(x, self.padding)
        # print(x.shape)
        # return x
        x = self.conv2(x)
        x = tf.contrib.layers.layer_norm(x)
        if training:
            x = self.dropout2(x, training=training)
        x = self.dropout2(x, training=training)
        #print(x.shape)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        return tf.nn.relu(x + inputs)

def test_tcn_cell():
    with tf.Graph().as_default():
        with tf.variable_scope("test_tcn_cell"):
            x_placeholder = tf.placeholder(tf.float32, shape=(None, 3, 4))
            tcnblock = TemporalBlock(4, 2)
            preds = tcnblock(x_placeholder, training=True)

            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                np.random.seed(121)
                # x = np.random.randn(2, 3, 4)
                x = np.arange(1, 25).reshape((2, 3, 4))
                print(x)
                y = session.run([preds], feed_dict={x_placeholder: x})
                print(y)

if __name__ == '__main__':
    logger.info("Testing tcn_cell")
    test_tcn_cell()
    logger.info("Passed!")
