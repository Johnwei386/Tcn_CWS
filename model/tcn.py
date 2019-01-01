#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from __future__ import division

import logging
import tensorflow as tf

from ner_model import NERModel
from tcn_cell import TemporalBlock

logger = logging.getLogger('ner_model')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class TemporalConvnet(NERModel):
    """
    Implements a temporal convnetwork.
    """

    def __init__(self, helper, config, pretrained_embeddings, datautil):
        super(TemporalConvnet, self).__init__(helper, config, datautil)
        self.report = config.is_report
        self.isDropout = self.config.training
        self.max_length = min(config.max_length, helper.max_length)
        self.pretrained_embeddings = pretrained_embeddings
        self.num_channels = [config.filters_size] * (config.num_layers - 1) + [config.embed_size]
        self.layers = []
        for i in range(config.num_layers):
            dilation_size = 2 ** i
            out_channels = self.num_channels[i]
            self.layers.append(
                TemporalBlock(n_outputs=out_channels,
                              kernel_size=config.kernel_size, 
                              strides=1, 
                              dilation_rate=dilation_size, 
                              dropout=config.dropout,
                              name="tblock_{}".format(i))
            )
        self.decoder = tf.layers.Dense(config.n_classes,
                                       activation=None,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        
        self.build()

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.max_length), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        """
        self.input_placeholder = tf.placeholder(
            tf.int32, shape = (None, self.max_length), name = 'input')
        self.labels_placeholder = tf.placeholder(
            tf.int32, shape = (None, self.max_length), name = 'labels')
        self.mask_placeholder = tf.placeholder(
            tf.bool, shape = (None, self.max_length), name = 'mask')

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None):
        """Creates the feed_dict for the dependency parser.
        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.mask_placeholder: mask_batch
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        embeddings = tf.nn.embedding_lookup(
            tf.Variable(self.pretrained_embeddings),
            self.input_placeholder)
        embeddings = tf.reshape(
            embeddings, [-1, self.max_length, self.config.embed_size])

        return embeddings

    def add_prediction_op(self):
        """All the forword computation
        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """
        inputs = self.add_embedding()
        with tf.variable_scope("TCN"):
            # print(inputs.shape)
            outputs = inputs
            for layer in self.layers:
                outputs = layer(outputs, training=self.isDropout)
            preds = self.decoder(outputs)
            # print(preds.shape)

        assert preds.get_shape().as_list() == [None, self.max_length, self.config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.max_length, self.config.n_classes], preds.get_shape().as_list())
        # preds = tf.reshape(preds, shape=[-1, self.max_length, self.config.n_classes])
        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        masked_logits = tf.boolean_mask( preds, self.mask_placeholder)
        masked_labels = tf.boolean_mask( self.labels_placeholder, self.mask_placeholder)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits( logits = masked_logits,
                                                            labels = masked_labels )
        )
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        for i, (sentence, labels) in enumerate(examples_raw):
            _, _, mask = examples[i]
            labels_ = [l for l, m in zip(preds[i], mask) if m] # only select elements of mask.
            assert len(labels_) == len(labels)
            ret.append([sentence, labels, labels_])
        return ret

    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch,
                                     mask_batch=mask_batch)
        self.isDropout = False
        predictions = sess.run(tf.argmax(self.pred, axis=2), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch,
                                     labels_batch=labels_batch,
                                     mask_batch=mask_batch)
        self.isDropout = True
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def preprocess_sequence_data(self, examples):
        ret = self.datautil.pad_sequences(examples, self.max_length)
        return ret

