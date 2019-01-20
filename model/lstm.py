#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from __future__ import division

import logging
import tensorflow as tf

from ner_model import NERModel

logger = logging.getLogger('ner_model')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class BiLSTM(NERModel):
    """
    Implements a Bi-LSTM network.
    """

    def __init__(self, helper, config, pretrained_embeddings, datautil):
        super(BiLSTM, self).__init__(helper, config, datautil)
        self.report = config.is_report
        self.dropout_rate = config.bi_dropout_rate
        self.max_length = min(config.max_length, helper.max_length)
        self.num_hiddens = self.max_length
        self.n_classes = config.n_classes
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.transition_params = None
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.seqlen_placeholser = None
        
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
        self.seqlen_placeholser = tf.placeholder(
            tf.int32, shape= (None), name = 'seqlen')

    def create_feed_dict(self, inputs_batch, mask_batch, seqlen_batch, labels_batch=None):
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
            self.mask_placeholder: mask_batch,
            self.seqlen_placeholser: seqlen_batch
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
        fwcell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hiddens, forget_bias=1.0)
        bwcell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hiddens, forget_bias=1.0)

        with tf.variable_scope('Decode'):
            W = tf.get_variable('W', (self.num_hiddens * 2, self.n_classes), initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', (self.n_classes), initializer=tf.constant_initializer(0))
        with tf.variable_scope("BiLSTM"):
            (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                                                                    fwcell, 
                                                                    bwcell, 
                                                                    inputs,
                                                                    dtype=tf.float32)
            outputs = tf.concat(values=[forward_output, backward_output], axis=2)
            outputs = tf.nn.dropout(outputs, self.dropout_rate)
            outputs = tf.reshape(outputs, [-1, 2*self.num_hiddens])
            outputs = tf.matmul(outputs, W) + b
            preds = tf.reshape(outputs, [-1, self.max_length, self.n_classes])
            # print(preds.shape)

        assert preds.get_shape().as_list() == [None, self.max_length, self.n_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.max_length, self.config.n_classes], preds.get_shape().as_list())
        # preds = tf.reshape(preds, shape=[-1, self.max_length, self.config.n_classes])
        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        '''masked_logits = tf.boolean_mask( preds, self.mask_placeholder)
        masked_labels = tf.boolean_mask( self.labels_placeholder, self.mask_placeholder)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits( logits = masked_logits,
                                                            labels = masked_labels )
        )'''
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            preds, self.labels_placeholder, self.seqlen_placeholser)
        loss = tf.reduce_mean(-log_likelihood)
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
            _, _, mask, _ = examples[i]
            labels_ = [l for l, m in zip(preds[i], mask) if m] # only select elements of mask.
            assert len(labels_) == len(labels),'{}:{}:{}'.format(i,len(labels),len(labels_))
            ret.append([sentence, labels, labels_])
        return ret

    def predict_on_batch(self, sess, inputs_batch, mask_batch, seqlen_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch,
                                     mask_batch=mask_batch,
                                     seqlen_batch=seqlen_batch)
        scores = sess.run(self.pred, feed_dict=feed)
        transition_params = sess.run(self.transition_params, feed_dict=feed)
        #predictions, _ = tf.contrib.crf.viterbi_decode(scores, self.transition_params)
        #decode_tags, _ = tf.contrib.crf.crf_decode(self.pred, self.transition_params, seqlen_batch)
        #predictions = sess.run(decode_tags, feed_dict=feed)
        predictions = []
        for score_, seq_len_ in zip(scores, seqlen_batch):
            score_ = score_[:seq_len_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(score_, transition_params)
            predictions.append(viterbi_sequence)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch, seqlen_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch,
                                     labels_batch=labels_batch,
                                     mask_batch=mask_batch,
                                     seqlen_batch=seqlen_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def preprocess_sequence_data(self, examples):
        ret = self.datautil.pad_sequences(examples, self.max_length)
        return ret

