#!/usr/bin/env python
# _*_ coding:utf8 _*_

import argparse
import logging
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from Config import Config
from utils.data_util import DataUtil, ModelHelper
from utils.util import read_conll
from model.ner_model import NERModel
from model.tcn import TemporalConvnet
from model.lstm import BiLSTM

logger = logging.getLogger('ner_model')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def train(args, config):
    train_dir = args.train
    dev_dir = args.dev
    datautil = DataUtil(config)
    helper, train_data, dev_data, train_raw, dev_raw = datautil.load_and_process_data(train_dir, dev_dir)
    embeddings = datautil.load_embeddings(args.embedding, helper)
    helper.save(config.output_path)

    # save log to file
    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = None
        if config.model == 'tcn':
            model = TemporalConvnet(helper, config, embeddings, datautil)
        elif config.model == 'bilstm':
            model = BiLSTM(helper, config, embeddings, datautil)
        else:
            logger.warning("Selected model does not exist!")
            assert False
        assert model is not None,"Model is None, stop excuted!"
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            best_score, scores, losses = model.fit(sess, saver, train_data, dev_data)
            logger.info("Each epoch F1 score is:\n")
            print(scores)
            logger.info("Each epoch loss value is:\n")
            print(losses)
            logger.info("Best F1 score is %.4f", best_score)

            # illustrate
            x = range(config.n_epochs)
            plt.figure(1)
            plt.plot(x, scores, 'b-')
            plt.xlabel("epoch")
            plt.ylabel("F1 score")
            #plt.title("Accurary")
            plt.figure(2)
            plt.plot(x, losses, 'b-')
            plt.xlabel("epoch")
            plt.ylabel("loss")
            #plt.title("Loss")
            plt.show()

def evaluate(args, config):
    test_dir = args.test
    datautil = DataUtil(config)
    helper = ModelHelper.load(args.model_path)
    test_raw = read_conll(test_dir)
    test_data = helper.vectorize(test_raw, config)
    embeddings = datautil.load_embeddings(args.embedding, helper)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = None
        if config.model == 'tcn':
            model = TemporalConvnet(helper, config, embeddings, datautil)
        elif config.model == 'bilstm':
            model = BiLSTM(helper, config, embeddings, datautil)
        else:
            logger.warning("Selected model does not exist!")
            assert False
        assert model is not None,"Model is None, stop excuted!"
        logger.info("took %.2f seconds", time.time() - start)
        test_examples = model.preprocess_sequence_data(test_data)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, config.model_output)
            token_cm, entity_scores = model.evaluate(sess, test_examples, test_data)
            logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
            logger.info("Entity level P/R/F1: %.4f/%.4f/%.4f", *entity_scores)


if __name__ == '__main__':
    paser = argparse.ArgumentParser(description='Tuning with temporal Convolutional network for CWS')
    paser.add_argument('-m', '--model', choices=['tcn', 'bilstm'], help='Which model to perform', default='tcn')
    paser.add_argument('-e', '--embedding', help='Path of char Embedding', default='datasets/vec100.txt')
    paser.add_argument('--train', help='Dir of train data', default='datasets/nlpcc2016/train')
    paser.add_argument('--dev', help='Dir of evaluation data', default=None)
    paser.add_argument('--test', help='Dir of test data', default='datasets/nlpcc2016/test')
    paser.add_argument('--model_path', help='Dir of files for saved model parameters')
    paser.add_argument('-o', choices=['train', 'evaluate'], help='Which opration to Perform for model')
    args = paser.parse_args()

    scheme = args.o
    config = Config(args)
    if scheme == 'train':
        train(args, config)
    elif scheme == 'evaluate':
        evaluate(args, config)
    else:
        pass

