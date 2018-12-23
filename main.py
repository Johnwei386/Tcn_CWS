#!/usr/bin/env python
# _*_ coding:utf8 _*_

import argparse
import logging
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from Config import Config
from utils.data_util import DataUtil
from model.ner_model import NERModel
from model.tcn import TemporalConvnet

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
        elif config.model == 'uni-lstm':
            pass
        elif config.model == 'bi-lstm':
            pass
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
            logger.info("Bets F1 score is %.4f", best_score)
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



if __name__ == '__main__':
    paser = argparse.ArgumentParser(description='Tuning with temporal Convolutional network for CWS')
    paser.add_argument('--model', choices=['tcn','uni-lstm','bi-lstm'], help='Which model to perform', default='tcn')
    paser.add_argument('--embedding', help='Path of char Embedding', default='datasets/charVec100d.txt')
    paser.add_argument('--train', help='Dir of train data', default='datasets/nlpcc2016/train')
    paser.add_argument('--dev', help='Dir of evaluation data', default=None)
    paser.add_argument('--test', help='Dir of test data', default='datasets/nlpcc2016/test')
    paser.add_argument('--model_path', help='Dir of saving model parameter', default=None)
    paser.add_argument('--training', help='Whether perform train step', default=True)
    args = paser.parse_args()

    training = args.training
    config = Config(args)
    if training:
        train(args, config)
    else:
        pass

