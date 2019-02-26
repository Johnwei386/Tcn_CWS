#!/usr/bin/env python
# _*_ coding:utf8 _*_

import logging
import tensorflow as tf
import numpy as np
from model import Model
from auxiliary import ConfusionMatrix, Progbar

logger = logging.getLogger("ner_model")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class NERModel(Model):
    """
    Implements special functionality for NER models.
    A model for named entity recognition.
    """

    def __init__(self, helper, config, datautil):
        self.helper = helper
        self.config = config
        self.datautil = datautil
        self.report = config.is_report
        self.isDropout = config.training

    def preprocess_sequence_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of vectorized input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def consolidate_predictions(self, data_raw, data, preds):
        """
        Convert a sequence of predictions according to the batching
        process back into the original sequence.
        """
        raise NotImplementedError("Each Model must re-implement this method.")


    def evaluate(self, sess, examples, examples_raw):
        """Evaluates model performance.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples_raw: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
        LBLS = self.config.LABELS
        token_cm = ConfusionMatrix(labels=LBLS)

        correct_preds, total_correct, total_preds = 0., 0., 0.
        for _, labels, labels_  in self.output(sess, examples_raw, examples):
            for l, l_ in zip(labels, labels_):
                token_cm.update(l, l_)
            gold = set(self.datautil.get_chunks_cws(labels)) 
            pred = set(self.datautil.get_chunks_cws(labels_))
            correct_preds += len(gold.intersection(pred))
            total_preds += len(pred)
            total_correct += len(gold)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0 
        return token_cm, (p, r, f1)

    def run_epoch(self, sess, train_examples, dev_set, train_examples_raw, dev_set_raw):
        prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))
        losses = []
        for i, batch in enumerate(self.datautil.minibatches(train_examples)):
            loss = self.train_on_batch(sess, *batch)
            prog.update(i + 1, [("train loss", loss)])
            losses.append(loss)
            if self.report: self.report.log_train_loss(loss)
        print("")

        self.isDropout = False
        logger.info("Evaluating on development data")
        token_cm, entity_scores = self.evaluate(sess, dev_set, dev_set_raw)
        logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        #logger.debug("Token-level scores:\n" + token_cm.summary())
        logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        f1 = entity_scores[-1]
        return f1, np.mean(losses)

    def output(self, sess, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        if inputs is None:
            inputs = self.preprocess_sequence_data(self.helper.vectorize(inputs_raw, self.config))

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(self.datautil.minibatches(inputs, shuffle=False)):
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds)

    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        best_score = 0.0
        losses = []
        scores = []

        train_examples = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)
        
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score,loss = self.run_epoch(sess, train_examples, dev_set, train_examples_raw, dev_set_raw)
            scores.append(score)
            losses.append(loss)
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s",
                                self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()
        return best_score, scores, losses
