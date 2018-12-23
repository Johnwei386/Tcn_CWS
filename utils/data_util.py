# _*_ coding:utf8 _*_
import os
import sys
import time
import pickle
import logging
import numpy as np
from util import read_conll, build_dict, normalize, load_word_vector_mapping, get_minibatches

logger = logging.getLogger('ner_model')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class DataUtil:
    def __init__(self, config):
        self.config = config
        self.multi_word = "Multi"
        self.single_word = "Single"

    def load_and_process_data(self, train_path, dev_path=None):
        # load train and dev data
        logger.info("Loading training data...")
        train_raw = read_conll(train_path)
        logger.info("Done. Read %d sentences", len(train_raw))
        logger.info("Loading dev data...")
        if dev_path is None:
            # samples from train in tail to dev
            dev_seg_size = self.config.dev_seg_size
            dev_raw = train_raw[len(train_raw) - dev_seg_size:]
            train_raw = train_raw[:len(train_raw) - dev_seg_size]
            logger.info("Divided train data. Read %d sentences of train", len(train_raw))
        else:
            dev_raw = read_conll(dev_path)
        logger.info("Done. Read %d sentences", len(dev_raw))

        helper = ModelHelper.build(train_raw, self.config)
        logger.info("Corpus of train max sentence length is %d", helper.max_length)

        # process all the input data
        train_data = helper.vectorize(train_raw, self.config)
        dev_data = helper.vectorize(dev_raw, self.config)

        return helper, train_data, dev_data, train_raw, dev_raw

    def load_embeddings(self, embedding_path, helper):
        logger.info("Loading embeddings...")
        ranseed = self.config.random_seed
        np.random.seed(ranseed)
        embeddings = np.array(np.random.randn(len(helper.tok2id) + 1, self.config.embed_size), dtype=np.float32)
        embeddings[0] = 0.0 # auto-broadcasting embedding size
        if embedding_path is not None and os.path.exists(embedding_path):
            logger.info("Loading pretrain embeddings...")
            pretrain_embeddings = load_word_vector_mapping(embedding_path)
            logger.info("Load %d pretrain word embeddings", len(pretrain_embeddings))
            sub_count = 0
            for word, vec in pretrain_embeddings.items():
                if self.config.is_normalize:
                    word = normalize(word)
                if word in helper.tok2id:
                    assert len(vec) == self.config.embed_size
                    embeddings[helper.tok2id[word]] = vec
                    sub_count += 1
            logger.info("Substitude %d embedding from pretrain embeddings", sub_count)

        logger.info("Initialize embeddings done.")
        return embeddings

    def get_chunks(self, seq, default):
        '''Breaks input of 4 4 4 0 0 4 0 -> (0, 4, 5), (0, 6, 7)'''
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            if tok == default and chunk_type is not None:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
            elif tok != default:
                if chunk is None:
                    chunk_type, chunk_start = tok, i
                elif tok != chunk_type:
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok, i
                else:
                    pass
            else:
                pass

        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            chunks.append(chunk)
        return chunks

    def get_chunks_cws(self, seq):
        chunks = []
        labels = self.config.LABELS
        chunk_type, chunk_start = None, None
        for i, tag in enumerate(seq):
            if tag == labels.index('B'):
                chunk_start = i
                chunk_type = self.multi_word
            elif tag == labels.index('E'):
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
            elif tag == labels.index('S'):
                chunk_type = self.single_word
                chunk = (chunk_type, i, i)
                chunks.append(chunk)
            else:
                pass
        return chunks

    def minibatches(self, data, shuffle=True):
        # batchs = [[sentences sets] [labels sets]]
        batches = [np.array(col) for col in zip(*data)]
        return get_minibatches(batches, self.config.batch_size, shuffle)

    def pad_sequences(self, data, max_length):
        """Ensures each input-output seqeunce pair in @data is of length
        @max_length by padding it with zeros and truncating the rest of the
        sequence.

        Args:
            data: is a list of (sentence, labels) tuples. @sentence is a list
                containing the words in the sentence and @labels is a list of
                output labels.For example,([[9],[1],[8],[13]],[0,1,2,3]).
            max_length: the desired length for all input/output sequences.
        Returns:
            a new list of data points of structure (sentence', labels', mask).
            Each os sentence',labels' and mask are of length @max_length.
        """
        ret = []
        zero_vector = 0
        zero_label = -1
        for sentence, labels in data:
            len_sentence = len(sentence)
            sentence = np.array(sentence, dtype=np.int32).reshape(len_sentence)
            #print(sentence.shape)
            add_length = max_length - len_sentence
            if add_length > 0:
                #filled_sentence = np.append(sentence, [zero_vector] * add_length)
                filled_sentence = np.concatenate((sentence, [zero_vector] * add_length))
                #print(filled_sentence.shape)                
                filled_labels = labels + ([zero_label] * add_length)
                mark = [True] * len_sentence
                mark.extend([False] * add_length)
            else:
                mark = [True] * max_length
                filled_sentence = sentence[:max_length]
                filled_labels = labels[:max_length]
            ret.append((filled_sentence, filled_labels, mark))
        return ret


class ModelHelper(object):
    """
    This helper takes care of preprocessing data, constructing embeddings, etc.
    """
    def __init__(self, tok2id, max_length):
        self.tok2id = tok2id
        self.max_length = max_length

    def vectorize_example(self, sentence, labels=None, config=None):
        assert config is not None,"Config can not be None"
        assert labels is not None,"labels can not be None"
        LBLS = config.LABELS
        UNK = config.UNK
        sentence_ = [[self.tok2id.get(normalize(word), self.tok2id[UNK])] if config.is_normalize  else [self.tok2id.get(word, self.tok2id[UNK])] for word in sentence]
        labels_ = [LBLS.index(l) for l in labels]
        return sentence_, labels_

    def vectorize(self, data, config):
        return [self.vectorize_example(sentence, labels, config) for sentence, labels in data]

    @classmethod # class method
    def build(cls, data, config):
        # Preprocess data to construct an embedding
        # Reserve 0 for the special NIL token.
        # return: {'char1': 1, ... ,'charN': N}, sorted by frequency of character
        tok2id = build_dict((normalize(word) if config.is_normalize else word for sentence, _ in data for word in sentence), offset=1)
        tok2id[config.UNK] = len(tok2id) + 1 
        # print(sorted(tok2id.values()))
        # print(tok2id[config.UNK])
        # tok2id index from 1
        assert sorted(tok2id.items(), key=lambda t: t[1])[0][1] == 1
        logger.info("Built dictionary for %d features.", len(tok2id))

        max_length = max(len(sentence) for sentence, _ in data)
        # for i,d in enumerate(data):
        #     print('{} {}'.format(i, len(d[0])))

        return cls(tok2id, max_length) # return a class instance 

    def save(self, path):
        # Make sure the directory exists.
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the tok2id map.
        with open(os.path.join(path, "features.pkl"), "w") as f:
            pickle.dump([self.tok2id, self.max_length], f)

    @classmethod
    def load(cls, path):
        # Make sure the directory exists.
        assert os.path.exists(path) and os.path.exists(os.path.join(path, "features.pkl"))
        # Save the tok2id map.
        with open(os.path.join(path, "features.pkl")) as f:
            tok2id, max_length = pickle.load(f)
        return cls(tok2id, max_length)

        