# _*_ coding:utf8 _*_

from __future__ import division

import logging
import numpy as np
from collections import Counter, OrderedDict

logger = logging.getLogger('ner_model')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def read_conll(fname):
    ret = []
    with open(fname, 'r') as f:
        current_toks = []
        current_labels = []
        for line in f:
            line = line.strip().split()
            if len(line) == 0:
                if len(current_toks) > 0:
                    assert len(current_toks) == len(current_labels)
                    ret.append((current_toks, current_labels))
                current_toks = []
                current_labels = []
            else:
                tok = line[0].decode('utf-8')
                label = line[-1]
                current_toks.append(tok)
                current_labels.append(label)
                # print('%s : %s' % (tok, label))

        if len(current_toks) > 0:
            assert len(current_toks) == len(current_labels)
            ret.append((current_toks, current_labels))
        # print(len(ret))
        # print(ret[0])
    return ret

def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    # words = [(char1, x),...,(charN,y)], x>y
    if max_words:
        words = cnt.most_common(max_words)
    else:
        words = cnt.most_common()
    # print(len(words))
    '''
    fw = [w[0] for w in words[:10]]
    for i in range(len(fw)):
        print('%s' % fw[i])
    assert False
    '''
    # enumerate index from 0
    return {word: i + offset for i, (word, _) in enumerate(words)}

def normalize(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def load_word_vector_mapping(embedding_path):
    ret = OrderedDict() # bi-direction link table
    with open(embedding_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            word = line[0].decode('utf-8')
            ret[word] = np.array(list(map(np.float32, line[1:])))
            # print(ret[word].shape)

    return ret

def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size): # auto-dropout unfit batch_size item
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data else minibatch(data, minibatch_indices)

def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]
