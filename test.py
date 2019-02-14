#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops

def get_chunks_cws(seq):
    chunks = []
    labels = ['B', 'M', 'E', 'S']
    chunk_type, chunk_start = None, None
    for i, tag in enumerate(seq):
        if tag == labels.index('B'):
            chunk_start = i
            chunk_type = "Mu"
        elif tag == labels.index('E'):
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
        elif tag == labels.index('S'):
            chunk_type = "Si"
            chunk = (chunk_type, i, i)
            chunks.append(chunk)
        else:
            pass
    return chunks 

def viterbi_decode(score, transition_params):
  """Decode the highest scoring sequence of tags outside of TensorFlow.
  This should only be used at test time.
  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indices.
    viterbi_score: A float containing the score for the Viterbi sequence.
  """
  trellis = np.zeros_like(score)
  backpointers = np.zeros_like(score, dtype=np.int32)
  trellis[0] = score[0]

  for t in range(1, score.shape[0]):
      v = np.expand_dims(trellis[t - 1], 1) + transition_params
      trellis[t] = score[t] + np.max(v, 0)
      backpointers[t] = np.argmax(v, 0)

  viterbi = [np.argmax(trellis[-1])]
  for bp in reversed(backpointers[1:]):
      viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()

  viterbi_score = np.max(trellis[-1])
  return viterbi, viterbi_score

def test_viterbi_decode():
    score = np.arange(18).reshape(6,3)
    print(score)
    #np.random.seed(1234)
    transition_params = np.random.randn(3,3)
    print(transition_params)
    viterbi, viterbi_score = viterbi_decode(score, transition_params)
    print(viterbi)
    print(viterbi_score)

def test_alpha_value():
    with tf.Graph().as_default():
        tr_matrix = tf.constant([[0.1, 0.6, 0.3],[0.32, 0.18, 0.5],[0.4, 0.23, 0.27]], dtype=tf.float32)
        alpha0 = tf.constant([[0.2, 0.4, 1.1]], dtype=tf.float32)
        input1 = tf.constant([[1.32, 0.25, 0.56]], dtype=tf.float32)
        etr_matrix = tf.expand_dims(tr_matrix, 0)
        print(tr_matrix.shape)
        print(etr_matrix.shape)
        state = tf.expand_dims(alpha0, 2)
        print(state.shape)
        tr_scores = state + etr_matrix
        logsum = tf.reduce_logsumexp(tr_scores, [1])
        alpha1 = input1 + tf.reduce_logsumexp(tr_scores, [1])
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            print(sess.run(input1))
            print(sess.run(etr_matrix))
            print(sess.run(state))
            print(sess.run(tr_scores))
            print(sess.run(logsum))
            print(sess.run(alpha1))
            

def test_tesor_mask():    
    with tf.Graph().as_default():
        inputs = np.array([[[1.0, 2.0, 3.0],[4.0, 5.0, 6.0],[7.0, 8.0, 9.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], 
                           [[1.0, 3.0, 4.0],[2.0, 5.0, 8.0],[8.0, 7.0, 6.0],[5.0, 4.0, 2.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]])
        mask = [[True, True, True, False, False, False],
                [True, True, True, True,  False, False]]
        tensor = ops.convert_to_tensor(inputs, name="tensor")
        mask = ops.convert_to_tensor(mask, name="mask")
        indices = tf.where(mask)
        # real_inputs = tf.boolean_mask(inputs, mask)
        # reverse_inputs = tf.reverse(tensor, [1])
        # reverse_mask = tf.reverse(mask, [1])
        new_inputs = tf.gather_nd(inputs, indices)
        input2 = tf.slice(inputs, [0 ,0, 0], [-1, 1, -1])
        first_input = tf.squeeze(input2, [1])
        log_norm = tf.reduce_logsumexp(first_input, [1])

        batch_size = tf.shape(inputs)[0]
        max_seq_len = tf.shape(inputs)[1]
        num_tags = tf.shape(inputs)[2]
        example_inds = tf.reshape(tf.range(batch_size), [-1,1])
        tag_indices = np.array([[1],[0]], dtype=np.int32)
        tag_indices = ops.convert_to_tensor(tag_indices)
        sequence_scores = tf.gather_nd(tf.squeeze(input2, [1]), tf.concat([example_inds, tag_indices], axis=1))

        testout1 = tf.range(batch_size) * max_seq_len * num_tags
        offsets = tf.expand_dims(tf.range(batch_size) * max_seq_len * num_tags, 1)
        offsets += tf.expand_dims(tf.range(max_seq_len) * num_tags, 0)
        indices3 = tf.constant([[0, 1, 1, 2, 0, 1], [1, 2, 1, 0, 1, 2]], dtype=tf.int32)
        flattend_tag_indices = tf.reshape(offsets + indices3, [-1])

        tp = tf.reshape(tf.range(16, dtype=tf.float32),[4,4])
        ftp = tf.reshape(tp, [-1])

        etp = tf.expand_dims(tp, 0)
        state = tf.expand_dims(tp, 2)
        nstate = state + etp
        lognstate = tf.reduce_logsumexp(nstate, [1])
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            print(inputs.shape)
            #print(sess.run(testout1))
            #print(sess.run(offsets))
            #print(sess.run(flattend_tag_indices))
            print(sess.run(tp))
            #print(sess.run(ftp))
            print(state.shape)
            print(sess.run(state))
            print(nstate.shape)
            print(sess.run(nstate))
            print(lognstate.shape)
            print(sess.run(lognstate))



if __name__ == '__main__':
    #test_tesor_mask()
    #test_viterbi_decode()
    test_alpha_value()