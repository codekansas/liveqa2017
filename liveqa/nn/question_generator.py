#!/usr/bin/env python
"""
Components for the question generator network.

Reference code for serving models:
https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py
"""

from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile

import tensorflow as tryef

import tensorflow as tf
import numpy as np

import yahoo

# Defines command line flags.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            'size of each training minibatch.')
tf.app.flags.DEFINE_integer('batches_per_epoch', 100,
                            'number of minibatches per training epoch')
tf.app.flags.DEFINE_integer('nb_epoch', 100,
                            'number of epochs to train the model for')
tf.app.flags.DEFINE_string('weights_save_path', '/tmp/qgen_weights.h5',
                           'where to save the model')
tf.app.flags.DEFINE_bool('rebuild_model', True,
                         'if set, reset the model weights')
FLAGS = tf.app.flags.FLAGS


class QuestionGenerator(object):
    """Class implementing the question generator.

    """

    def __init__(self,
                 input_len,
                 output_len,
                 num_classes):
        """Constructs a new question generator.

        Args:
            input_len: int, the input length.
            output_len: int, the output length.
            num_classes: int, the number of classes in the model.
        """

        self._input_len = input_len
        self._output_len = output_len
        self._num_classes = num_classes

        input_shape = (None, input_len)
        output_shape = (None, None)
        self._input_pl = tf.placeholder(tf.int32, input_shape)
        self._target_pl = tf.placeholder(tf.int32, output_shape)
        self._target_len_pl = tf.placeholder(tf.int32, (None,))

        self._built = False
        self._weights = []

    def get_weight(self, name, shape, init='glorot', device='gpu'):
        """Returns a new weight.

        Args:
            name: str, the name of the variable.
            shape: tuple of ints, the shape of the variable.
            init: str, the type of initialize to use.
            device: str, 'cpu' or 'gpu'.
        """

        init = init.lower()
        if init == 'normal':
            weight_val = tf.random_normal(shape, stddev=0.05)
        elif init == 'uniform':
            weight_val = tf.random_uniform(shape, maxval=0.05)
        elif init == 'glorot':
            stddev = np.sqrt(6. / sum(shape))
            weight_val = tf.random_normal(shape, stddev=stddev)
        elif init == 'eye':
            assert all(i == shape[0] for i in shape)
            weight_val = tf.eye(shape[0])
        elif init == 'zero':
            weight_val = tf.zeroes(shape)
        else:
            raise ValueError('Invalid init: "%s"' % init)

        device = device.lower()
        if device == 'gpu':
            on_gpu = True
        elif device == 'cpu':
            on_gpu = False
        else:
            raise ValueError('Invalid device: "%s"' % device)

        with tf.device('/gpu:0' if on_gpu else '/cpu:0'):
            weight = tf.Variable(weight_val, name=name)
        self._weights.append(weight)

        return weight

    def build(self, sess, num_emb=100, num_rnn=2):
        """Builds the model, initializing weights.

        TODO
        """

        if self._built:
            return

        x = self.input_pl  # Represents the output after each layer.

        # Converts input to one-hot encoding.
        emb = self.get_weight('emb', (self.num_classes, self.num_classes),
                              init='eye')
        x = tf.nn.embedding_lookup(emb, x)
        x = tf.expand_dims(x, axis=2)  # (batch_size, time, 1, num_emb)

        def _add_conv(x, layer_num, nb_filters, filt_length, pool_length):
            in_channels = x.get_shape()[3].value
            filt = self.get_weight('filter_%d' % layer_num,
                                   (filt_length, 1, in_channels, nb_filters))
            x = tf.nn.conv2d(x, filt, (1, 1, 1, 1), 'VALID')
            pool_shape = (1, pool_length, 1, 1)
            x = tf.nn.max_pool(x, pool_shape, pool_shape, 'VALID')
            return x

        x = _add_conv(x, 1, 64, 4, 2)
        x = _add_conv(x, 2, 64, 3, 2)
        x = _add_conv(x, 3, 64, 2, 2)
        x = _add_conv(x, 4, 64, 2, 2)

        # Convolutions are used to create the initial hidden state.
        x = ([_add_conv(x, i + 5, 128, 1, 1) for i in range(num_rnn - 1)]
             + [_add_conv(x, i + 5, self.num_classes, 1, 1)])
        x = [tf.reduce_max(i, axis=[1, 2]) for i in x]
        x = tuple(x)

        # Saves the encoder state for later.
        enc_state = x

        # Builds the RNN decoder training function.
        cells = ([tf.contrib.rnn.GRUCell(128) for _ in range(num_rnn - 1)]
                 + [tf.contrib.rnn.GRUCell(self.num_classes)])
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(x)
        seq_vals = tf.one_hot(self.target_pl, self.num_classes)

        x, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            cell=cell,
            decoder_fn=train_decoder_fn,
            inputs=seq_vals,
            sequence_length=self.target_len_pl)

        weights = tf.sequence_mask(self.target_len_pl, dtype=tf.float32)

        self.loss = tf.contrib.seq2seq.sequence_loss(
            x, self.target_pl, weights)

        # Builds the training op.
        optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
        self.train_op = optimizer.minimize(self.loss)

        # Builds the RNN decoder inference function.
        output_fn = lambda x: tf.contrib.layers.linear(x, self.num_classes)
        infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
            output_fn=output_fn,
            encoder_state=enc_state,
            embeddings=emb,
            start_of_sequence_id=2,
            end_of_sequence_id=3,
            maximum_length=self.output_len,
            num_decoder_symbols=self.num_classes)

        # Reuse the RNN from earlier.
        tf.get_variable_scope().reuse_variables()
        self.inferred_question, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            cell=cell,
            decoder_fn=infer_decoder_fn)

        # Creates the log directory.
        logdir = tempfile.mkdtemp()
        sys.stdout.write('logdir: "%s"\n' % logdir)
        self.summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        self.summary_op = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())

        self._built = True

    def train(self, sess, a_sample, q_sample, q_len):
        """Trains the model on the provided input data."""

        if not self._built:
            raise ValueError('Must build the model first.')

        feed_dict = {
            self.input_pl: a_sample,
            self.target_pl: q_sample[:, :np.max(q_len)],
            self.target_len_pl: q_len,
        }

        _ = sess.run(
            [self.train_op],
            feed_dict=feed_dict)


    def sample(self, x):
        """Samples from the model.

        Args:
            x: Numpy array with shape (batch_size, input_len), ints
                representing the encoded question.

        Returns:
            y: Numpy array with shape (batch_size, output_len), ints
                representing a question produced by the model.
        """

        if not self._built:
            raise ValueError('Must build the model first.')

        raise NotImplementedError()

    @property
    def input_pl(self):
        return self._input_pl

    @property
    def target_pl(self):
        return self._target_pl

    @property
    def target_len_pl(self):
        return self._target_len_pl

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def output_len(self):
        return self._output_len

    @property
    def input_len(self):
        return self._input_len

    @property
    def weights(self):
        return self._weights


def main(_):

    # Constants.
    BATCH_SIZE = FLAGS.batch_size
    BATCHES_PER_EPOCH = FLAGS.batches_per_epoch
    NB_EPOCH = FLAGS.nb_epoch
    WEIGHTS_SAVE_PATH = FLAGS.weights_save_path
    REBUILD_MODEL = FLAGS.rebuild_model

    # Interactive session to avoid unpleasant parts.
    sess = tf.InteractiveSession()

    model = QuestionGenerator(yahoo.ANSWER_MAXLEN,
                              yahoo.QUESTION_TITLE_MAXLEN,
                              yahoo.NUM_TOKENS)
    model.build(sess)

    if not REBUILD_MODEL and os.path.exists(WEIGHTS_SAVE_PATH):
        sys.stdout.write('Loading model weights: "%s"\n' % WEIGHTS_SAVE_PATH)
        sys.stdout.flush()
        model.load_weights(MODEL_SAVE_PATH)
    elif not REBUILD_MODEL:
        sys.stdout.write('No weights found: "%s"\n' % WEIGHTS_SAVE_PATH)
        sys.stdout.flush()

    raw_input('Press enter to begin training: ')

    # Gets the data iterator.
    data_iter = yahoo.iterate_answer_to_question(BATCH_SIZE)

    # Builds the model sampler checkpoint.
    a_sample, q_sample, _ = data_iter.next()

    for epoch_idx in xrange(1, NB_EPOCH + 1):
        for batch_idx in xrange(1, BATCHES_PER_EPOCH + 1):
            a_sample, q_sample, q_len = data_iter.next()

            model.train(sess, a_sample, q_sample, q_len)

            sys.stdout.write('epoch %d: %d / %d\r'
                             % (epoch_idx, batch_idx, BATCHES_PER_EPOCH))
            sys.stdout.flush()
        sys.stdout.write('\n')


if __name__ == '__main__':
    tf.app.run()
