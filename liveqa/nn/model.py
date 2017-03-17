"""Components for the question generator network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest

import six

if __name__ == '__main__':
    raise RuntimeError('Cannot run this file directly.')

import sys
import tempfile

from .. import yahoo

import tensorflow as tf
import numpy as np

# linear function used by RNNs.
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear


def check_built(f):
    """Function decorator that makes sure the model is built before calling."""

    def wrapper(self, *args, **kwargs):
        if not self._built:
            self.build()

        return f(self, *args, **kwargs)

    return wrapper


class ActivationWrapper(tf.contrib.rnn.RNNCell):
    """Operator adding an activation to the output of an RNN."""

    def __init__(self, cell, function, reuse=None):
        if not isinstance(cell, tf.contrib.rnn.RNNCell):
            raise TypeError('The parameter cell is not an RNNCell.')

        if isinstance(function, six.string_types):
            try:
                function = getattr(tf, function)
            except AttributeError:
                try:
                    function = getattr(tf.nn, function)
                except AttributeError:
                    raise ValueError('The desired function "%s" was '
                                     'not found.' % function)

        self._cell = cell
        self._function = function

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        output, res_state = self._cell(inputs, state)
        return self._function(output), res_state


class AttentionCellWrapper2D(tf.contrib.rnn.RNNCell):
    """Basic attention cell wrapper, using 2D attention input.

    This isn't really an "attention" mechanism, more like a "peak" mechanism.
    There were a number of issues
    """

    def __init__(self, cell, attn_vec):
        if not isinstance(cell, tf.contrib.rnn.RNNCell):
            raise TypeError('The parameter cell is not an RNNCell.')

        self._cell = cell
        self._attn_vec = attn_vec

    def __call__(self, inputs, state, scope=None):
        """GRU with attention."""

        with tf.variable_scope(scope or 'attention_cell_wrapper'):
            output, _ = self._cell(inputs, state)
            output = _linear([output, self._attn_vec],
                             self.output_size, bias=True)
            output = tf.tanh(output)

        return output, output

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.state_size


class QuestionGenerator(object):
    """Class implementing the question generator."""

    def __init__(self,
                 sess,
                 input_len,
                 output_len,
                 num_classes,
                 logdir=None,
                 only_cpu=False):
        """Constructs a new question generator.

        Args:
            sess: the current TensorFlow session.
            input_len: int, the input length.
            output_len: int, the output length.
            num_classes: int, the number of classes in the model.
            logdir: str, the training directory (otherwise a new temporary one
                is created).
            only_cpu: bool, if set, don't use the GPU.
        """

        self._input_len = input_len
        self._output_len = output_len
        self._num_classes = num_classes
        self._only_cpu = only_cpu

        input_shape = (None, None)
        output_shape = (None, None)
        self._input_pl = tf.placeholder(tf.int32, input_shape)
        self._target_pl = tf.placeholder(tf.int32, output_shape)
        self._input_len_pl = tf.placeholder(tf.int32, (None,))
        self._target_len_pl = tf.placeholder(tf.int32, (None,))

        self._built = False
        self._weights = []

        self._saver = None
        self._sess = sess
        self.logdir = logdir

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
            weight_val = tf.zeros(shape)
        else:
            raise ValueError('Invalid init: "%s"' % init)

        device = device.lower()
        if device == 'gpu':
            on_gpu = True
        elif device == 'cpu':
            on_gpu = False
        else:
            raise ValueError('Invalid device: "%s"' % device)

        if self._only_cpu:
            on_gpu = False

        with tf.device('/gpu:0' if on_gpu else '/cpu:0'):
            weight = tf.Variable(weight_val, name=name)
        self._weights.append(weight)

        return weight

    def build(self, num_rnn=2, rnn_size=128):
        """Builds the model, initializing weights.

        TODO: Docstring.
        """

        if self._built:
            return

        x = self.input_pl  # Represents the output after each layer.

        # Converts input to one-hot encoding.
        emb = self.get_weight('emb', (self.num_classes, 200),
                              init='glorot')
        x = tf.nn.embedding_lookup(emb, x)

        with tf.variable_scope('encoder'):
            cell = tf.contrib.rnn.GRUCell(rnn_size)
            cell = tf.contrib.rnn.FusedRNNCellAdaptor(cell,
                                                      use_dynamic_rnn=True)

            with tf.variable_scope('forward'):
                attn_vec_fw, enc_states_fw = cell(
                    tf.transpose(x, (1, 0, 2)),
                    dtype=tf.float32,
                    sequence_length=self.input_len_pl)
                # attn_vec_fw = tf.reduce_max(attn_vec_fw, axis=0)

            with tf.variable_scope('backward'):
                cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(cell)
                attn_vec_bw, enc_states_bw = cell_bw(
                    tf.transpose(x, (1, 0, 2)),
                    dtype=tf.float32,
                    sequence_length=self.input_len_pl)
                # attn_vec_bw = tf.reduce_max(attn_vec_bw, axis=0)

            attn_vec = tf.concat([attn_vec_fw, attn_vec_bw], axis=-1)
            enc_states = tf.concat([enc_states_fw, enc_states_bw], axis=-1)

        attn_vec = tf.transpose(attn_vec, (1, 0, 2))

        # Builds the attention component.
        k, v, score_fn, construct_fn = tf.contrib.seq2seq.prepare_attention(
            attn_vec, 'luong', rnn_size * 2)

        train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(
            enc_states, k, v, score_fn, construct_fn)

        # Appends the start token to the first position of the target_pl.
        start_idx = tf.ones((tf.shape(self.target_pl)[0], 1), dtype=tf.int32)
        start_idx *= yahoo.START_IDX
        seq_vals = tf.concat([start_idx, self.target_pl], axis=1)
        seq_vals = tf.nn.embedding_lookup(emb, seq_vals)

        # Builds the RNN decoder training function.
        cell = tf.contrib.rnn.GRUCell(rnn_size * 2)

        x, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            cell=cell,
            decoder_fn=train_decoder_fn,
            inputs=seq_vals,
            sequence_length=self.target_len_pl)

        print('rnn size:', rnn_size, 'num_classes:', self.num_classes)

        output_W = self.get_weight('out_W', (rnn_size * 2, self.num_classes))
        output_b = self.get_weight('out_b', (self.num_classes,), init='zero')

        # Applies the output function time-wise.
        x = tf.einsum('ijk,kl->ijl', x, output_W)
        x = tf.nn.bias_add(x, output_b)

        # Function for decoding layer on.
        output_fn = lambda x: tf.matmul(x, output_W) + output_b

        # Masks all the values beyond the answer.
        weights = tf.sequence_mask(self.target_len_pl, dtype=tf.float32)

        with tf.variable_scope('reg_loss'):
            reg_loss = 0
            for w in tf.trainable_variables():
                reg_loss += tf.nn.l2_loss(w) * 1e-4

        seq_loss = tf.contrib.seq2seq.sequence_loss(
            x, self.target_pl, weights)
        tf.summary.scalar('loss/seq', seq_loss)
        tf.summary.scalar('loss/reg', reg_loss)

        self.loss = seq_loss + reg_loss
        tf.summary.scalar('loss/total', self.loss)

        # Builds the training op.
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        gvs = optimizer.compute_gradients(self.loss)
        capped = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped)

        # Reuse the RNN from earlier.
        tf.get_variable_scope().reuse_variables()

        # Builds the attention decoder.
        infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(
            output_fn=output_fn,
            encoder_state=enc_states,
            attention_keys=k,
            attention_values=v,
            attention_score_fn=score_fn,
            attention_construct_fn=construct_fn,
            embeddings=emb,
            start_of_sequence_id=yahoo.START_IDX,
            end_of_sequence_id=yahoo.END_IDX,
            maximum_length=self.output_len,
            num_decoder_symbols=self.num_classes)

        self.inferred_question, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            cell=cell,
            decoder_fn=infer_decoder_fn)

        # Creates the log directory.
        if self.logdir is None:
            self.logdir = tempfile.mkdtemp()
            sys.stdout.write('logdir: "%s"\n' % self.logdir)
        summary_writer = tf.summary.FileWriter(self.logdir, self._sess.graph)
        self.summary_writer = summary_writer
        self.summary_op = tf.summary.merge_all()

        # Creates the time variable on the CPU.
        with tf.device('/cpu:0'):
            time = tf.Variable(0, dtype=tf.int64, name='time')
            self.time_op = time.assign(time + 1)

        self._saver = tf.train.Saver()
        self._sess.run(tf.global_variables_initializer())
        self._built = True

        return self

    @check_built
    def train(self, qsamples, asamples, qlens, alens):
        """Trains the model on the provided input data.

        Args:
            qsamples: Numpy array with shape (batch_size, num_timesteps),
                ints representing the encoded question.
            asamples: Numpy array with shape (batch_size, num_timesteps),
                ints representing the encoded answer.
            qlens: Numpy array with shape (batch_size), ints representing
                the length of the questions.
            alens: Numpy array with shape (batch_size), ints representing
                the length of the answers.

        Returns:
            loss: float, the total loss of the model.
        """

        feed_dict = {
            self.input_pl: asamples[:, :np.max(alens)],
            self.input_len_pl: alens,
            self.target_pl: qsamples[:, :np.max(qlens)],
            self.target_len_pl: qlens,
        }

        _, current_time, summary, loss = self._sess.run(
            [self.train_op, self.time_op, self.summary_op, self.loss],
            feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, current_time)

        return loss

    @check_built
    def sample(self, x, xlen):
        """Samples from the model.

        Args:
            x: Numpy array with shape (batch_size, input_len), ints
                representing the encoded answer.
            xlens: Numpy array with shape (batch_size), ints representing
                the length of the answers.

        Returns:
            y: Numpy array with shape (batch_size, output_len, num_classes),
                probability distribution over characters in questions.
        """

        feed_dict = {
            self.input_pl: x,
            self.input_len_pl: xlen,
        }

        y = self._sess.run(self.inferred_question, feed_dict=feed_dict)

        return y

    @check_built
    def load(self, ignore_missing=False):
        """Loads the model from the logdir.

        Args:
            ignore_missing: bool, if set, ignore when no save_dir exists,
                otherwise raises an error.
        """

        ckpt = tf.train.get_checkpoint_state(self.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
        elif ignore_missing:
            return
        elif not ckpt:
            raise ValueError('No checkpoint found: "%s"' % self.logdir)
        else:
            raise ValueError('Checkpoint found, but no model checkpoint path '
                             'in "%s"' % self.logdir)

    @check_built
    def save(self):
        """Saves the model to the logdir."""

        self._saver.save(self._sess, self.logdir + 'model.ckpt')

    @property
    def input_pl(self):
        return self._input_pl

    @property
    def input_len_pl(self):
        return self._input_len_pl

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

    @check_built
    @property
    def weights(self):
        return self._weights
