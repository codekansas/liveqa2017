"""Components for the question generator network."""

from __future__ import division

if __name__ == '__main__':
    raise RuntimeError('Cannot run this file directly.')

import sys
import tempfile

from .. import yahoo

import tensorflow as tf
import numpy as np


def check_built(f):
    """Function decorator that makes sure the model is built before calling."""

    def wrapper(self, *args, **kwargs):
        if not self._built:
            self.build()

        return f(self, *args, **kwargs)

    return wrapper


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

    def build(self, num_rnn=2, rnn_size=256):
        """Builds the model, initializing weights.

        TODO: Docstring.
        """

        if self._built:
            return

        x = self.input_pl  # Represents the output after each layer.

        # Converts input to one-hot encoding.
        emb = tf.eye(self.num_classes)
        x = tf.nn.embedding_lookup(emb, x)
        x = tf.transpose(x, (1, 0, 2))  # Put into time major.

        with tf.variable_scope('encoder'):
            cells = [tf.contrib.rnn.GRUCell(rnn_size) for _ in range(num_rnn)]
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell,
                                                          self.num_classes)
            cell = tf.contrib.rnn.FusedRNNCellAdaptor(cell,
                                                      use_dynamic_rnn=True)
            _, enc_states = cell(x,
                                 dtype=tf.float32,
                                 sequence_length=self.input_len_pl)

        train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(
            encoder_state=enc_states)

        # Appends the start token to the first position of the target_pl.
        start_idx = tf.ones((tf.shape(self.target_pl)[0], 1), dtype=tf.int32)
        start_idx *= yahoo.START_IDX
        seq_vals = tf.concat([start_idx, self.target_pl], axis=1)
        seq_vals = tf.nn.embedding_lookup(emb, seq_vals)

        # Builds the RNN decoder training function.
        cells = [tf.contrib.rnn.GRUCell(rnn_size) for _ in range(num_rnn)]
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, self.num_classes)

        x, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            cell=cell,
            decoder_fn=train_decoder_fn,
            inputs=seq_vals,
            sequence_length=self.target_len_pl)

        # Masks all the values beyond the answer.
        weights = tf.sequence_mask(self.target_len_pl, dtype=tf.float32)

        self.loss = tf.contrib.seq2seq.sequence_loss(
            x, self.target_pl, weights)
        tf.summary.scalar('loss', self.loss)

        # Builds the training op.
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.train_op = optimizer.minimize(self.loss)

        # Builds the RNN decoder inference function.
        infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
            output_fn=tf.identity,
            encoder_state=enc_states,
            embeddings=emb,
            start_of_sequence_id=yahoo.START_IDX,
            end_of_sequence_id=yahoo.END_IDX,
            maximum_length=self.output_len,
            num_decoder_symbols=self.num_classes)

        # Reuse the RNN from earlier.
        tf.get_variable_scope().reuse_variables()
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
