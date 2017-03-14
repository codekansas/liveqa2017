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
                 num_latent=512,
                 logdir=None,
                 only_cpu=False):
        """Constructs a new question generator.

        Args:
            sess: the current TensorFlow session.
            input_len: int, the input length.
            output_len: int, the output length.
            num_classes: int, the number of classes in the model.
            num_latent: int, the number of latent dimensions.
            logdir: str, the training directory (otherwise a new temporary one
                is created).
            only_cpu: bool, if set, don't use the GPU.
        """

        self._input_len = input_len
        self._output_len = output_len
        self._num_classes = num_classes
        self._only_cpu = only_cpu
        self._num_latent = num_latent

        input_shape = (None, input_len)
        output_shape = (None, None)
        self._input_pl = tf.placeholder(tf.int32, input_shape)
        self._target_pl = tf.placeholder(tf.int32, output_shape)
        self._target_len_pl = tf.placeholder(tf.int32, (None,))
        self._latent_pl = tf.placeholder(tf.float32, (None, num_latent))

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

    def build(self, num_emb=100, num_rnn=2, rnn_size=512, conv_size=512):
        """Builds the model, initializing weights.

        TODO: Docstring.
        """

        if self._built:
            return

        x = self.input_pl  # Represents the output after each layer.

        # Converts input to one-hot encoding.
        emb = tf.eye(self.num_classes)
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

        def _add_dense(x, layer_num, nb_outputs,
                       init='glorot', activation='tanh'):
            in_channels = x.get_shape()[1].value
            w = self.get_weight('w_%d' % layer_num,
                                (in_channels, nb_outputs), init=init)
            b = self.get_weight('b_%d' % layer_num,
                                (nb_outputs,), init='zero')

            try:
                activation = getattr(tf, activation)
            except Exception:
                activation = getattr(tf.nn, activation)

            return activation(tf.matmul(x, w) + b)

        with tf.variable_scope('encoder'):
            x = _add_conv(x, 1, conv_size, 5, 1)
            x = _add_conv(x, 2, conv_size, 5, 3)
            x = _add_conv(x, 3, conv_size, 5, 3)
            x = _add_conv(x, 4, conv_size, 5, 3)
            x = tf.squeeze(x, axis=2)

            cells = [tf.contrib.rnn.GRUCell(rnn_size) for _ in range(num_rnn)]
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, self.num_latent)
            cell = tf.contrib.rnn.FusedRNNCellAdaptor(cell, use_dynamic_rnn=True)

            x, _ = cell(tf.transpose(x, (1, 0, 2)), dtype=tf.float32)
            x = x[-1]  # Get the last timestep.

            # Latent vector inspired by variational autoencoder.
            mu = _add_dense(x, 1, self.num_latent,
                            activation='identity', init='zero')
            sigma = _add_dense(x, 2, self.num_latent,
                               activation='identity', init='zero')
            eps = tf.random_normal(tf.shape(sigma), 0, 1,
                                   dtype=tf.float32,
                                   name='eps')
            x = mu + tf.sqrt(tf.exp(sigma)) * eps

        # Add variational penalty.
        latent_pen = -0.5 * tf.reduce_sum(1 + sigma
                                          - tf.square(mu)
                                          - tf.exp(sigma))
        tf.summary.scalar('loss/latent', latent_pen)

        # Matrix multiplies to get the RNN hidden states.
        x = tuple([_add_dense(x, i + 10, rnn_size) for i in range(num_rnn)])

        # Saves the encoder state for later.
        enc_state = x

        # Builds the RNN decoder training function.
        cells = [tf.contrib.rnn.GRUCell(rnn_size) for _ in range(num_rnn)]
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, self.num_classes)
        train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(x)

        # Appends the start token to the first position of the target_pl.
        start_idx = tf.ones((tf.shape(self.target_pl)[0], 1), dtype=tf.int32)
        start_idx *= yahoo.START_IDX
        seq_vals = tf.concat([start_idx, self.target_pl], axis=1)
        seq_vals = tf.nn.embedding_lookup(emb, seq_vals)

        x, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            cell=cell,
            decoder_fn=train_decoder_fn,
            inputs=seq_vals,
            sequence_length=self.target_len_pl)

        weights = tf.sequence_mask(self.target_len_pl, dtype=tf.float32)

        sequence_loss = tf.contrib.seq2seq.sequence_loss(
            x, self.target_pl, weights)
        tf.summary.scalar('loss/sequence', sequence_loss)

        self.loss = sequence_loss + latent_pen
        tf.summary.scalar('loss/total', self.loss)

        # Builds the training op.
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.train_op = optimizer.minimize(self.loss)

        # Builds the RNN decoder inference function.
        output_fn = lambda x: tf.identity(x)
        infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
            output_fn=output_fn,
            encoder_state=enc_state,
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

        # "Dreams" a question (reusing the existing dense layers).
        generator_state = tuple([_add_dense(self.latent_pl, i + 10, rnn_size)
                                 for i in range(num_rnn)])
        generate_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
            output_fn=output_fn,
            encoder_state=generator_state,
            embeddings=emb,
            start_of_sequence_id=yahoo.START_IDX,
            end_of_sequence_id=yahoo.END_IDX,
            maximum_length=self.output_len,
            num_decoder_symbols=self.num_classes)
        self.generated_question, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            cell=cell,
            decoder_fn=generate_decoder_fn)

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
    def train(self, a_sample, q_sample, q_len):
        """Trains the model on the provided input data.

        Args:
            a_sample: Numpy array with shape (batch_size, num_timesteps),
                ints representing the encoded answer.
            q_sample: Numpy array with shape (batch_size, num_timesteps),
                ints representing the encoded question.
            q_len: Numpy array with shape (batch_size), ints representing
                the length of the questions.

        Returns:
            loss: float, the total loss of the model.
        """

        feed_dict = {
            self.input_pl: a_sample,
            self.target_pl: q_sample[:, :np.max(q_len)],
            self.target_len_pl: q_len,
        }

        _, current_time, summary, loss = self._sess.run(
            [self.train_op, self.time_op, self.summary_op, self.loss],
            feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, current_time)

        return loss

    @check_built
    def sample(self, x):
        """Samples from the model.

        Args:
            x: Numpy array with shape (batch_size, input_len), ints
                representing the encoded question.

        Returns:
            y: Numpy array with shape (batch_size, output_len, num_classes),
                probability distribution over characters.
        """

        feed_dict = {
            self.input_pl: x,
        }

        y, = self._sess.run([self.inferred_question], feed_dict=feed_dict)

        return y

    @check_built
    def generate(self, x=None, nb_samples=None):
        """Generates a "dream" from the model.

        Args:
            x: Numpy array with shape (batch_size, num_latent), the starting
                latent representation.
            nb_samples: int, the number of samples to generate (if `x` is None).

        Returns:
            y: Numpy array with shape (batch_size, output_len, num_classes),
                probability distribution over characters.
        """

        if x is None and nb_samples is None:
            raise ValueError('Either the latent vector `x` or the number of '
                             'samples to produce `nb_samples` must be '
                             'specified.')

        if x is None:
            x = np.random.normal(size=(nb_samples, self.num_latent))

        feed_dict = {
            self.latent_pl: x,
        }

        y, = self._sess.run([self.generated_question], feed_dict=feed_dict)

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
    def target_pl(self):
        return self._target_pl

    @property
    def target_len_pl(self):
        return self._target_len_pl

    @property
    def latent_pl(self):
        return self._latent_pl

    @property
    def num_latent(self):
        return self._num_latent

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
