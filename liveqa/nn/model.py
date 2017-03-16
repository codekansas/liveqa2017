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
    """Basic attention cell wrapper, using 2D attention input."""

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


def noisy_decoder_fn_inferrence(output_fn, encoder_state, embeddings,
                                start_of_sequence_id, end_of_sequence_id,
                                maximum_length, num_decoder_symbols,
                                temperature, dtype=dtypes.int32, name=None):
    """ Simple decoder function for a sequence-to-sequence model used in the
    `dynamic_rnn_decoder`.

    The `simple_decoder_fn_inference` is a simple inference function for a
    sequence-to-sequence model. It should be used when `dynamic_rnn_decoder` is
    in the inference mode.

    The `simple_decoder_fn_inference` is called with a set of the user arguments
    and returns the `decoder_fn`, which can be passed to the
    `dynamic_rnn_decoder`, such that

    ```
    dynamic_fn_inference = simple_decoder_fn_inference(...)
    outputs_inference, state_inference = dynamic_rnn_decoder(
        decoder_fn=dynamic_fn_inference, ...)
    ```

    Further usage can be found in the `kernel_tests/seq2seq_test.py`.

    Args:
        output_fn: An output function to project your `cell_output` onto class
        logits.
        encoder_state: The encoded state to initialize the
        `dynamic_rnn_decoder`.
        embeddings: The embeddings matrix used for the decoder sized
        `[num_decoder_symbols, embedding_size]`.
        start_of_sequence_id: The start of sequence ID in the decoder
        embeddings.
        end_of_sequence_id: The end of sequence ID in the decoder embeddings.
        maximum_length: The maximum allowed of time steps to decode.
        num_decoder_symbols: The number of classes to decode at each time step.
        temperature: float or 0D Tensor, the temperature of the noise.
        dtype: (default: `dtypes.int32`) The default data type to use when
        handling integer objects.
        name: (default: `None`) NameScope for the decoder function;
            defaults to "simple_decoder_fn_inference"

    Returns:
        A decoder function with the required interface of `dynamic_rnn_decoder`
        intended for inference.
    """
    with ops.name_scope(name, "simple_decoder_fn_inference",
                        [output_fn, encoder_state, embeddings,
                         start_of_sequence_id, end_of_sequence_id,
                         maximum_length, num_decoder_symbols, dtype]):
        start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
        end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
        maximum_length = ops.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
        encoder_info = nest.flatten(encoder_state)[0]
        batch_size = encoder_info.get_shape()[0].value
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = array_ops.shape(encoder_info)[0]

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        """ Decoder function used in the `dynamic_rnn_decoder` with the purpose
        of inference.

        The main difference between this decoder function and the `decoder_fn`
        in `simple_decoder_fn_train` is how `next_cell_input` is calculated. In
        this decoder function we calculate the next input by applying an argmax
        across the feature dimension of the output from the decoder. This is a
        greedy-search approach. (Bahdanau et al., 2014) & (Sutskever et al.,
        2014) use beam-search instead.

        Args:
            time: positive integer constant reflecting the current timestep.
            cell_state: state of RNNCell.
            cell_input: input provided by `dynamic_rnn_decoder`.
            cell_output: output of RNNCell.
            context_state: context state provided by `dynamic_rnn_decoder`.

        Returns:
            A tuple (done, next state, next input, emit output, next context
            state) where:

            done: A boolean vector to indicate which sentences has reached a
            `end_of_sequence_id`. This is used for early stopping by the
            `dynamic_rnn_decoder`. When `time>=maximum_length` a boolean vector
            with all elements as `true` is returned.

            next state: `cell_state`, this decoder function does not modify the
            given state.

            next input: The embedding from argmax of the `cell_output` is used
            as `next_input`.

            emit output: If `output_fn is None` the supplied `cell_output` is
            returned, else the `output_fn` is used to update the `cell_output`
            before calculating `next_input` and returning `cell_output`.

            next context state: `context_state`, this decoder function does not
            modify the given context state. The context state could be modified
            when applying e.g. beam search.
    """
        with ops.name_scope(name, "simple_decoder_fn_inference",
                            [time, cell_state, cell_input, cell_output,
                             context_state, temperature]):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" %
                                 cell_input)
            if cell_output is None:
                next_input_id = array_ops.ones([batch_size,], dtype=dtype) * (
                        start_of_sequence_id)
                done = array_ops.zeros([batch_size,], dtype=dtypes.bool)
                cell_state = encoder_state
                cell_output = array_ops.zeros([num_decoder_symbols],
                                              dtype=dtypes.float32)
            else:
                cell_output = output_fn(cell_output)
                log_div = math_ops.log(nn_ops.softmax(cell_output))
                exp_log_div = math_ops.exp(log_div / temperature)
                renormalized = nn_ops.softmax(exp_log_div)
                choices = random_ops.multinomial(renormalized, 1)
                choices = array_ops.reshape(choices, (batch_size,))
                next_input_id = math_ops.cast(choices, dtype=dtype)
                done = math_ops.equal(next_input_id, end_of_sequence_id)
            next_input = array_ops.gather(embeddings, next_input_id)
            done = control_flow_ops.cond(math_ops.greater(time, maximum_length),
                lambda: array_ops.ones([batch_size,], dtype=dtypes.bool),
                lambda: done)
            return (done, cell_state, next_input, cell_output, context_state)

    return decoder_fn


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
        self._temp_pl = tf.placeholder(tf.float32, ())

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
        emb = self.get_weight('emb', (self.num_classes, self.num_classes),
                              init='eye')
        x = tf.nn.embedding_lookup(emb, x)

        def _add_conv(x, layer_num, nb_filters, filt_length, pool_length):
            in_channels = x.get_shape()[3].value
            filt = self.get_weight('filter_%d' % layer_num,
                                   (filt_length, 1, in_channels, nb_filters))
            x = tf.nn.conv2d(x, filt, (1, 1, 1, 1), 'SAME')
            pool_shape = (1, pool_length, 1, 1)
            x = tf.nn.max_pool(x, pool_shape, pool_shape, 'SAME')
            return x

        with tf.variable_scope('encoder'):
            # x = tf.expand_dims(x, axis=2)
            # x = _add_conv(x, 1, 64, 5, 2)
            # x = _add_conv(x, 1, 64, 5, 2)
            # x = _add_conv(x, 1, 64, 5, 2)
            # x = tf.squeeze(x, axis=2)

            cells = [tf.contrib.rnn.GRUCell(rnn_size) for _ in range(num_rnn)]
            cells = [tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=0.7) for cell in cells]
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            cell = tf.contrib.rnn.FusedRNNCellAdaptor(cell,
                                                      use_dynamic_rnn=True)

            with tf.variable_scope('forward'):
                attn_vec_fw, enc_states_fw = cell(
                    tf.transpose(x, (1, 0, 2)),
                    dtype=tf.float32,
                    sequence_length=self.input_len_pl)
                attn_vec_fw = tf.reduce_max(attn_vec_fw, axis=0)

            with tf.variable_scope('backward'):
                cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(cell)
                attn_vec_bw, enc_states_bw = cell_bw(
                    tf.transpose(x, (1, 0, 2)),
                    dtype=tf.float32,
                    sequence_length=self.input_len_pl)
                attn_vec_bw = tf.reduce_max(attn_vec_bw, axis=0)

            attn_vec = tf.concat([attn_vec_fw, attn_vec_bw], axis=-1)
            enc_states = tuple(tf.concat([f, b], axis=-1) for f, b in
                               zip(enc_states_fw, enc_states_bw))

        train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(
            encoder_state=enc_states)

        # Appends the start token to the first position of the target_pl.
        start_idx = tf.ones((tf.shape(self.target_pl)[0], 1), dtype=tf.int32)
        start_idx *= yahoo.START_IDX
        seq_vals = tf.concat([start_idx, self.target_pl], axis=1)
        seq_vals = tf.nn.embedding_lookup(emb, seq_vals)

        # Builds the RNN decoder training function.
        cells = [tf.contrib.rnn.GRUCell(rnn_size * 2) for _ in range(num_rnn)]
        cells = [tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=0.7) for cell in cells]
        cells = [AttentionCellWrapper2D(cell, attn_vec) for cell in cells]
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, self.num_classes)
        # cell = ActivationWrapper(cell, 'softplus')

        x, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            cell=cell,
            decoder_fn=train_decoder_fn,
            inputs=seq_vals,
            sequence_length=self.target_len_pl)

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

        # infer_decoder_fn = noisy_decoder_fn_inferrence(
        #     output_fn=None,
        #     encoder_state=enc_states,
        #     embeddings=emb,
        #     start_of_sequence_id=yahoo.START_IDX,
        #     end_of_sequence_id=yahoo.END_IDX,
        #     maximum_length=self.output_len,
        #     num_decoder_symbols=self.num_classes,
        #     temperature=self._temp_pl)

        infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
            output_fn=None,
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
    def sample(self, x, xlen, temp):
        """Samples from the model.

        Args:
            x: Numpy array with shape (batch_size, input_len), ints
                representing the encoded answer.
            xlens: Numpy array with shape (batch_size), ints representing
                the length of the answers.
            temp: float, the temperature for sampling.

        Returns:
            y: Numpy array with shape (batch_size, output_len, num_classes),
                probability distribution over characters in questions.
        """

        feed_dict = {
            self.input_pl: x,
            self.input_len_pl: xlen,
            self.temp_pl: temp,
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
    def temp_pl(self):
        return self._temp_pl

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
