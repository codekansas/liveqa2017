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
                 embeddings=None,
                 logdir=None,
                 only_cpu=False):
        """Constructs a new question generator.

        Args:
            sess: the current TensorFlow session.
            input_len: int, the input length.
            output_len: int, the output length.
            num_classes: int, the number of classes in the model.
            embeddings: pre-trained word embeddngs to use as Numpy array.
            logdir: str, the training directory (otherwise a new temporary one
                is created).
            only_cpu: bool, if set, don't use the GPU.
        """

        self._input_len = input_len
        self._output_len = output_len
        self._num_classes = num_classes
        self._only_cpu = only_cpu
        self._embeddings = embeddings

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

    def get_weight(self, name, shape,
                   init='glorot',
                   device='gpu',
                   weight_val=None,
                   trainable=True):
        """Returns a new weight.

        Args:
            name: str, the name of the variable.
            shape: tuple of ints, the shape of the variable.
            init: str, the type of initialize to use.
            device: str, 'cpu' or 'gpu'.
            weight_val: Numpy array to use as the initial weights.
            trainable: bool, whether or not this weight is trainable.
        """

        if weight_val is None:
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
        else:
            weight_val = weight_val.astype('float32')

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
            weight = tf.Variable(weight_val, name=name, trainable=trainable)
        self._weights.append(weight)

        return weight

    def get_scope_variables(self, scope):
        """Returns all the variables in scope.

        Args:
            scope: str, the scope to use.

        Returns:
            list of variables.
        """

        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def build(self, num_rnn=2, rnn_size=256, num_embed=500):
        """Builds the model, initializing weights.

        Args:
            num_rnn: int, number of stacked RNNs in the encoder.
            rnn_size: int, number of weights in each stacked RNN in the encoder.
            num_embed: int, number of embedding dimensions for each word.
        """

        if self._built:
            return

        x = self.input_pl  # Represents the output after each layer.

        if self.embeddings is not None:
            num_embed = self.embeddings.shape[1]

        # Converts input to one-hot encoding.
        emb = self.get_weight('emb', (self.num_classes, num_embed),
                              init='glorot', weight_val=self.embeddings,
                              trainable=False)
        x = tf.nn.embedding_lookup(emb, x)

        with tf.variable_scope('encoder'):
            cells = [tf.contrib.rnn.GRUCell(rnn_size) for _ in range(num_rnn)]
            cell = tf.contrib.rnn.MultiRNNCell(cells)
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
            enc_states = [tf.concat([f, b], axis=-1)
                          for f, b in zip(enc_states_fw, enc_states_bw)]
            enc_states = tf.concat(enc_states, axis=-1)

            enc_W = self.get_weight('enc_W',
                                    (rnn_size * 2 * num_rnn, rnn_size * 2))
            enc_b = self.get_weight('enc_b', (rnn_size * 2,), init='zero')
            enc_states = tf.nn.bias_add(tf.matmul(enc_states, enc_W), enc_b)

        # Puts the batch dimension first.
        attn_vec = tf.transpose(attn_vec, (1, 0, 2))

        # Builds the attention component.
        k, v, score_fn, construct_fn = tf.contrib.seq2seq.prepare_attention(
            attn_vec, 'bahdanau', rnn_size * 2)

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
                if w not in [emb]:  # Ignores certain weights.
                    reg_loss += tf.nn.l2_loss(w)
            reg_loss *= 1e-4

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
    def sample_from_text(self, texts):
        """Samples given some text.

        Args:
            texts: str or list of str, the text to sample from.

        Returns:
            pred_question: str, the predicted question about the text.
        """

        if isinstance(texts, six.string_types):
            texts = [texts]

        texts_enc = []
        texts_lens = []
        for text in texts:
            _, text, _, text_len, rev_dict = yahoo.tokenize(
                answer=text, use_pad=False, include_rev=True)
            texts_enc.append(text)
            texts_lens.append(text_len)

        questions = self.sample(texts_enc, texts_lens)
        questions = [yahoo.detokenize(question, rev_dict, argmax=True)
                     for question in questions]

        if len(texts) == 1:
            return questions[0]
        else:
            return questions

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
    def embeddings(self):
        return self._embeddings

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


class QuestionGAN(QuestionGenerator):
    """Question generator using SeqGAN model.

    This is still a work in progress. It seems to converge on the same word
    repeated over and over.
    """

    @check_built
    def train(self, qsamples, asamples, qlens, alens, pretrain=False):
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
            pretrain: bool, whether or not to do the pretraining phase.

        Returns:
            gen_loss, dis_loss: floats, the losses of the model.
        """

        feed_dict = {
            self.input_pl: asamples[:, :np.max(alens)],
            self.input_len_pl: alens,
            self.target_pl: qsamples[:, :np.max(qlens)],
            self.target_len_pl: qlens,
        }

        train_op = self.seq_op if pretrain else self.gan_train_op

        _, current_time, summary, gen_acc, dis_acc = self._sess.run(
            [train_op, self.time_op, self.summary_op] +
            [self.gen_acc, self.dis_acc],
            feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, current_time)

        return gen_acc, dis_acc

    def build_generator(self, attn_tensor, embeddings,
                        enc_states, scope='generator'):
        """Builds the generator model, given `attn_vec` and `enc_states`.

        Args:
            attn_tensor: 3D Tensor with shape (batch_size, timesteps, features),
                the vector to pay attention to.
            embeddings: 2D Tensor with shape (num_classes, num_emb_states),
                the embedding vectors for the words (trained using word2vec).
            enc_states: 2D Tensor with shape (batch_size, num_outputs),
                the initial state for the seq2seq decoder.
            scope: str, the generator variable scope.

        Returns:
            generated_sequence: 3D Tensor with shape (batch_size, timesteps,
                num_classes), the sequence produced by the model.
            generator_variables: list of trainable model variables.
            inferred_question: 2D Tensor with shape (batch_size, timesteps),
                an int tensor representing the inferred question.
            seq_loss: the sequence loss (for direct supervised training).
        """

        with tf.variable_scope(scope):
            attn_tensor = tf.transpose(attn_tensor, (1, 0, 2))
            enc_dim = enc_states.get_shape()[1].value

            # Builds the attention component.
            k, v, score_fn, construct_fn = tf.contrib.seq2seq.prepare_attention(
                attn_tensor, 'bahdanau', enc_dim)

            train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(
                enc_states, k, v, score_fn, construct_fn)

            # Appends the start token to the first position of the target_pl.
            start_idx = tf.ones((tf.shape(self.target_pl)[0], 1),
                                dtype=tf.int32) * yahoo.START_IDX
            seq_vals = tf.concat([start_idx, self.target_pl], axis=1)
            seq_vals = tf.nn.embedding_lookup(embeddings, seq_vals)

            # Builds the RNN decoder training function.
            cell = tf.contrib.rnn.GRUCell(enc_dim)

            gen_sequence, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=cell,
                decoder_fn=train_decoder_fn,
                inputs=seq_vals,
                sequence_length=self.target_len_pl)

            # Transforms the RNN hidden weights into class predictions.
            out_W = self.get_weight('out_W', (enc_dim, self.num_classes))
            out_b = self.get_weight('out_b', (self.num_classes,), init='zero')

            # Creates a function that can be used to get the output.
            output_fn = lambda x: tf.matmul(x, out_W) + out_b

            # Applies the output function time-wise.
            gen_sequence = tf.einsum('ijk,kl->ijl', gen_sequence, out_W)
            gen_sequence = tf.nn.bias_add(gen_sequence, out_b)

            # Computes a sequence loss tensor.
            weights = tf.sequence_mask(self.target_len_pl, dtype=tf.float32)
            seq_loss = tf.contrib.seq2seq.sequence_loss(
                gen_sequence, self.target_pl, weights)

            generator_variables = self.get_scope_variables(scope)

            # Reuses the variables from the generator for the inferrer.
            tf.get_variable_scope().reuse_variables()

            # Builds the inferrence functions.
            infer_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(
                output_fn=output_fn,
                encoder_state=enc_states,
                attention_keys=k,
                attention_values=v,
                attention_score_fn=score_fn,
                attention_construct_fn=construct_fn,
                embeddings=embeddings,
                start_of_sequence_id=yahoo.START_IDX,
                end_of_sequence_id=yahoo.END_IDX,
                maximum_length=self.output_len,
                num_decoder_symbols=self.num_classes)

            inferred_question, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=cell,
                decoder_fn=infer_fn)

        return gen_sequence, generator_variables, inferred_question, seq_loss

    def build_discriminator(self, sequence, real_sequence, attn_tensor,
                            embeddings, enc_states, reuse=False,
                            scope='discriminator'):
        """Builds model to descriminate between real and fake `sequence`.

        Args:
            sequence: 2D Tensor with shape (batch_size, num_timesteps),
                the real or fake sequence.
            real_sequence: 2D Tensor with shape (batch_size, num_timesteps),
                the real or fake sequence.
            attn_tensor: 3D Tensor with shape (batch_size, timesteps, features),
                the vector to pay attention to.
            embeddings: 2D Tensor with shape (num_classes, num_emb_states),
                the embedding vectors for the words (trained using word2vec).
            enc_states: 2D Tensor with shape (batch_size, num_outputs),
                the initial state for the seq2seq decoder.
            reuse: bool, whether or not to reuse scope variables.
            scope: str, the discriminator variable scope.

        Returns:
            predictions: 2D Tensor with shape (batch_size, num_timesteps)
                where each value ranges from 0 to 1, 0 being fake and 1 being
                real.
            discriminator_variables: list of trainable model variables.
        """

        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            attn_tensor = tf.transpose(attn_tensor, (1, 0, 2))
            enc_dim = enc_states.get_shape()[1].value

            # Builds the attention component.
            k, v, score_fn, construct_fn = tf.contrib.seq2seq.prepare_attention(
                attn_tensor, 'bahdanau', enc_dim)

            train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(
                enc_states, k, v, score_fn, construct_fn)

            # Creates the cell.
            cell = tf.contrib.rnn.GRUCell(enc_dim)

            # Concatenates with start token and embeds into the embedding space.
            start_idx = tf.ones((tf.shape(real_sequence)[0], 1),
                                dtype=tf.int32) * yahoo.START_IDX
            seq_input = tf.concat([start_idx, real_sequence], axis=1)
            q_emb = tf.nn.embedding_lookup(embeddings, seq_input)

            rnn_states, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=cell,
                decoder_fn=train_decoder_fn,
                inputs=q_emb,
                sequence_length=self.target_len_pl)

            def _add_linear_layer(input_var, output_dim, idx):
                input_dim = input_var.get_shape()[2].value
                W = self.get_weight('W_%d' % idx, (input_dim, output_dim))
                b = self.get_weight('b_%d' % idx, (output_dim,), init='zero')
                o = tf.nn.bias_add(tf.einsum('ijk,kl->ijl', input_var, W), b)
                return o

            # At each timestep, given the current decoder RNN state and the
            # sequence value, predict if the provided sequence token is
            # realistic or unrealistic.
            seq_embed = tf.nn.embedding_lookup(embeddings, sequence)
            preds_concat = tf.concat([rnn_states, seq_embed], axis=2)
            predictions = _add_linear_layer(preds_concat, 256, idx=1)
            predictions = _add_linear_layer(preds_concat, 1, idx=2)
            predictions = tf.squeeze(predictions, axis=2)

            predictions = tf.sigmoid(predictions)

            discriminator_variables = self.get_scope_variables(scope)

        return predictions, discriminator_variables

    def build_encoder(self, sequence, embeddings, num_enc_states=512,
                      rnn_size=256, num_rnns=2, scope='encoder'):
        """Builds the model that encodes the input sequence.

        Args:
            sequence: 2D Tensor with shape (batch_size, num_timesteps),
                the sequence to encode (as integers).
            embeddings: 2D Tensor with shape (num_classes, num_emb_states),
                the embedding vectors for the words (trained using word2vec).
            num_enc_states: int, number of encoder states.
            rnn_size: int, number of RNN hidden dimensions.
            num_rnns: int, number of encoder RNNs.
            scope: str, the encoder variable scope.

        Returns:
            attn_tensor: 3D Tensor with shape (batch_size, num_timesteps,
                num_attn_classes), the tensor to pay attention to.
            enc_states: 2D Tensor with shape (batch_size, num_enc_states),
                the RNN encoded states.
            enc_variables: list of trainable tensors.
        """

        with tf.variable_scope(scope):
            cells = [tf.contrib.rnn.GRUCell(512) for _ in range(num_rnns)]
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            cell = tf.contrib.rnn.FusedRNNCellAdaptor(
                cell, use_dynamic_rnn=True)

            # Converts input to one-hot encoding.
            sequence = tf.nn.embedding_lookup(embeddings, sequence)

            # Applies the forward RNN pass.
            with tf.variable_scope('forward'):
                attn_tensor_fw, enc_states_fw = cell(
                    tf.transpose(sequence, (1, 0, 2)),
                    dtype=tf.float32,
                    sequence_length=self.input_len_pl)

            # Applies the backward RNN pass.
            with tf.variable_scope('backward'):
                cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(cell)
                attn_tensor_bw, enc_states_bw = cell_bw(
                    tf.transpose(sequence, (1, 0, 2)),
                    dtype=tf.float32,
                    sequence_length=self.input_len_pl)

            # Concatenates forward and backward passes of attention vectors.
            attn_tensor = tf.concat([attn_tensor_fw, attn_tensor_bw], axis=-1)

            # Concatenates encoder states to get a single state.
            enc_states = [tf.concat([f, b], axis=-1)
                          for f, b in zip(enc_states_fw, enc_states_bw)]
            enc_states = tf.concat(enc_states, axis=-1)

            # Applies transformation to the attention tensor.
            attn_dim = attn_tensor.get_shape()[2].value
            attn_W = self.get_weight('attn_W', (attn_dim, num_enc_states))
            attn_b = self.get_weight('enc_b', (num_enc_states,), init='zero')
            attn_tensor = tf.einsum('ijk,kl->ijl', attn_tensor, attn_W)
            attn_tensor = tf.nn.bias_add(attn_tensor, attn_b)

            # Applies transformation to get encoder vector.
            output_dims = enc_states.get_shape()[1].value
            enc_W = self.get_weight('enc_W', (output_dims, num_enc_states))
            enc_b = self.get_weight('enc_b', (num_enc_states,), init='zero')
            enc_states = tf.nn.bias_add(tf.matmul(enc_states, enc_W), enc_b)

            enc_variables = self.get_scope_variables(scope)

        return attn_tensor, enc_states, enc_variables

    def build(self, discriminator_phase=None, num_embed=500):
        """Builds the model, initializing weights.

        Args:
            discriminator_phase: int (default: None), how often to alternate
                between training the discriminator and generator. If None, they
                are trained simultaneously.
            num_embed: int, number of embedding dimensions to use.
        """

        if hasattr(self, '_built') and self._built:
            return

        # Creates the time variable on the CPU.
        with tf.device('/cpu:0'):
            self.time = tf.Variable(1, dtype=tf.int64, name='time')
            self.time_op = self.time.assign(self.time + 1)

        x = self.input_pl

        embeddings = self.get_weight('embed', (self.num_classes, num_embed),
                              init='glorot', weight_val=self.embeddings,
                              trainable=False)

        # Builds the encoder model for the generator to use.
        attn_tensor, enc_states, enc_variables = self.build_encoder(
            self.input_pl, embeddings)

        # Builds the generator around the encoded answer.
        gen_sequence, gen_variables, inferred, seq_loss = self.build_generator(
            attn_tensor, embeddings, enc_states)

        # Saves the inferred question to use later.
        self.inferred_question = inferred

        def multinomial_3d(x):
            """Samples from a multinomial distribution from 3D Tensor."""
            a, b = tf.shape(x)[0], tf.shape(x)[1]
            x = tf.reshape(x, (a * b, -1))
            m = tf.multinomial(x, 1)
            return tf.reshape(m, (a, b))

        class_preds = tf.cast(tf.argmax(gen_sequence, axis=-1), tf.int32)
        # class_preds = multinomial_3d(gen_sequence)

        # Builds the discriminator around the real sequence.
        real_preds, dis_variables = self.build_discriminator(
            self.target_pl, self.target_pl, attn_tensor,
            embeddings, enc_states)

        # Builds the discriminator around the generated sequence.
        fake_preds, _ = self.build_discriminator(
            class_preds, self.target_pl, attn_tensor,
            embeddings, enc_states, reuse=True)

        # Computes losses for a normal GAN.
        gen_loss = -tf.reduce_mean(fake_preds)
        dis_loss = tf.reduce_mean(fake_preds) - tf.reduce_mean(real_preds)

        # Computes accuracies.
        self.gen_acc = tf.reduce_mean(fake_preds)
        self.dis_acc = tf.reduce_mean(real_preds)
        tf.summary.scalar('accuracy/generator', self.gen_acc)
        tf.summary.scalar('accuracy/discriminator', self.dis_acc)

        # Computes policy gradients using the generator loss.
        # pred_good = tf.where(tf.greater(fake_preds, 0.5),
        #                      tf.ones_like(fake_preds),
        #                      -tf.ones_like(fake_preds))
        policy = tf.expand_dims(1 - 2 * fake_preds, axis=-1)
        # generator_grad_loss = policy * gen_sequence
        ohe = tf.one_hot(class_preds, self.num_classes)  # Penalize chosen one.
        generator_grad_loss = policy * ohe

        self.gen_loss = gen_loss
        self.dis_loss = dis_loss
        self.gan_loss = self.gen_loss + self.dis_loss
        self.seq_loss = tf.reduce_mean(seq_loss)

        # Adds encoder weights to be trained.
        dis_variables += enc_variables
        # gen_variables += gen_enc_variables

        # Adds generator and discriminator loss summaries.
        tf.summary.scalar('loss/generator', self.gen_loss)
        tf.summary.scalar('loss/discriminator', self.dis_loss)
        tf.summary.scalar('loss/sequence', self.seq_loss)

        # Creates separate generator and discriminator optimizers.
        generator_opt = tf.train.AdamOptimizer(learning_rate=1e-4)
        discriminator_opt = tf.train.AdamOptimizer(learning_rate=3e-4)

        # Adds regularization loss.
        with tf.variable_scope('reg_loss'):
            gen_reg_loss = 0
            for w in gen_variables:
                gen_reg_loss += tf.nn.l2_loss(w)
            gen_reg_loss *= 1e-4

            dis_reg_loss = 0
            for w in dis_variables:
                dis_reg_loss += tf.nn.l2_loss(w)
            dis_reg_loss *= 1e-4

        # Adds regularization loss summaries.
        tf.summary.scalar('loss/regularization/generator', gen_reg_loss)
        tf.summary.scalar('loss/regularization/discriminator', dis_reg_loss)

        # Computes discriminator gradients the normal way.
        dis_op = discriminator_opt.minimize(
            self.dis_loss + dis_reg_loss, var_list=dis_variables)

        # Computes generator gradients.
        gvs = generator_opt.compute_gradients(
            gen_sequence, gen_variables, grad_loss=generator_grad_loss)
        gen_op = generator_opt.apply_gradients(gvs)

        # Adds in the regularization loss (normal minimization).
        gen_op = tf.group(gen_op, generator_opt.minimize(gen_reg_loss))

        # Combines the two ops.
        if discriminator_phase is None:
            self.gan_train_op = tf.group(gen_op, dis_op)
        else:
            tick = tf.mod(self.time, discriminator_phase)
            self.gan_train_op = tf.cond(
                tf.equal(tick, 0),
                lambda: gen_op,
                lambda: dis_op)

        seq_minimize = discriminator_opt.minimize(
            self.seq_loss, var_list=gen_variables)
        self.seq_op = tf.group(seq_minimize, dis_op)  # Pretrain discriminator.

        # Adds the seq op to the gan training op.
        # self.gan_train_op = tf.group(self.gan_train_op, self.seq_op)
        # self.gan_train_op = self.seq_op

        # Creates the log directory and saving objects.
        if self.logdir is None:
            self.logdir = tempfile.mkdtemp()
            sys.stdout.write('logdir: "%s"\n' % self.logdir)
        self.summary_writer = tf.summary.FileWriter(
            self.logdir, self._sess.graph)
        self.summary_op = tf.summary.merge_all()

        self._saver = tf.train.Saver()
        self._sess.run(tf.global_variables_initializer())
        self._built = True

        return self


if __name__ == '__main__':
    LOGDIR = 'model/'  # Where the model is stored.
    sess = tf.Session()

    model = QuestionGenerator(sess,
                              yahoo.ANSWER_MAXLEN,
                              yahoo.QUESTION_MAXLEN,
                              yahoo.NUM_TOKENS,
                              logdir=LOGDIR,
                              only_cpu=True)
    model.load(ignore_missing=True)

    query = raw_input('Enter some text [None to end]: ')
    while query:
        print('Question:', model.sample_from_text(query))
        query = raw_input('Enter some more text [None to end]: ')
