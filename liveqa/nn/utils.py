"""Some utils used by either the model or the ranker."""

from __future__ import absolute_import
from __future__ import print_function

import keras.backend as K
import keras


class RecurrentAttention(keras.layers.Wrapper):
    """Makes a recurrent layer pay attention to an attention tensor.
    This implementation takes an attention tensor with shape (batch_size,
    num_input_timesteps, num_features). On each recurrent step, the hidden
    state is weighted by the a vector `s`, which is computed as a weighted sum
    of the input vectors as follows:
        t = time_dist_activation(dot(h, U_t) + b_t)
        w = sum(t * attention)
        s = attn_gate_func(dot(w, U_a) + b_a)
        h_new = s * h
    Generally, on each timestep, the hidden state is used to compute a weight
    distribution over each timestep in the attention tensor. This is used to
    get a weighted sum, which has shape (batch_size, num_attn_feats). This is
    linearly transformed to get `s`, which weights the hidden state.
    Args:
        layer: Keras Recurrent layer, the layer to apply the attention to.
        attention: Keras tensor with shape (batch_size, num_timesteps,
            num_features). For example, this could the output of a Dense or
            GlobalMaxPooling1D layer.
        time_dist_activation: activation function. Can be the name of an
            existing function (str) or another function. See Keras
            [activations](https://keras.io/activations/). A softmax function
            intuitively means "determine how important each time input is".
        attn_gate_func: activation function. Can be the name of an existing
            function (str) or another function. See Keras
            [activations](https://keras.io/activations/) and the equations.
        W_regularizer: instance of Keras WeightRegularizer. See Keras
            [regularizers](https://keras.io/regularizers/). Applied to all of
            the weight matrices.
        b_regularizer: instance of Keras WeightRegularizer. See Keras
            [regularizers](https://keras.io/regularizers/). Applied to all of
            the bias vectors.
    """

    def __init__(self, layer, attention, time_dist_activation='softmax',
                 attn_gate_func='sigmoid', kernel_initializer='glorot_uniform',
                 W_regularizer=None, b_regularizer=None, **kwargs):

        if not isinstance(layer, keras.layers.Recurrent):
            raise ValueError('The RecurrentAttention wrapper only works on '
                             'recurrent layers.')

        # Should know this so that we can handle multiple hidden states.
        self._wraps_lstm = isinstance(layer, keras.layers.LSTM)

        if not hasattr(attention, '_keras_shape'):
            raise ValueError('Attention should be a Keras tensor.')

        if len(K.int_shape(attention)) != 3:
            raise ValueError('The attention input for RecurrentAttention '
                             'should be a tensor with shape (batch_size, '
                             'num_timesteps, num_features). Got shape=%s.' %
                             str(K.int_shape(attention)))

        self.supports_masking = True
        self.attention = attention

        self.time_dist_activation = keras.activations.get(time_dist_activation)
        self.attn_gate_func = keras.activations.get(attn_gate_func)
        self.kernel_initializer = kernel_initializer

        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        super(RecurrentAttention, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.input_spec = keras.engine.InputSpec(shape=input_shape)

        # Builds the wrapped layer.
        if not self.layer.built:
            self.layer.build(input_shape)

        super(RecurrentAttention, self).build()

        num_attn_timesteps, num_attn_feats = K.int_shape(self.attention)[1:]
        output_shape = self.layer.compute_output_shape(input_shape)
        output_dim = output_shape[-1]

        self.attn_U_t = self.add_weight((output_dim, num_attn_timesteps),
                                   initializer=self.kernel_initializer,
                                   name='{}_attn_U_t'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.attn_b_t = self.add_weight((num_attn_timesteps,),
                                   initializer='zero',
                                   name='{}_attn_b_t'.format(self.name),
                                   regularizer=self.b_regularizer)

        self.attn_U_a = self.add_weight((num_attn_feats, output_dim),
                                   initializer=self.kernel_initializer,
                                   name='{}_attn_U_a'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.attn_b_a = self.add_weight((output_dim,),
                                   initializer='zero',
                                   name='{}_attn_b_a'.format(self.name),
                                   regularizer=self.b_regularizer)

        self.built = True

    def reset_states(self):
        self.layer.reset_states()

    def _compute_attention(self, h):
        t_weights = K.expand_dims(K.dot(h, self.attn_U_t) + self.attn_b_t, -1)
        t_weights = self.time_dist_activation(t_weights)
        weighted_sum = K.sum(t_weights * self.attention, axis=1)
        attn_vec = K.dot(weighted_sum, self.attn_U_a) + self.attn_b_a
        return self.attn_gate_func(attn_vec)

    def step(self, x, states):
        if self._wraps_lstm:  # If the recurrent layer is an LSTM.
            h, [_, c] = self.layer.step(x, states)
            h *= self._compute_attention(h)
            return h, [h, c]

        else:  # All other RNN types.
            h, [h] = self.layer.step(x, states)
            h *= self._compute_attention(h)
            return h, [h, c]

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, inputs, mask=None, initial_state=None, training=None):
        input_shape = K.int_shape(inputs)
        constants = self.layer.get_constants(inputs, training=None)
        preprocessed_input = self.layer.preprocess_input(inputs, training=None)

        initial_states = self.layer.get_initial_states(inputs)

        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1])

        return outputs if self.layer.return_sequences else last_output

    def get_config(self):
        raise NotImplementedError('Saving attention components is not '
                                  'supported yet.')
