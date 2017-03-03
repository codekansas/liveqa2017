#!/usr/bin/env python
"""
Components for the question generator network.

This is only a temporary starting point, since it isn't a true seq2seq model.
The eventual "deployable" model will be built in raw TensorFlow, since Keras
doesn't support seq2seq models very well. In it's current iteration, the model
isn't very good at producing novel results.
"""

from __future__ import division

import os
import sys

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras.layers import Convolution1D
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import LSTM as RNN
from keras.layers import MaxPooling1D
from keras.layers import merge
from keras.layers import RepeatVector
from keras.models import load_model
from keras.models import Model

import numpy as np

import yahoo


def build_model(input_len, output_len, num_classes):
    """Builds the question generator network.

    The input to the model is answer, which is being analyzed. The output of
    the network is the generated question. The model is trained to produce
    the original question.

    Args:
        input_len: int, the length of the input sequence.
        output_len: int, the length of the output sequence.
        num_classes: int, the number of input classes (e.g. characters).
    """

    input_var = Input(shape=(input_len,), dtype='int32')

    # Embeds the input into a vector space.
    x = Embedding(num_classes, num_classes, init='identity')(input_var)

    # Applies the encoder part.
    for i in range(5):
        a = Convolution1D(nb_filter=32, filter_length=2, border_mode='same')(x)
        b = Convolution1D(nb_filter=32, filter_length=1, border_mode='same')(x)
        b = Convolution1D(nb_filter=32, filter_length=3, border_mode='same')(b)
        c = Convolution1D(nb_filter=32, filter_length=5, border_mode='same')(x)
        x = merge([a, b, c], mode='concat', concat_axis=-1)
        x = MaxPooling1D(2)(x)
        x = BatchNormalization()(x)
    x = GlobalMaxPooling1D()(x)

    # Applies the decoder part.
    x = RepeatVector(output_len)(x)
    for i in range(2):
        x = RNN(64, return_sequences=True)(x)
    for i in range(2):
        x_part = RNN(64, return_sequences=True)(x)
        x = merge([x, x_part])

    # Produces a softmax distribution over the outputs.
    x = Convolution1D(num_classes, 1, activation='softmax')(x)

    return Model([input_var], [x])


class ModelSampler(Callback):
    """Callback that generates samples from the model."""

    def __init__(self, input_data, target_data, *args, **kwargs):
        self.num_samples = kwargs.pop('num_samples', 5)
        self.input_data = input_data[:self.num_samples]
        self.target_data = target_data[:self.num_samples]

        # Checks the input and target data shapes.
        if (self.input_data.shape[0] != self.num_samples or
            self.target_data.shape[0] != self.num_samples):
            raise ValueError('ModelSampler data should have at least %d '
                             '[=kwargs.num_samples] samples, but got %d '
                             'input samples and %d target samples.'
                             % (self.num_samples, self.input_data.shape[0],
                                self.target_data.shape[0]))

        # Builds the input strings (to avoid rebuilding each epoch).
        input_strs = [yahoo.detokenize(s, argmax=False)
                      for s in self.input_data]
        target_strs = [yahoo.detokenize(s, argmax=False)
                       for s in self.target_data]
        self.print_strs = ['Sample ' + str(i) + ':\n  Input = ' + input_str +
                           '\n  Target = ' + target_str +
                           '\n  Output = ' for i, (input_str, target_str) in
                           enumerate(zip(input_strs, target_strs))]

        super(ModelSampler, self).__init__(*args, **kwargs)

    def on_epoch_end(self, batch, logs={}):
        preds = self.model.predict(self.input_data)

        # Writes the output predictions.
        sys.stdout.write('\n')
        for i in range(self.num_samples):
            sys.stdout.write(self.print_strs[i])
            sys.stdout.write(yahoo.detokenize(preds[i], argmax=True))
            sys.stdout.write('\n')
        sys.stdout.flush()


if __name__ == '__main__':

    # Constants.
    BATCH_SIZE = 32
    SAMPLES_PER_EPOCH = BATCH_SIZE * 100
    NB_EPOCH = 100
    WEIGHTS_SAVE_PATH = '/tmp/qgen_weights.h5'
    REBUILD_MODEL = False

    model = build_model(input_len=yahoo.ANSWER_MAXLEN,
                        output_len=yahoo.QUESTION_TITLE_MAXLEN,
                        num_classes=yahoo.NUM_TOKENS)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

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

    # Builds the checkpoint to save weights.
    checkpointer = ModelCheckpoint(filepath=WEIGHTS_SAVE_PATH,
                                   save_best_only=True,
                                   save_weights_only=True)

    # Builds the model sampler checkpoint.
    a_sample, q_sample = data_iter.next()
    q_sample = np.squeeze(q_sample)
    sampler = ModelSampler(a_sample, q_sample)

    # Fits the data to the model.
    model.fit_generator(data_iter,
                        samples_per_epoch=SAMPLES_PER_EPOCH,
                        nb_epoch=NB_EPOCH,
                        callbacks=[checkpointer, sampler])
