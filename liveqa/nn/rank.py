"""Neural network re-ranking for question-answer vector embedding."""

from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import datetime

import numpy as np

from keras.models import Model
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional
from keras.losses import cosine_proximity
import keras.backend as K

from liveqa.nn.utils import RecurrentAttention
from liveqa import yahoo

BASE = os.path.dirname(os.path.realpath(__file__))
SAVE_LOC = os.path.join(BASE, 'ranking_model.h5')


def rank_candidate_answers(model, question, answers):
    """Ranks answers by relevance to the question.

    Args:
        model: the Keras model (defined in `build_model`).
        question: str, the question string.
        ansewrs: list of str, the answer strings.

    Returns:
        the answers, sorted by decreasing relevance (i.e. first answer is the
            most relevant answer).
    """

    raise NotImplementedError()


def build_model(embeddings, question_len, answer_len):
    """Builds the question-answer similarity model.

    Args:
        embeddings: numpy array with size (vocab_size, num_embedding_dims),
            the initial embeddings to use for the model.
        question_len: int, maximum length of a question.
        answer_len: int, maximum length of an answer.

    Returns:
        a trainable keras model.
    """

    RNN_DIMS = 64

    question_var = Input(shape=(question_len,), dtype='int32')
    answer_var = Input(shape=(answer_len,), dtype='int32')
    neg_answer_var = Input(shape=(answer_len,), dtype='int32')

    # Applies the context representation layer.
    vocab_size, num_embedding_dims = embeddings.shape
    q_emb = Embedding(vocab_size, num_embedding_dims, weights=[embeddings])
    q = q_emb(question_var)
    q_context = Bidirectional(LSTM(RNN_DIMS, return_sequences=True))(q)
    a_emb = Embedding(vocab_size, num_embedding_dims, weights=[embeddings])
    a = a_emb(answer_var)
    a_lstm = Bidirectional(LSTM(RNN_DIMS, return_sequences=True))
    a_context = a_lstm(a)
    n = a_emb(neg_answer_var)
    n_context = a_lstm(n)

    # Applies the matching layer.
    q_matching_lstm = LSTM(RNN_DIMS, return_sequences=True)
    q_matching_lstm = RecurrentAttention(q_matching_lstm, a_context)
    q_match = q_matching_lstm(q_context)
    a_matching_lstm = LSTM(RNN_DIMS, return_sequences=True)
    a_matching_lstm = RecurrentAttention(a_matching_lstm, q_context)
    a_match = a_matching_lstm(a_context)
    n_match = a_matching_lstm(n_context)

    # Applies the aggregation layer.
    q_agg = Bidirectional(LSTM(RNN_DIMS, return_sequences=False))(q_match)
    a_agg_lstm = Bidirectional(LSTM(RNN_DIMS, return_sequences=False))
    a_agg = a_agg_lstm(a_match)
    n_agg = a_agg_lstm(n_match)

    # Builds the similarity loss function.
    def sim_diff_func(x):
        q, a, n = x
        q = K.l2_normalize(q, axis=-1)
        a = K.l2_normalize(a, axis=-1)
        n = K.l2_normalize(n, axis=-1)
        qa = -K.mean(q * a, axis=-1, keepdims=True)
        qn = -K.mean(q * n, axis=-1, keepdims=True)
        return qn - qa
    sim_layer = Lambda(sim_diff_func)
    sim_diff = sim_layer([q_agg, a_agg, n_agg])

    def similarity_loss(_, y_pred):
        return K.maximum(0., 0.2 + y_pred)

    # The model is trained to maximize similarity.
    model = Model(inputs=[question_var, answer_var, neg_answer_var],
                  outputs=[sim_diff])
    model.compile(optimizer='adam', loss=similarity_loss)

    return model


if __name__ == '__main__':  # Tests the model on some dummy data.
    BATCH_SIZE = 32
    NB_EPOCH = 100
    BATCHES_PER_EPOCH = 100

    embeddings = yahoo.get_word_embeddings()
    model = build_model(embeddings, yahoo.QUESTION_MAXLEN, yahoo.ANSWER_MAXLEN)

    if os.path.exists(SAVE_LOC):
        model.load_weights(SAVE_LOC)

    # Gets the data iterator.
    data_iter = yahoo.iterate_answer_to_question(BATCH_SIZE, False)
    sample_iter = yahoo.iterate_answer_to_question(1, True)

    q_old, a_old, _, _ = data_iter.next()
    target = np.ones(shape=(BATCH_SIZE,))

    total_start_time = datetime.datetime.now()
    for epoch_idx in xrange(1, NB_EPOCH + 1):
        start_time = datetime.datetime.now()
        for batch_idx in xrange(1, BATCHES_PER_EPOCH + 1):
            q_new, a_new, _, _ = data_iter.next()
            model.train_on_batch([q_old, a_old, a_new], [target])
            loss = model.train_on_batch([q_new, a_new, a_old], [target])
            q_old, a_old = q_new, a_new
            sys.stdout.write('\repoch %d -- batch %d -- loss = %.3f        '
                             % (epoch_idx, batch_idx, loss))
            sys.stdout.flush()

        time_passed = (datetime.datetime.now() - start_time).total_seconds()
        total_time_passed = (datetime.datetime.now()
                             - total_start_time).total_seconds()

        # Saves the current model weights.
        model.save_weights(SAVE_LOC)
