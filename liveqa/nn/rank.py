"""Neural network re-ranking for question-answer vector embedding."""

from __future__ import absolute_import
from __future__ import print_function

import sys

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


def build_model(vocab_size, num_embedding_dims, question_len, answer_len):
    """Builds the question-answer similarity model.

    Args:
        vocab_size: int, number of symbols in the vocabulary.
        num_embedding_dims: int, number of dimensions to embed the question
            and answer (corresponds to pre-trained word2vec embeddings).
        question_len: int, maximum length of a question.
        answer_len: int, maximum length of an answer.

    Returns:
        a trainable keras model.
    """

    RNN_DIMS = 256

    question_var = Input(shape=(question_len,), dtype='int32')
    answer_var = Input(shape=(answer_len,), dtype='int32')
    neg_answer_var = Input(shape=(answer_len,), dtype='int32')

    # Applies the context representation layer.
    q = Embedding(vocab_size, num_embedding_dims)(question_var)
    q_context = Bidirectional(LSTM(RNN_DIMS, return_sequences=True))(q)
    a_emb = Embedding(vocab_size, num_embedding_dims)
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
        return qa - qn
    sim_layer = Lambda(sim_diff_func)
    sim_diff = sim_layer([q_agg, a_agg, n_agg])

    def similarity_loss(y_true, _):
        return K.maximum(0., 0.2 - y_true)

    # The model is trained to maximize similarity.
    model = Model(inputs=[question_var, answer_var, neg_answer_var],
                  outputs=[sim_diff])
    model.compile(optimizer='adam', loss='mse')

    return model


if __name__ == '__main__':  # Tests the model on some dummy data.
    VOCAB_SIZE = 200
    NUM_EMBED = 500
    QUESTION_LEN, ANSWER_LEN = 50, 100
    BATCH_SIZE = 32

    model = build_model(VOCAB_SIZE, NUM_EMBED, QUESTION_LEN, ANSWER_LEN)

    def _iterate_questions():
        while True:
            question = np.random.randint(
                0, VOCAB_SIZE, size=(BATCH_SIZE, QUESTION_LEN), dtype='int32')
            answer = np.random.randint(
                0, VOCAB_SIZE, size=(BATCH_SIZE, ANSWER_LEN), dtype='int32')
            yield question, answer

    iterator = _iterate_questions()
    q_1, a_1 = iterator.next()

    for i in xrange(5):
        q_2, a_2 = iterator.next()
        t = np.ones(shape=(BATCH_SIZE,))
        model.train_on_batch([q_1, a_1, a_2], [t])
        model.train_on_batch([q_2, a_2, a_1], [t])
        q_1, a_1 = q_2, a_2  # Moves new batch into the old batch.
        sys.stdout.write('\rprocessed %d' % i)
        sys.stdout.flush()
