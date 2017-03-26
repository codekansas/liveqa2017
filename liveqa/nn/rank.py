"""Neural network re-ranking for question-answer vector embedding."""

from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import datetime

import numpy as np

from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Masking
from keras.layers import Input
from keras.layers import GlobalMaxPooling1D
from keras.layers import Lambda
from keras.layers import LSTM
from keras.layers.merge import concatenate
from keras.layers.wrappers import Bidirectional
from keras.losses import cosine_proximity
import keras.backend as K

from liveqa.nn.utils import RecurrentAttention
from liveqa import yahoo

BASE = os.path.dirname(os.path.realpath(__file__))
SAVE_LOC = os.path.join(BASE, 'ranking_model.h5')


def red(t):
    return '\033[91m' + t + '\033[0m'


def green(t):
    return '\033[92m' + t + '\033[0m'


def yellow(t):
    return '\033[93m' + t + '\033[0m'


def blue(t):
    return '\033[94m' + t + '\033[0m'


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

    # Gets the batch size.
    b_size = len(answers)

    # Tokenizes the answers.
    q_arr, a_arr = [], []
    for answer in answers:
        q_tok, a_tok, _, _, rev_dict = yahoo.tokenize(
            question=question,
            answer=answer,
            use_pad=True,
            include_rev=True)
        a_arr.append(a_tok)
    q_arr = np.stack(q_arr)
    a_arr = np.stack(a_arr)

    # Gets the predictions and ranks them.
    preds = model.predict([q_arr, a_arr])
    a_ranked = [answers[i] for i in np.argsort(preds)][::-1]

    return a_ranked


def build_model(embeddings, question_len, answer_len, mode='recurrent'):
    """Builds the question-answer similarity model.

    Args:
        embeddings: numpy array with size (vocab_size, num_embedding_dims),
            the initial embeddings to use for the model.
        question_len: int, maximum length of a question.
        answer_len: int, maximum length of an answer.
        mode: str, either 'convolutional' or 'recurrent', the model type to
            use.

    Returns:
        a trainable keras model.
    """

    RNN_DIMS = 64
    CONV_DIMS = 64
    MARGIN = 0.2

    question_var = Input(shape=(question_len,))
    answer_var = Input(shape=(answer_len,))

    q_var, a_var = question_var, answer_var

    if mode == 'recurrent':
        mask = Masking(yahoo.END_IDX)
        q_var, a_var = mask(q_var), mask(a_var)

    # Applies the embedding layer.
    vocab_size, num_embedding_dims = embeddings.shape
    emb = Embedding(vocab_size, num_embedding_dims, weights=[embeddings],
                    trainable=False)
    q_var = emb(q_var)
    a_var = emb(a_var)

    if mode == 'convolutional':
        conv_filters = [1, 2, 3, 4, 5, 6]
        for _ in range(3):
            convs = [Conv1D(CONV_DIMS, i, padding='same')
                       for i in conv_filters]
            reg_conv = Conv1D(CONV_DIMS, 1, padding='same')
            batch_norm = BatchNormalization()

            q_var = reg_conv(q_var)
            q_var = concatenate([c(q_var) for c in convs])
            q_var = batch_norm(q_var)

            convs = [Conv1D(CONV_DIMS, i, padding='same')
                       for i in conv_filters]
            reg_conv = Conv1D(CONV_DIMS, 1, padding='same')
            batch_norm = BatchNormalization()

            a_var = reg_conv(a_var)
            a_var = concatenate([c(a_var) for c in convs])
            a_var = batch_norm(a_var)

        q_var = GlobalMaxPooling1D()(q_var)
        a_var = GlobalMaxPooling1D()(a_var)

    elif mode == 'recurrent':
        # Applies the context layer.
        q_context = Bidirectional(LSTM(RNN_DIMS, return_sequences=True))(q_var)
        a_lstm = Bidirectional(LSTM(RNN_DIMS, return_sequences=True))
        a_context = a_lstm(a_var)

        # Applies the matching layer.
        q_matching_lstm = LSTM(RNN_DIMS, return_sequences=True)
        q_matching_lstm = RecurrentAttention(q_matching_lstm, a_context)
        q_match = q_matching_lstm(q_context)
        a_matching_lstm = LSTM(RNN_DIMS, return_sequences=True)
        a_matching_lstm = RecurrentAttention(a_matching_lstm, q_context)
        a_match = a_matching_lstm(a_context)

        # Applies the aggregation layer.
        q_var = Bidirectional(LSTM(RNN_DIMS, return_sequences=False))(q_match)
        a_agg_lstm = Bidirectional(LSTM(RNN_DIMS, return_sequences=False))
        a_var = a_agg_lstm(a_match)

    qa_var = concatenate([q_var, a_var])

    # Applies a couple of layers on top.
    d1 = Dense(512, activation='tanh')
    d2 = Dense(512, activation='tanh')
    d3 = Dense(1, activation='sigmoid')
    qa_var = d3(d2(d1(qa_var)))

    def max_cross_entropy(y_true, y_pred):
        """Binary crossentropy, but only on the most confusing example.

        This is potentially a good idea because it makes the model focus on
        examples near the decision boundary instead of worrying about examples
        that are far away from the decision boundary.
        """
        return K.max(K.binary_crossentropy(y_pred, y_true))

    # The model is trained to maximize similarity.
    model = Model(inputs=[question_var, answer_var], outputs=[qa_var])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    eval_model = Model(inputs=[question_var, answer_var], outputs=[qa_var])

    return model, eval_model


def evaluate(model, questions, answers, rev_dict, n_eval=5):
    """Evaluates the model on some questions and answers.

    Args:
        model: the keras model being evaluated.
        questions: numpy array with shape (batch_size, question_len,
            num_classes), the encoded questions.
        answers: numpy array with shape (batch_size, answer_len, num_classes),
            the encoded answers.
        rev_dict: dict, the tokens->words dictionary.
        n_eval: int, number of question-answer pairs to print.
    """

    b_size = questions.shape[0]
    n_eval = min(b_size, n_eval)

    for i in range(n_eval):
        question = np.stack([questions[i] for _ in range(b_size)])
        preds = model.predict([question, answers])
        question = yahoo.detokenize(questions[i], rev_dict[i])
        answer = yahoo.detokenize(answers[i], rev_dict[i])
        pidx = np.argmax(preds)
        pred = yahoo.detokenize(answers[pidx], rev_dict[i])
        s = []
        ctext = green('CORRECT') if pidx == i else red('INCORRECT')
        s.append('[ question %d %s ] %s' % (i + 1, ctext, question))
        s.append('%s : %s (%s)' % (blue('prediction'), pred,
                                   yellow('%.3f' % preds[pidx])))
        s.append('%s : %s (%s)' % (blue('target'), answer,
                                        yellow('%.3f' % preds[i])))
        s.append('%s : %s - %s' % (blue('pred range'),
                                   yellow('%.3f' % np.min(preds)),
                                   yellow('%.3f' % np.max(preds))))
        sys.stdout.write(' -- '.join(s) + '\n')
    sys.stdout.flush()


if __name__ == '__main__':  # Tests the model on some dummy data.
    BATCH_SIZE = 32
    NB_EPOCH = 10000
    BATCHES_PER_EPOCH = 100
    N_EVAL = 5

    embeddings = yahoo.get_word_embeddings()
    model, eval_model = build_model(embeddings, yahoo.QUESTION_MAXLEN, yahoo.ANSWER_MAXLEN)

    if os.path.exists(SAVE_LOC):
        model.load_weights(SAVE_LOC)

    # Gets the data iterator.
    data_iter = yahoo.iterate_answer_to_question(BATCH_SIZE, False)
    sample_iter = yahoo.iterate_answer_to_question(100, True)

    q_old, a_old, _, _ = data_iter.next()
    pos_targets = np.ones(shape=(BATCH_SIZE,))
    neg_targets = np.zeros(shape=(BATCH_SIZE,))
    targets = np.concatenate([pos_targets, neg_targets,
                              neg_targets, pos_targets])

    total_start_time = datetime.datetime.now()
    for epoch_idx in xrange(1, NB_EPOCH + 1):
        start_time = datetime.datetime.now()
        for batch_idx in xrange(1, BATCHES_PER_EPOCH + 1):
            q_new, a_new, _, _ = data_iter.next()
            q_in = np.concatenate([q_new, q_new, q_old, q_old])
            a_in = np.concatenate([a_new, a_old, a_new, a_old])
            loss, acc = model.train_on_batch([q_in, a_in], [targets])
            q_old, a_old = q_new, a_new
            sys.stdout.write('\repoch %d -- batch %d '
                             '-- loss = %.3f -- acc = %.3f    '
                             % (epoch_idx, batch_idx, loss, acc))
            sys.stdout.flush()

        time_passed = (datetime.datetime.now() - start_time).total_seconds()
        total_time_passed = (datetime.datetime.now()
                             - total_start_time).total_seconds()
        sys.stdout.write(yellow('\repoch %d -- loss = %.3f '
                                '-- acc = %.3f -- time passed = '
                                '%d (%d) seconds\n'
                                % (epoch_idx, loss, acc,
                                   time_passed, total_time_passed)))
        sys.stdout.flush()

        # Saves the current model weights.
        model.save_weights(SAVE_LOC)

        # Evaluates the model on some training data.
        q_eval, a_eval, _, _, rev_dict = sample_iter.next()
        evaluate(eval_model, q_eval, a_eval, rev_dict, n_eval=N_EVAL)
