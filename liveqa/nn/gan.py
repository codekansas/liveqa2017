"""Neural network for generating questions from answers."""

from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import datetime

import numpy as np

from keras import losses
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
from keras.layers import RepeatVector
from keras.layers.merge import concatenate
from keras.layers.wrappers import Bidirectional
import keras.backend as K

from liveqa.nn.utils import RecurrentAttention
from liveqa import yahoo

BASE = os.path.dirname(os.path.realpath(__file__))
SAVE_LOC = os.path.join(BASE, 'gan_model.h5')


def red(t):
    return '\033[91m' + t + '\033[0m'


def green(t):
    return '\033[92m' + t + '\033[0m'


def yellow(t):
    return '\033[93m' + t + '\033[0m'


def blue(t):
    return '\033[94m' + t + '\033[0m'


def cosine_similarity(x, y):
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.sum(x * y, axis=-1)


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


def build_generator(l_var, a_var, embeddings):
    """Builds a question generator model.

    Args:
        l_var: keras tensor, the latent vector input.
        a_var: keras tensor, the answer input.
        embeddings: numpy array, the embeddings to use for knn decoding.
    """

    latent_var = Input(tensor=l_var, name='latent_var_pl')
    answer_var = Input(tensor=a_var, name='gen_answer_pl')
    l_var, a_var = latent_var, answer_var

    RNN_DIMS = 64
    vocab_size, num_embedding_dims = embeddings.shape

    # Computes context of the answer.
    a_lstm = Bidirectional(LSTM(RNN_DIMS, return_sequences=True))
    a_context = a_lstm(a_var)

    # Uses context to formulate a question.
    q_matching_lstm = LSTM(RNN_DIMS, return_sequences=True)
    q_matching_lstm = RecurrentAttention(q_matching_lstm, a_context)
    q_var = q_matching_lstm(l_var)
    q_var = LSTM(RNN_DIMS, return_sequences=True)(q_var)
    q_var = Dense(num_embedding_dims)(q_var)

    # Builds the model from the variables (not compiled).
    model = Model(inputs=[latent_var, answer_var], outputs=[q_var])

    return model


def build_discriminator(q_var, a_var, mode='recurrent'):
    """Builds the question-answer similarity model.

    Args:
        q_var: keras tensor representing the question input, with shape
            (batch_size, question_len, num_embedding_dims).
        a_var: keras tensor representing the answer input, with shape
            (batch_size, answer_len, num_embedding_dims).
        question_len: int, maximum length of a question.
        answer_len: int, maximum length of an answer.
        mode: str, either 'convolutional' or 'recurrent', the model type to
            use.

    Returns:
        a trainable keras model.
    """

    question_var = Input(tensor=q_var, name='dis_question_pl')
    answer_var = Input(tensor=a_var, name='dis_answer_pl')
    q_var, a_var = question_var, answer_var

    RNN_DIMS = 64
    CONV_DIMS = 64
    MARGIN = 0.2

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

    else:
        raise ValueError('Invalid mode: "%s"' % mode)

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

    return qa_var, model


def build_gan(num_latent_dims):
    """Builds a generative adversarial network.

    To train the GAN, run the updates on the generator and discriminator model
    in a loop.

    Args:
        num_latent_dims: int, number of latent dimensions in the generator.
    """

    embeddings = yahoo.get_word_embeddings()

    question_var = Input(shape=(yahoo.QUESTION_MAXLEN,), name='question_var')
    answer_var = Input(shape=(yahoo.ANSWER_MAXLEN,), name='answer_var')
    latent_var = Input(shape=(num_latent_dims,), name='latent_var')

    vocab_size, num_embedding_dims = embeddings.shape
    emb = Embedding(vocab_size, num_embedding_dims, weights=[embeddings],
                    trainable=False)

    q_var = emb(question_var)
    a_var = emb(answer_var)
    l_var = RepeatVector(yahoo.QUESTION_MAXLEN)(latent_var)

    # Creates the two models.
    gen_model = build_generator(l_var, a_var, embeddings)
    real_preds, dis_model = build_discriminator(q_var, a_var)

    # Builds the model to train the generator.
    dis_model.trainable = False
    gen_preds = dis_model([gen_model([l_var, a_var]), a_var])

    # Builds the model to train the discriminator.
    dis_model.trainable = True
    gen_model.trainable = False
    fake_preds = dis_model([q_gen, a_var])

    # Computes predictions.
    preds = pred_model([l_var, a_var])

    return gen_model, dis_model


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
    NUM_LATENT_DIMS = 100

    embeddings = yahoo.get_word_embeddings()
    model = build_gan(NUM_LATENT_DIMS)

    if os.path.exists(SAVE_LOC):
        model.load_weights(SAVE_LOC)

    # Gets the data iterator.
    data_iter = yahoo.iterate_answer_to_question(BATCH_SIZE, False)
    sample_iter = yahoo.iterate_answer_to_question(100, True)

    targets = np.ones(shape=(BATCH_SIZE, yahoo.NUM_TOKENS))

    total_start_time = datetime.datetime.now()
    for epoch_idx in xrange(1, NB_EPOCH + 1):
        start_time = datetime.datetime.now()
        for batch_idx in xrange(1, BATCHES_PER_EPOCH + 1):
            q, a, _, _ = data_iter.next()
            n = np.random.normal(size=(BATCH_SIZE, NUM_LATENT_DIMS))
            loss = model.train_on_batch([q, a, n], [targets])
            sys.stdout.write('\repoch %d -- batch %d '
                             '-- loss = %.3f    '
                             % (epoch_idx, batch_idx, loss))
            sys.stdout.flush()

        # Saves the current model weights.
        model.save_weights(SAVE_LOC)
