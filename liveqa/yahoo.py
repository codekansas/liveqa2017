"""
Code for reading from Yahoo L6 data.

The raw data is in XML format. These utils iterate through this data and
return question-answer pairs. This provides a way to standardize access to the
corpus.
"""

from __future__ import absolute_import
from __future__ import print_function

from gensim import models, corpora, similarities

import cPickle as pkl
import itertools
import os
import six
import sys
import re
import xml.etree.cElementTree as ET

from collections import Counter

import numpy as np
import logging

# Constants.
QUESTION_MAXLEN = 20
ANSWER_MAXLEN = 100
DICT_SIZE = 50000  # Maximum number of words to include in the corpus.
DATA_ENV_NAME = 'YAHOO_DATA'  # Directory containing the Yahoo data, unzipped.
YAHOO_L6_URL = 'http://webscope.sandbox.yahoo.com/catalog.php?datatype=l'

# Variables that will be filled in later.
BASE = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'FullOct2007.xml')
DICTIONARY_FILE = os.path.join(BASE, 'data', 'dictionary_dict.pkl')
EMBEDDINGS_FILE = os.path.join(BASE, 'data', 'word_embeddings.h5')

if not os.path.exists(DATA_PATH):
    raise RuntimeError('File not found: "%s". To create it from the existing '
                       'files, run:  cat FullOct2007.xml.part1 '
                       'FullOct2007.xml.part2 > FullOct2007.xml' % DATA_PATH)


def word_tokenize(text):
    """Tokenizes text (as a string) to a list of tokens."""

    if text is None:
        text = ''
    elif not isinstance(text, six.string_types):
        text = '' if text is None else text.text

    text = re.sub('\<.+?\>', '', text)
    text = re.findall('[\w\d\']+|[?\.,!-\*;\"@#$%^&\(\)]+', text.lower())

    return text


if not os.path.exists(DICTIONARY_FILE):  # Creates the dictionary.
    counter = Counter()
    with open(DATA_PATH, 'r') as f:
        parser = ET.iterparse(f)
        num_docs = 0
        for event, elem in parser:
            if elem.tag == 'document':
                num_docs += 1
                counter.update(word_tokenize(elem.find('subject')))
                counter.update(word_tokenize(elem.find('bestanswer')))

            if num_docs % 1000 == 0:
                sys.stdout.write('\rparsed %d docs, %d words'
                                 % (num_docs, len(counter)))
                sys.stdout.flush()

            if num_docs == 200000:
                break

    _DICTIONARY = [w for w, _ in counter.most_common(DICT_SIZE)]
    with open(DICTIONARY_FILE, 'wb') as f:
        pkl.dump(_DICTIONARY, f)
else:
    with open(DICTIONARY_FILE, 'rb') as f:
        _DICTIONARY = pkl.load(f)

# Extra tokens:
# PADDING = 0
OUT_OF_VOCAB = 0
START_IDX = 1
END_IDX = 2
NUM_SPECIAL = 3 + ANSWER_MAXLEN  # OUT_OF_VOCAB, START_IDX, END_IDX

_CHAR_TO_IDX = dict((c, i + NUM_SPECIAL) for i, c in enumerate(_DICTIONARY))
NUM_TOKENS = NUM_SPECIAL + len(_DICTIONARY)


def tokenize(question=None, answer=None, use_pad=False, include_rev=False):
    """Converts text to tokens.

    Args:
        question: str or None, the question as a string.
        answer: str or None, the answer as a string.
        use_pad: bool, if set, pad to a specific length with end tokens.
        include_rev: bool, if set, include the "reverse" dictionary.

    Returns:
        tok_questions: list of ints, the tokenized question.
        tok_answers: list of ints, the tokenized answer.
        question_len: int, the number of tokens in the question.
        answer_len: int, the number of tokens in the answer.
        if include_rev:
            rev_dict: dictionary of int -> str pairs, for reversing from
                the answer to the question.
    """

    if question is None:
        question = ''

    tok_questions = word_tokenize(question)[:QUESTION_MAXLEN-1]
    tok_answers = word_tokenize(answer)[:ANSWER_MAXLEN-1]
    del question, answer

    idxs = dict((c, i + 3) for i, c in enumerate(tok_answers))

    def _encode(w):
        if w in _CHAR_TO_IDX:
            return _CHAR_TO_IDX[w]
        if w in idxs:
            return idxs[w]
        return OUT_OF_VOCAB

    enc_questions = [_encode(w) for w in tok_questions] + [END_IDX]
    enc_answers = [_encode(w) for w in tok_answers] + [END_IDX]
    del tok_questions

    question_len = len(enc_questions)
    answer_len = len(enc_answers)

    if use_pad:
        enc_questions += [END_IDX] * (QUESTION_MAXLEN - len(enc_questions))
        enc_answers += [END_IDX] * (ANSWER_MAXLEN - len(enc_answers))

    if include_rev:
        rev_dict = dict((i + 3, c) for i, c in enumerate(tok_answers))
        del tok_answers
        return enc_questions, enc_answers, question_len, answer_len, rev_dict
    else:
        del tok_answers
        return enc_questions, enc_answers, question_len, answer_len


def detokenize(tokens, rev_dict, argmax=False, show_missing=False):
    """Converts question back to tokens.

    Args:
        tokens: list of ints, the tokens to be reversed.
        rev_dict: the reverse dictionary (provided by the answer text).
        argmax: bool, if set, take argmax over last dimension of tokens.
        show_missing: bool, if set, use 'X' to represent missing tokens.
    """

    if argmax:
        tokens = np.argmax(tokens, axis=-1)

    def _decode(i):
        if i >= NUM_SPECIAL:
            return _DICTIONARY[i - NUM_SPECIAL]
        elif i in rev_dict:
            return '%s(%d)' % (rev_dict[i], i)
        else:
            return 'X' if show_missing else ''

    words = [_decode(w) for w in tokens]
    sentence = ' '.join(w for w in words if w)

    return sentence


def get_word_embeddings(num_dimensions=500,
                        cache_loc=EMBEDDINGS_FILE):
    """Generates word embeddings.

    Args:
        num_dimensions: int, number of embedding dimensions.
        cache_loc: str, where to cache the word embeddings.

    Returns:
        numpy array representing the embeddings, with shape (NUM_TOKENS,
            num_dimensions).
    """

    if os.path.exists(cache_loc):
        embeddings = np.load(cache_loc)
    else:
        class SentenceGenerator(object):
            def __iter__(self):
                for i, (question, answer) in enumerate(iterate_qa_pairs(), 1):
                    q, a, _, _ = tokenize(question=question, answer=answer,
                                          use_pad=False, include_rev=False)
                    yield [str(w) for w in q]
                    yield [str(w) for w in a]

                    del q, a, w

                    if i % 1000 == 0:
                        sys.stderr.write('\rprocessed %d' % i)
                        sys.stderr.flush()

                sys.stderr.write('\rprocessed %d\n' % i)
                sys.stderr.flush()

        # The default embeddings.
        embeddings = np.random.normal(size=(NUM_TOKENS, num_dimensions))

        sentences = SentenceGenerator()
        model = models.Word2Vec(sentences, size=num_dimensions)

        word_vectors = model.wv
        del model

        # Puts the Word2Vec weights into the right order.
        weights = word_vectors.syn0
        vocab = word_vectors.vocab
        for k, v in vocab.items():
            embeddings[int(k)] = weights[v.index]

        with open(cache_loc, 'wb') as f:
            np.save(f, embeddings)
            pass

    assert embeddings.shape == (NUM_TOKENS, num_dimensions)
    return embeddings


def iterate_qa_pairs(num_iter=None):
    """Iterates through question-answer pairs in a single file.

    Args:
        num_iter: int (default: None), number of times to iterate. If None,
            iterates infinitely.

    Yields:
        subject: the question title (max length = QUESTION_TITLE_MAXLEN)
        bestanswer: the body of the best answer
            (max length = ANSWER_MAXLEN)
    """

    def _parse_document(elem):
        subject = elem.find('subject')
        bestanswer = elem.find('bestanswer')

        return ('' if subject is None else subject.text,
                '' if bestanswer is None else bestanswer.text)

    if num_iter is None:
        iterator = itertools.count()
    else:
        iterator = xrange(num_iter)

    for _ in iterator:
        with open(DATA_PATH, 'r') as f:
            parser = ET.iterparse(f)
            for event, elem in parser:
                if elem.tag == 'document':
                    yield _parse_document(elem)
                    elem.clear()  # Important for avoiding memory issues.


def iterate_answer_to_question(batch_size, include_ref, num_iter=None):
    """Yields Numpy arrays, representing the data."""

    q_toks, a_toks, q_lens, a_lens, refs = [], [], [], [], []

    for i, (question, answer) in enumerate(iterate_qa_pairs(num_iter), 1):
        args = tokenize(
            question=question, answer=answer,
            use_pad=True, include_rev=include_ref)

        q_toks.append(args[0])
        a_toks.append(args[1])
        q_lens.append(args[2])
        a_lens.append(args[3])

        if include_ref:
            refs.append(args[4])

        if i % batch_size == 0:
            r = [np.asarray(q_toks), np.asarray(a_toks),
                 np.asarray(q_lens), np.asarray(a_lens)]

            if include_ref:
                r.append(refs)

            yield r

            q_toks, a_toks, q_lens, a_lens, refs = [], [], [], [], []


if __name__ == '__main__':  # Scropt for building the embeddings.
    emb = get_word_embeddings()
    print('emb:', emb.shape)
