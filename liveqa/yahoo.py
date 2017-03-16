"""
Code for reading from Yahoo L6 data.

The raw data is in XML format. These utils iterate through this data and
return question-answer pairs. This provides a way to standardize access to the
corpus.
"""

from __future__ import absolute_import
from __future__ import print_function

import cPickle as pkl
import itertools
import os
import sys
import re
import xml.etree.cElementTree as ET

import numpy as np

# Constants.
QUESTION_TITLE_MAXLEN = 140
QUESTION_BODY_MAXLEN = 500
ANSWER_MAXLEN = 250
DATA_ENV_NAME = 'YAHOO_DATA'  # Directory containing the Yahoo data, unzipped.
YAHOO_L6_URL = 'http://webscope.sandbox.yahoo.com/catalog.php?datatype=l'

# Variables that will be filled in later.
BASE = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'FullOct2007.xml')

if not os.path.exists(DATA_PATH):
    raise RuntimeError('File not found: "%s". To create it from the existing '
                       'files, run:  cat FullOct2007.xml.part1 '
                       'FullOct2007.xml.part2 > FullOct2007.xml' % DATA_PATH)

# Extra tokens:
# PADDING = 0
OUT_OF_VOCAB = 0
NUM_EXTRA = 1
TEXT = ('@#abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '1234567890'
        '?.!$%^&*()/\\|:\'"-_=+ ')
START_TOKEN, END_TOKEN = TEXT[0], TEXT[1]
START_IDX, END_IDX = NUM_EXTRA, NUM_EXTRA + 1
TOKENS = dict((c, i + NUM_EXTRA) for i, c in enumerate(TEXT))
NUM_TOKENS = len(TOKENS) + NUM_EXTRA


def tokenize(text, pad_len=None):
    """Converts text to tokens."""

    idxs = [TOKENS.get(c, OUT_OF_VOCAB) for c in text]

    if pad_len is not None:
        idxs = idxs[:pad_len]
        idxs += [END_IDX] * (pad_len - len(idxs))

    return idxs


def detokenize(tokens, argmax=False):
    """Converts tokens to text."""

    if argmax:
        tokens = np.argmax(tokens, axis=-1)

    # Remove end indices.
    tokens = [t for t in tokens if t != END_IDX]

    return ''.join(['' if t < NUM_EXTRA else TEXT[t-NUM_EXTRA]
                    for t in tokens])


def clean_text(text):
    """Cleans text of HTML, double spaces, etc."""

    text = re.sub('(\<.+?\>|\n|  +)+', ' ', text)
    text = re.sub(START_TOKEN + '|' + END_TOKEN, '', text)
    # text = START_TOKEN + text + END_TOKEN
    return text


def iterate_qa_pairs(convert_to_tokens=True):
    """Iterates through question-answer pairs in a single file.

    Args:
        convert_to_tokens: bool, if set, convert characters to tokens.

    Yields:
        subject: the question title (max length = QUESTION_TITLE_MAXLEN)
        content: the question body (max length = QUESTION_BODY_MAXLEN)
        bestanswer: the body of the best answer
            (max length = ANSWER_MAXLEN)
    """

    def _parse_document(elem):
        subject = elem.find('subject')
        content = elem.find('content')
        bestanswer = elem.find('bestanswer')

        subject = clean_text('' if subject is None else subject.text)
        content = clean_text('' if content is None else content.text)
        bestanswer = clean_text('' if bestanswer is None
                                else bestanswer.text)

        subject_len = min(len(subject) + 1, QUESTION_TITLE_MAXLEN)
        content_len = min(len(content) + 1, QUESTION_BODY_MAXLEN)
        bestanswer_len = min(len(bestanswer) + 1, ANSWER_MAXLEN)

        if convert_to_tokens:
            subject = tokenize(subject, pad_len=QUESTION_TITLE_MAXLEN)
            content = tokenize(content, pad_len=QUESTION_BODY_MAXLEN)
            bestanswer = tokenize(bestanswer, pad_len=ANSWER_MAXLEN)

        return (subject, content, bestanswer,
                subject_len, content_len, bestanswer_len)

    with open(DATA_PATH, 'r') as f:
        parser = ET.iterparse(f)
        for event, elem in parser:
            if elem.tag == 'document':
                yield _parse_document(elem)


def iterate_answer_to_question(batch_size):
    """Iterator for producing answer -> question data.

    Args:
        batch_size: int, the number of samples per batch.

    Yields:
        bestanswer: numpy array with shape
            (batch_size, ANSWER_MAXLEN)
        subject: numpy array with shape
            (batch_size, QUESTION_TITLE_MAXLEN)
    """

    iterable = itertools.cycle(iterate_qa_pairs())

    qtitles, abodies, qlens, alens = [], [], [], []

    for i, (qtitle, _, abody, qlen, _, alen) in enumerate(iterable, 1):
        qtitles.append(qtitle)
        abodies.append(abody)
        alens.append(alen)
        qlens.append(qlen)

        if i % batch_size == 0:
            yield (np.asarray(qtitles),
                   np.asarray(abodies),
                   np.asarray(qlens),
                   np.asarray(alens))
            qtitles, abodies, qlens, alens = [], [], [], []
