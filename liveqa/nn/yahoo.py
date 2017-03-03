"""model with the right shapes
Code for reading from Yahoo L6 data.

The raw data is in XML format. These utils iterate through this
data and returns question-answer pairs.
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
QUESTION_BODY_MAXLEN = 4000
ANSWER_MAXLEN = 4000
DATA_ENV_NAME = 'YAHOO_DATA'  # Directory containing the Yahoo data, unzipped.
YAHOO_L6_URL = 'http://webscope.sandbox.yahoo.com/catalog.php?datatype=l'

# Variables that will be filled in later.
DATA_PATH = None  # Path to the XML file with the data.


if DATA_ENV_NAME not in os.environ:
    raise RuntimeError('The environment variable "%s" was not found. You '
                       'should set it to point at the directory with the '
                       'files downloaded from [%s].'
                       % (DATA_ENV_NAME, YAHOO_L6_URL))

DATA_PATH = os.path.join(os.environ[DATA_ENV_NAME], 'FullOct2007.xml')

if not os.path.exists(DATA_PATH):
    raise RuntimeError('File not found: "%s". To create it from the existing '
                       'files, run:  cat FullOct2007.xml.part1 '
                       'FullOct2007.xml.part2 > FullOct2007.xml' % DATA_PATH)

TEXT = ('abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '1234567890'
        '!@#$%^&*()\\|\'"-_=+')
TOKENS = dict((c, i + 2) for i, c in enumerate(TEXT))
NUM_TOKENS = len(TOKENS) + 2


def tokenize(text, pad_len=None):
    """Converts text to tokens."""

    idxs = [TOKENS.get(c, 1) for c in text]

    if pad_len is not None:
        idxs = idxs[:pad_len]
        idxs += [0] * (pad_len - len(idxs))

    return idxs


def detokenize(tokens):
    """Converts tokens to text."""

    return ''.join(['' if t < 2 else TEXT[t-2] for t in tokens])


def clean_text(text):
    """Cleans text of HTML, double spaces, etc."""

    text = re.sub('(\<.+?\>|\n|  +)+', ' ', text)

    return text


def iterate_qa_pairs(convert_to_tokens=True):
    """Iterates through question-answer pairs in a single file.

    Args:
        convert_to_tokens: bool, if set, convert characters to tokens.

    Yields:
        subject: the question title (max length = QUESTION_TITLE_MAXLEN)
        content: the question body (max length = QUESTION_BODY_MAXLEN)
        bestanswer: the body of the best answer (max length = ANSWER_MAXLEN)
    """

    def _parse_document(elem):
        subject = elem.find('subject')
        content = elem.find('content')
        bestanswer = elem.find('bestanswer')

        subject = clean_text('' if subject is None else subject.text)
        content = clean_text('' if content is None else content.text)
        bestanswer = clean_text('' if bestanswer is None else bestanswer.text)

        if convert_to_tokens:
            subject = tokenize(subject, pad_len=QUESTION_TITLE_MAXLEN)
            content = tokenize(content, pad_len=QUESTION_BODY_MAXLEN)
            bestanswer = tokenize(bestanswer, pad_len=ANSWER_MAXLEN)

        return subject, content, bestanswer

    with open(DATA_PATH, 'r') as f:
        parser = ET.iterparse(f)
        for event, elem in parser:
            if elem.tag == 'document':
                yield _parse_document(elem)


def iterate_qa_data(batch_size):
    """Iterates through question-answer data as Numpy arrays.

    Args:
        batch_size: int, the number of samples per batch.

    Yields:
        subject: numpy array with shape (batch_size, QUESTION_TITLE_MAXLEN)
        content: numpy array with shape (batch_size, QUESTION_BODY_MAXLEN)
        bestanswer: numpy array with shape (batch_size, ANSWER_MAXLEN)
    """

    iterable = itertools.cycle(iterate_qa_pairs())

    qtitles, qbodies, abodies = [], [], []
    counter = 0

    for qtitle, qbody, abody in iterable:
        qtitles.append(qtitle)
        qbodies.append(qbody)
        abodies.append(abody)

        counter += 1
        if counter == batch_size:
            counter = 0
            yield np.asarray(qtitles), np.asarray(qbodies), np.asarray(abodies)
            qtitles, qbodies, abodies = [], [], []
