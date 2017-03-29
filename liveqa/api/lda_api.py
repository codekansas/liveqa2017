"""Defines callable functions for the API.

To use, simply import the API call you'd like to issue. The model processing
is handled in a background thread.

To experiment with the command-line utility, use:
    python -m liveqa.api.command_line

To use the get_question and get_answer methods, use:
    from liveqa.api import get_question, get_answer

This is also a useful reference for future APIs.
"""

from __future__ import absolute_import
from __future__ import division

import xml.sax
import random

from liveqa.rank.shallow_rank import ShallowRank
from liveqa.vertical.indexing import Indexing
from liveqa.vertical.xml_parsing import XmlParseHandler

# Imports everything from the API utils.
from liveqa.api.api_utils import *

import threading
from Queue import Queue

import logging

logging.info('By importing the API submodule, you are creating background '
             'processes on different threads, which handle calls to the '
             'ranking components.')

# Creates the main events queue that handles inter-thread communication.
query_queue = Queue(maxsize=MAX_JOBS)

logging.info('Initializing ShallowRank object...')
_ranker = ShallowRank()  # This is the LSI ranker.

logging.info('Initializing Whoosh parser...')
_handler = XmlParseHandler(index_mode='read')

_parser = xml.sax.make_parser()
_parser.setFeature(xml.sax.handler.feature_namespaces, 0)
_parser.setContentHandler(_handler)


def process_thread():
    """Thread that processes queries in the background."""

    while True:
        query_job = query_queue.get()  # Blocks until a new question appears.

        if query_job.query_type == 'question':
            candidates = _handler.indexing.get_top_n_questions(
                query_job.query, limit=SHALLOW_LIMIT)
            if candidates:
                question = _ranker.get_candidates(
                    query_job.query, candidates, nc=1)[0]
                # question = random.choice(candidates)
                query_job.set_response(question)
            else:
                query_job.set_response('No questions found.')
            query_job.set_flag()

        elif query_job.query_type == 'answer':
            candidates = _handler.indexing.get_top_n_answers(
                query_job.query, limit=SHALLOW_LIMIT)
            if candidates:
                answers = _ranker.get_candidates(
                    query_job.query, candidates, nc=1)
                query_job.set_response(answers[0])
            else:
                query_job.set_response('No answers found.')
            query_job.set_flag()

        else:
            raise ValueError('Invalid query type: "%s"'
                             % query_job.query_type)

for i in range(NUM_THREADS):
    logging.info('Creating processing thread %d' % i)
    t = threading.Thread(name='process_thread_%d' % i,
                         target=process_thread)
    t.setDaemon(True)  # Dies when the main thread dies.
    t.start()


def get_response(text, query_type):
    """Gets a question or answer.

    Args:
        text: str, the reference text to use.
        query_type: str, either "question" or "answer".

    Returns:
        the most relevant response, as a string.
    """

    if query_queue.full():
        raise RuntimeError('The answer queue currently has more than %d jobs, '
                           'and cannot take any more.')

    flag = threading.Event()
    query_job = QueryJob(query=text, query_type=query_type, flag=flag)

    query_queue.put(query_job)
    flag.wait(timeout=MAX_WAIT_TIME)  # Wait for the job to finish.

    try:
        response = query_job.response
        if isinstance(response, (list, tuple)):
            return str(response[0])
        else:
            return str(response)
    except IOError:
        return 'The query job timed out.'


def get_question(text):
    """Gets the most relevant question for the provided text.

    Args:
        text: str, the reference text to use.

    Returns:
        question: a string, the most relevant question for the provided text.
    """

    return get_response(text, query_type='question')


def get_answer(text):
    """Gets the most relevant answer for the provided text.

    Args:
        text: str, the reference text to use.

    Returns:
        answer: a string, the most relevant answer for the provided text.
    """

    return get_response(text, query_type='answer')
