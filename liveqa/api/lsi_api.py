"""Defines callable functions for the API.

To use, simply import the API call you'd like to issue. The model processing
is handled in a background thread.

To experiment with the command-line utility, use:
    python -m liveqa.api.command_line

To use the get_question and get_answer methods, use:
    from api import get_question, get_answer
"""

from __future__ import division
from __future__ import absolute_import

import xml.sax

from liveqa.rank.shallow_rank import ShallowRank
from liveqa.vertical.indexing import Indexing
from liveqa.vertical.xml_parsing import XmlParseHandler

import threading
from Queue import Queue
from Queue import Empty as QueueEmptyException

import logging

logging.info('By importing the API submodule, you are creating background '
             'processes on different threads, which handle calls to the '
             'ranking components.')

# Defines constants.
MAX_JOBS = 10  # Maximum number of jobs that can be on the queue at once.
SHALLOW_LIMIT = 100  # Max number of candidates to pass to shallow ranker.
MAX_WAIT_TIME = 20  # Maximum number of seconds to wait for a query.
NUM_THREADS = 1  # Number of background threads to run at once.

# Creates the main events queue that handles inter-thread communication.
query_queue = Queue(maxsize=MAX_JOBS)


class QueryJob(object):
    """Container to hold a query job, which the background threads process."""

    def __init__(self, query, query_type, flag):
        """Creates a new query job.

        Args:
            query: str, the query string.
            query_type: either "question" (if you are requesting similar
                questions to the provided query) or "answer" (if you are
                requesting similar answers to the provided query).
            flag: a threading flag that blocks until the job is completed.
        """

        query_type = query_type.lower()
        assert query_type in ['question', 'answer']

        self._query = query
        self._query_type = query_type
        self._flag
        self._response = None

    @property
    def query(self):
        return self._query

    @property
    def query_type(self):
        return self._query_type

    @property
    def response(self):
        if self._response is None:
            raise ValueError('This query job has not yet been processed.')

        return self._response

    def set_response(self, response):
        self._response = response

    def set_flag(self):
        self._flag.set()


def process_thread(e):
    """Thread that processes queries in the background.

    Args:
        e: a Threading event that is set once everything is ready to go.
    """

    _query_queue = Queue(maxsize=MAX_JOBS)
    _ranker = ShallowRanker()  # This is the LSI ranker.

    _handler = XmlParseHandler()
    _handler.indexing.turnOnReadMode()  # Only use handler in read mode.

    _parser = xml.sax.make_parser()
    _parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    _parser.setContentHandler(handler)

    e.set()  # Tells main thread that everything is read to go.

    while True:
        query_job = query_queue.get()  # Blocks until a new question appears.

        if query_type == 'question':
            candidates = _handler.indexing.get_top_n_questions(
                query, limit=SHALLOW_LIMIT)
            question = _ranker.get_candidates(query, candidates, nc=1)[0]
            query_job.set_response(question)
            query_job.set_flag()

        elif query_job.query_type == 'answer':
            candidates = _handler.indexing.get_top_n_answers(
                query, limit=SHALLOW_LIMIT)
            answer = _ranker.get_candidates(query, candidates, nc=1)[0]
            query_job.set_response(answer)
            query_job.set_flag()

        else:
            raise ValueError('Invalid query type: "%s"' % query_job.query_type)

for i in range(NUM_THREADS):
    flag = threading.Event()
    t = threading.Thread(name='process_thread_%d' % i,
                         target=process_thread,
                         args=(flag,))
    t.setDaemon(True)  # Dies when the main thread dies.
    t.start()
    flag.wait()  # Waits for the thread to get up and running.


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
    event.wait(timeout=MAX_WAIT_TIME)  # Wait for the job to finish.

    return query_queue.response


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
