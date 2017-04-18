"""Defines some common utils for building APIs.

The main feature of these APIs is that they are multi-threaded; in the
background, there are a number of threads handling requests that maintain
resources across calls (so that resources don't have to continually be
reloaded).

In particular, a QueryJob holds the input (for example, a question), gets put
on a queue, processed, and triggers a flag to tell the calling process that
it finished. This behavior is useful because it allows the user to choose the
number of processing threads.
"""

import logging
import threading

# Defines constants.
MAX_JOBS = 10  # Maximum number of jobs that can be on the queue at once.
SHALLOW_LIMIT = 100  # Max number of candidates to pass to shallow ranker.
MAX_WAIT_TIME = 100  # Maximum number of seconds to wait for a query.
NUM_THREADS = 1  # Number of background threads to run at once.


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
        self._flag = flag
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
            raise IOError('This query job has not yet been processed.')

        return self._response

    def set_response(self, response):
        self._response = response

    def set_flag(self):
        self._flag.set()


def build_qa_funcs(process_thread_func, query_queue):
    """Builds get_question and get_answer functions.

    This starts background threads for processing the queries as they are
    passed, then returns functions for adding and removing from the process
    queue.

    Args:
        process_thread_func: a function to call (with no arguments) that
            processes queries on the query queue.
        query_queue: a queue that holds incoming queries.

    Returns:
        (get_question, get_answer), functions that will add questions and
            answers to the query queue, block until they complete, then return
            them to the main thread.
    """

    for i in range(NUM_THREADS):
        logging.info('Creating processing thread %d' % i)
        t = threading.Thread(name='process_thread_%d' % i,
                             target=process_thread_func)
        t.setDaemon(True)
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

    return get_question, get_answer
