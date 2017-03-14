"""Serving code for the question generator model.

To run a local server:
    python -m liveqa.nn.generator.serve
"""

from __future__ import division

from .model import QuestionGenerator
from .. import yahoo

import atexit
import sys
import threading

from Queue import Queue
from Queue import Empty as QueueEmptyException

from flask import Flask
from flask import abort
from flask import jsonify

import tensorflow as tf
import numpy as np

MAX_JOBS = 1000  # Maximum number of jobs on the queue.
BATCH_SIZE = 100  # Maximum number of jobs to process at once.
MAX_TF_WAIT = 100 / 1000  # Max wait time to fill a batch.
MAX_PROC_WAIT = 5000 / 1000  # Max wait time for a question to be generated.
NUM_TF_THREADS = 1  # Number of threads to spawn to handle questions.
LOGDIR = 'model/'  # Where the model is stored.

answer_queue = Queue(maxsize=MAX_JOBS)  # Answers model has to process.


def create_app():
    app = Flask(__name__)

    def question_generator(e):
        """Processes answers in answer queue.

        Args:
            e: a threading event that is set once the model is set up.
        """

        sess = tf.Session()
        model = QuestionGenerator(sess,
                                  yahoo.ANSWER_MAXLEN,
                                  yahoo.QUESTION_TITLE_MAXLEN,
                                  yahoo.NUM_TOKENS,
                                  logdir=LOGDIR,
                                  only_cpu=True)
        model.load(ignore_missing=False)  # Model must already exist.

        e.set()  # Tells the parent process that the model is ready.

        while True:
            # Block until the first thing shows up on the answer queue.
            pairs = [answer_queue.get(block=True)]

            # Wait MAX_TF_WAIT to fill up BATCH_SIZE events.
            while len(pairs) < BATCH_SIZE:
                try:
                    factor = (BATCH_SIZE - len(pairs)) / BATCH_SIZE
                    timeout = MAX_TF_WAIT * factor
                    pairs.append(answer_queue.get(block=True, timeout=timeout))
                except QueueEmptyException:
                    break

            events, answers = zip(*pairs)

            # Converts to a numpy
            answers_arr = np.asarray([a[0] for a in answers])
            answer_lens = np.asarray([a[1] for a in answers])

            # Runs the model to generate questions.
            questions = model.sample(answers_arr, answer_lens, 0.5)

            for e, a, q in zip(events, answers, questions):
                a.append(q)
                e.set()  # Signals that the model has been processed.

    for i in range(NUM_TF_THREADS):
        e = threading.Event()
        t = threading.Thread(name='tf_%d' % i,
                             target=question_generator, args=(e,))
        t.setDaemon(True)
        t.start()
        e.wait()  # Wait for the TF thread to get ready.

    def get_question(answer):
        if answer_queue.full():
            raise RuntimeError('Answer queue is full.')

        # Turns the answer into the model's input.
        answer = yahoo.clean_text(answer)
        answer_len = min(len(answer) + 1, yahoo.ANSWER_MAXLEN)
        answer = yahoo.tokenize(answer, pad_len=yahoo.ANSWER_MAXLEN)
        event = threading.Event()

        # The question generator will append the generated question to this
        # array (which, since this is Python, is mutable).
        answer_mutable = [answer, answer_len]
        answer_queue.put((event, answer_mutable))

        # Waits for the answer to be processed.
        event.wait(timeout=MAX_PROC_WAIT)

        if len(answer_mutable) != 3:
            raise RuntimeError('The event timed out before the '
                               'process could finish.')

        # Decodes the answer.
        question = answer_mutable[2]
        question = yahoo.detokenize(question, argmax=True)

        return question

    @app.route('/')
    def index():
        """Defines the homepage."""

        return '''
            <html>
                <head></head>
                <body>
                    <h1>Question Generation API</h1>
                    <p>To use this API:</p>
                    <p><code>/api/sample/[answer]</code></p>
                    <p>Example:</p>
                    <p>
                        <code>
                            <a href="/api/sample/this%20is%20an%20answer">
                                /api/sample/this%20is%20an%20answer
                            </a>
                        </code></p>
                    <p>Returns JSON:</p>
                    <p>
                        <code>
                        {
                            'question': 'what is this question?'
                        }
                        </code>
                    </p>
                <body>
            </html>'''

    @app.route('/api/sample/<string:answer>', methods=['GET'])
    def sample(answer):
        """Samples from the model."""

        try:
            question = get_question(answer)
        except RuntimeError, e:
            abort(503)

        return jsonify({'question': question})

    return app


app = create_app()

if __name__ == '__main__':
    app.run()
