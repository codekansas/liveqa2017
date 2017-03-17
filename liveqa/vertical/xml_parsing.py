"""This script provides access to the ranking functions.

To run the demo script (which first queries the Whoosh index, then does
fine-grain filtering using the LDA index in shallow_rank.py), run:
    python -m liveqa.vertical.xml_parsing

When READ_MODE=True, this will provide an interactive console to ask questions
and get responses from the model.

To train the model, set the READ_MODE variable below to False. This process
takes a while to run, since it has to build the index for 4 million documents.
It is probably easier to get this from someone else.
"""

from __future__ import absolute_import
from __future__ import print_function

import logging
import xml.sax

from liveqa import yahoo

import sys

from liveqa.rank.shallow_rank import ShallowRank
from liveqa.vertical.indexing import Indexing

MAX_DOCS = 10000000
READ_MODE = True


class XmlParseHandler(xml.sax.ContentHandler):

    def __init__(self, max_docs=MAX_DOCS):
        """Creates an XML parse handler.

        Args:
            max_docs: int, maximum number of docs to index.
        """

        self.tag = ''
        self.subject = ''
        self.content = ''
        self.bestAnswer = ''
        self.indexing = Indexing()
        self.max_docs = max_docs
        self.i = 0

    def startElement(self, tag, attrs):
        self.tag = tag

    def endElement(self, tag):
        if tag == 'document':
            if self.num_docs % 1000 == 0:
                logging.info('Indexed document %d', self.num_docs)

            self.indexing.indexing(self.subject,
                                   self.content,
                                   self.bestAnswer)
            self.tag = ''
            self.subject = u''
            self.content = u''
            self.bestAnswer = u''
            self.num_docs = self.num_docs + 1

    def characters(self, content):
        if self.tag == 'subject':
            self.subject += content
        elif self.tag == 'content':
            self.content += content
        elif self.tag == 'bestanswer':
            self.bestAnswer += content

if (__name__ == '__main__'):

    # Turn on logging.
    logging.basicConfig(level=logging.DEBUG)

    # Creates an XMLReader.
    parser = xml.sax.make_parser()

    # Turns off namespaces.
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # Overrides the default ContextHandler.
    handler = XmlParseHandler()
    parser.setContentHandler(handler)

    if READ_MODE:
        handler.indexing.turnOnReadMode()

    if handler.indexing.isWriteModeOn:
        x = raw_input('About to index Yahoo L6 data. Proceed? [y/N] ')
        if x[0].lower() != 'y':
            raise ValueError('You chose not to start indexing.')

        parser.parse(yahoo.DATA_PATH)
        handler.indexing.closeWriter()
        handler.indexing.closeIndexing()
    else:
        ranker = ShallowRank()
        query = raw_input('Enter your question [None to quit]:\n')

        while query:

            # Gets candidate answers.
            candidates = handler.indexing.get_top_n_answers(query, limit=100)
            answers = ranker.get_candidates(query, candidates, nc=10)

            # Gets candidate questions.
            candidates = handler.indexing.get_top_n_questions(query, limit=100)
            questions = ranker.get_candidates(query, candidates, nc=10)

            print('Top %d Answers, sorted by relevance:' % len(answers))
            for i, answer in enumerate(answers):
                print('%d.' % (i + 1), answer)

            print('Top %d Questions, sorted by relevance:' % len(questions))
            for i, question in enumerate(questions):
                print('%d.' % (i + 1), question)

            query = raw_input('Enter another question [None to quit]:\n')

        handler.indexing.closeIndexing()
