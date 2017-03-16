from __future__ import absolute_import

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
        # if tag == 'vespaadd':
        #     print ''

    def endElement(self, tag):
        if tag == 'document':

            if self.i % 1000 == 0:
                logging.info('Indexed document %d', self.i)

            if self.i < self.max_docs:
                self.indexing.indexing(self.subject,
                                       self.content,
                                       self.bestAnswer)
                self.tag = ''
                self.subject = u''
                self.content = u''
                self.bestAnswer = u''
                self.i = self.i + 1
            else:
                Handler.indexing.closeWriter()
                Handler.indexing.closeIndexing()

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
        query = 'query'
        while query:
            candidates = handler.indexing.get_top_n(query, limit=500)
            print ShallowRank(1, query, candidates).shallowRank()
            query = raw_input('Enter your question [None to quit]:\n')
        handler.indexing.closeIndexing()
