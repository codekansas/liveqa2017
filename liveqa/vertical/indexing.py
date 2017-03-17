from unidecode import unidecode
from whoosh.analysis import StemmingAnalyzer
from whoosh.index import create_in, open_dir, exists_in
from whoosh.writing import AsyncWriter
from whoosh.fields import *
from whoosh.qparser import QueryParser, MultifieldParser, syntax

import os.path

import logging

BASE = os.path.dirname(os.path.realpath(__file__))
INDEX_DIRECTORY = os.path.join(BASE, 'index/')


class Indexing(object):
    def __init__(self, directory=INDEX_DIRECTORY):
        """Creates an Indexing object to communicate with Woosh.

        Args:
            directory: str, where to index files (defaults to INDEX_DIRECTORY).
        """

        # self.schema = self.getSchema()
        # if not os.path.exists("indexdir"):
        #     print ("directory not exist")
        #     os.mkdir("indexdir")
        #     self.ix = create_in("indexdir", self.schema)
        # self.ix = open_dir("indexdir")
        # self.writer = AsyncWriter(self.ix)
        # self.writer = self.ix.writer()
        # self.ix.reader()
        self.directory = directory
        self.isWriteModeOn = False
        self.isReadModeOn = False
        self.turnOnWriteMode()
        # self.turnOnReadMode()

    def indexing(self, subject, content, bestAnswer):
        # title = subject
        # body = content
        # ba = bestAnswer
        if self.isWriteModeOn:
            # exists = exists_in("indexdir")
            # if not exists:

            self.writer.add_document(title=subject,
                                     body=content,
                                     ba=bestAnswer)
        # self.writer.commit()

    def getSchema(self):
        return Schema(title=TEXT(stored=True),
                      body=TEXT(analyzer=StemmingAnalyzer()),
                      ba=TEXT(stored=True))

    def closeWriter(self):
        self.writer.commit()

    def closeIndexing(self):
        self.ix.close()

    def turnOnReadMode(self):
        self.isReadModeOn = True
        self.isWriteModeOn = False
        self.schema = self.getSchema()
        self.ix = open_dir(self.directory)
        self.ix.reader()

    def turnOnWriteMode(self):
        self.isWriteModeOn = True
        self.isReadModeOn = False
        self.schema = self.getSchema()
        if not os.path.exists(self.directory):
            logging.info("directory does not exist")
            os.mkdir(self.directory)
            self.ix = create_in(self.directory, self.schema)
        self.ix = open_dir(self.directory)
        # self.writer = AsyncWriter(self.ix)
        self.writer = self.ix.writer()

    # def getIsNeedParse(self):
    #     return

    def clean(self, text):
        """Cleans text before returning it to the user.

        Args:
            text: str, the text to clean.

        Returns:
            string, the cleaned text.
        """

        text = text.replace('\n', ' ')

        return text

    def get_top_n_questions(self, query, limit=10):
        """Returns the top questions related to a given query.

        Args:
            query: str, the query to parse.
            limit: int, the maximum number of documents to return.

        Returns:
            list of strings, the top results for the given query.
        """

        logging.info('query: %s', query)
        self.query = MultifieldParser(['title', 'body', 'ba'],
                                      schema=self.ix.schema,
                                      group=syntax.OrGroup).parse(query)

        searcher = self.ix.searcher()
        results = searcher.search(self.query, limit=limit)

        # Cleans the retrieved results.
        results = [self.clean(result.get('title')) for result in results]

        return results

    def get_top_n_answers(self, query, limit=100):
        """Returns the top results for a given query.

        Args:
            query: str, the query to parse.
            limit: int, the maximum number of documents to return.

        Returns:
            list of strings, the top results for the given query.
        """

        logging.info('query: %s', query)
        self.query = QueryParser('title',
                                 schema=self.ix.schema,
                                 group=syntax.OrGroup).parse(query)

        searcher = self.ix.searcher()
        results = searcher.search(self.query, limit=limit)

        # Cleans the provided results.
        results = [self.clean(result.get('ba')) for result in results]

        return results
