from unidecode import unidecode
from whoosh.analysis import StemmingAnalyzer
from whoosh.index import create_in, open_dir, exists_in
from whoosh.writing import AsyncWriter
from whoosh.fields import *
from whoosh.qparser import QueryParser
import os.path

class Indexing(object):
    def __init__(self):

        self.schema = self.getSchema()
        if not os.path.exists("indexdir"):
            os.mkdir("indexdir")
            self.ix = create_in("indexdir", self.schema)
        self.ix = open_dir("indexdir")
        # self.writer = AsyncWriter(self.ix)
        self.writer = self.ix.writer()

    def indexing(self, subject, content, bestAnswer):
        # title = subject
        # body = content
        # ba = bestAnswer
        exists = exists_in("indexdir")
        if not exists:
            self.writer.add_document(title=subject, body=content, ba=bestAnswer)
        # self.writer.commit()

    def getSchema(self):
        return Schema(title = TEXT(stored=True),body = TEXT(analyzer=StemmingAnalyzer()), ba = TEXT(stored=True))

    def closeWriter(self):
        self.writer.commit()

    def searcher(self, query):

        # query = QueryParser("content", self.ix.schema).parse("Why are yawns contagious")
        self.query = QueryParser("ba", self.ix.schema).parse(query)

        try:
            searcher = self.ix.searcher()
            results = searcher.search(self.query, limit = 10)
            # results[0]
            for result in results:
                print result
            return [result for result in results]
            # print(results[0])

        finally:
            searcher.close()