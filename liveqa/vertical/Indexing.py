from unidecode import unidecode
from whoosh.analysis import StemmingAnalyzer
from whoosh.index import create_in, open_dir, exists_in
from whoosh.writing import AsyncWriter
from whoosh.fields import *
from whoosh.qparser import QueryParser, syntax
import os.path


class Indexing(object):
    def __init__(self, dir="/Users/Q/PycharmProjects/liveqa2017/liveqa/vertical/indexdir/"):

        # self.schema = self.getSchema()
        # if not os.path.exists("indexdir"):
        #     print ("directory not exist")
        #     os.mkdir("indexdir")
        #     self.ix = create_in("indexdir", self.schema)
        # self.ix = open_dir("indexdir")
        # self.writer = AsyncWriter(self.ix)
        # self.writer = self.ix.writer()
        # self.ix.reader()
        self.dir = dir
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

            self.writer.add_document(title=subject, body=content, ba=bestAnswer)
        # self.writer.commit()

    def getSchema(self):
        return Schema(title = TEXT(stored=True),body = TEXT(analyzer=StemmingAnalyzer()), ba = TEXT(stored=True))

    def closeWriter(self):
        self.writer.commit()

    def closeIndexing(self):
        self.ix.close()

    def turnOnReadMode(self):
        self.isReadModeOn = True
        self.isWriteModeOn = False
        self.schema = self.getSchema()
        self.ix = open_dir(self.dir)
        self.ix.reader()

    def turnOnWriteMode(self):
        self.isWriteModeOn = True
        self.isReadModeOn = False
        self.schema = self.getSchema()
        if not os.path.exists(self.dir):
            print ("directory not exist")
            os.mkdir(self.dir)
            self.ix = create_in(self.dir, self.schema)
        self.ix = open_dir(self.dir)
        # self.writer = AsyncWriter(self.ix)
        self.writer = self.ix.writer()

    # def getIsNeedParse(self):
    #     return

    def searcher(self, query):

        # query = QueryParser("content", self.ix.schema).parse("Why are yawns contagious")
        self.query = QueryParser("title", self.ix.schema, group=syntax.OrGroup).parse(query)

        print self.query
        print self.ix.doc_count()
        # try:
        searcher = self.ix.searcher()
        results = searcher.search(self.query, limit = 10)
        # results[0]
        # for result in results:
        #     print result.get("ba").replace("\n", " ")
        return [result.get("ba").replace("\n", " ") for result in results]
            # print(results[0])
        # finally:
        #     searcher.close()