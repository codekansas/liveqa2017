from __future__ import absolute_import

import logging
import xml.sax

import sys

from liveqa.rank.ShallowRank import ShallowRank
from liveqa.vertical.Indexing import Indexing


class XmlParsehandler(xml.sax.ContentHandler):
    def __init__(self):
        self.tag = ""
        self.subject = ""
        self.content = ""
        self.bestAnswer = ""
        self.indexing = Indexing()
        self.i = 0

    def startElement(self, tag, attrs):
        self.tag = tag
        # if tag == "vespaadd":
        #     print ""

    def endElement(self, tag):
        if tag == "document":
            print("Indexing doc...")

            if self.i<10000000:
                self.indexing.indexing(self.subject,self.content,self.bestAnswer)
                self.tag = ""
                self.subject = u""
                self.content = u""
                self.bestAnswer = u""
                self.i = self.i + 1
            else:
                Handler.indexing.closeWriter()
                Handler.indexing.closeIndexing()

    def characters(self, content):
        if self.tag == "subject":
            self.subject += content
        elif self.tag == "content":
            self.content += content
        elif self.tag == "bestanswer":
            self.bestAnswer += content

if (__name__ == "__main__"):
    # create an XMLReader
    parser = xml.sax.make_parser()

    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # override the default ContextHandler
    Handler = XmlParsehandler()
    parser.setContentHandler(Handler)

    if Handler.indexing.isWriteModeOn:
        parser.parse("../data/FullOct2007.xml")
        Handler.indexing.closeWriter()
        Handler.indexing.closeIndexing()
    else:
        # parse the query
        while(1):
            query = raw_input("Enter your question:\n")
            # for line in sys.stdin.readline():
            # print test
            # Handler.indexing.searcher("do you know why are yawns contagious")
            candidates = Handler.indexing.searcher(query)
            Handler.indexing.closeIndexing()
            # print ShallowRank.getCandidates(1,query, candidates)
            print ShallowRank(1, query, candidates).shallowRank()
