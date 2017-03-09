import logging
import xml.sax

from liveqa.vertical.Indexing import Indexing


class XmlParsehandler(xml.sax.ContentHandler):
    def __init__(self):
        self.tag = ""
        self.subject = ""
        self.content = ""
        self.bestAnswer = ""
        self.indexing = Indexing()

    def startElement(self, tag, attrs):
        self.tag = tag
        # if tag == "vespaadd":
        #     print ""
    def endElement(self, tag):
        if tag == "document":
            print("Indexing doc...")
            self.indexing.indexing(self.subject,self.content,self.bestAnswer)
            self.tag = ""
            self.subject = u""
            self.content = u""
            self.bestAnswer = u""


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

    parser.parse("../data/FullOct2007.xml")

    # parse the query
    Handler.indexing.searcher("Why are yawns contagious")
    Handler.indexing.closeWriter()