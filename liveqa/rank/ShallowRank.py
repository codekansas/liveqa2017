from gensim import models,corpora,similarities

class ShallowRank(object):

    def __init__(self,numberOfCandidates,query,candidates):
        self.nc = numberOfCandidates
        self.query = query
        self.candidates = candidates

    def shallowRank(self):
        stoplist = set('for a of the and to in'.split())
        documents = [[word for word in document.lower().split() if word not in stoplist] for document in self.candidates]
        dictionary = corpora.Dictionary(documents)
        query_vec = dictionary.doc2bow(self.query.lower().split())
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
        vec_lsi = lsi[query_vec]
        index = similarities.MatrixSimilarity(lsi[corpus])
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        print sims

sr = ShallowRank(1,"test mode",["test1 is test","education natural language processing"])
sr.shallowRank()
