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
        if self.nc <= len(sims):
            selected = [sim[0] for sim in sims[0:self.nc]]
        else:
            selected = [sim[0] for sim in sims]
        results = [self.candidates[n] for n in selected]
        return results
