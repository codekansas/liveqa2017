from gensim import models,corpora,similarities
import pickle
class ShallowRank(object):
    def __init__(self):
        self.stoplist = set('for a of the and to in'.split())

    def train(self,corpus):
        documents = [[word for word in document.lower().split() if word not in self.stoplist] for document in
                     corpus]
        dictionary = corpora.Dictionary(documents)
        pickle.dump(dictionary,open("dictionary.p","wb"))

    def getCandidates(self,nc,query,candidates):
        dictionary = pickle.load(open("dictionary.p","rb"))
        query_vec = dictionary.doc2bow(query.lower().split())
        documents = [[word for word in document.lower().split() if word not in self.stoplist] for document in candidates]
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
        vec_lsi = lsi[query_vec]
        index = similarities.MatrixSimilarity(lsi[corpus])
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        if nc <= len(sims):
            selected = [sim[0] for sim in sims[0:nc]]
        else:
            selected = [sim[0] for sim in sims]
        results = [candidates[n] for n in selected]
        return results

