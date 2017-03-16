from __future__ import absolute_import
from __future__ import print_function

from gensim import models, corpora, similarities
from liveqa import yahoo

import logging

import os
import re
import pickle

BASE = os.path.dirname(os.path.realpath(__file__))
STOPWORDS_FILE = os.path.join(BASE, 'stopwords.txt')
DICTIONARY_FILE = os.path.join(BASE, 'dictionary.pkl')
MODEL_FILE = os.path.join(BASE, 'model.gensim')
NUM_TOPICS = 100


class ShallowRank(object):
    """Shallow ranking of documents using LSI."""

    def __init__(self,
                 num_topics=NUM_TOPICS,
                 dictionary_file=DICTIONARY_FILE,
                 model_file=MODEL_FILE):
        """Initializes the ranker.

        Args:
            num_topics: int (default: NUM_TOPICS), number of LSI topics to use.
            dictionary_file: str, where to save / load the dictionary file
                (defaults to DICTIONARY_FILE).
            model_file: str, where to save / load the model (defaults to
                MODEL_FILE).
        """

        self.dictionary = None
        self.model = None
        self.num_topics = num_topics
        self.dictionary_file = dictionary_file
        self.model_file = model_file

        # Loads stopwords from the associated file.
        with open(STOPWORDS_FILE, 'r') as f:
            self.stoplist = set(f.read().strip().split())

        # Loads an existing dictionary file, if one exists.
        if os.path.exists(self.dictionary_file):
            with open(self.dictionary_file, 'rb') as f:
                self.dictionary = pickle.load(f)

        # Loads an existing model file, if one exists.
        if os.path.exists(self.model_file):
            self.model = models.LdaModel.load(self.model_file)

    def tokenize(self, document):
        """Tokenizes a document.

        Args:
            document: str, the document to tokenize.

        Returns:
            list of str, the tokenized document.
        """

        # tokens = [w for w in document.lower().split()
        #           if w not in self.stoplist]

        # Use regex to handle punctuation.
        tokens = re.findall('[\w\d\']+|[?\.!-\*]+', document.lower())

        return tokens

    def train(self, corpus, passes=1):
        """Updates dictionary and model given a corpus.

        Args:
            corpus: list of str, the documents to tokenize.
        """

        if self.dictionary is not None or self.model is not None:
            x = raw_input('You are about to overwrite an existing '
                          'model file (%s). Are you sure? [y/N] '
                          % self.model_file)

            if x[0] != 'y':
                raise RuntimeError('You chose not to overwrite the '
                                   'existing model and dictionary.')

        # Tokenizes the corpus.
        documents = [self.tokenize(document) for document in corpus]

        # Builds a dictionary from the existing documents.
        self.dictionary = corpora.Dictionary(documents)

        # Dumps the dictionary to a pickled file to use later.
        pickle.dump(self.dictionary, open(self.dictionary_file, 'wb'))

        # Converts the corpus to tokens.
        corpus_bow = [self.dictionary.doc2bow(doc) for doc in documents]

        # Trains the LSI model.
        self.model = models.LdaModel(corpus_bow,
                                     passes=passes,
                                     id2word=self.dictionary,
                                     num_topics=self.num_topics)

        # Saves the model to use later.
        self.model.save(self.model_file)

    def get_candidates(self, query, candidatesm, nc=None):
        """Gets the top N candidate answers for a query.

        Args:
            query: str, the query provided by the user.
            candidates: list of str, the candidate answers.
            nc: int (default: None), number of candiate answers to retrieve,
                or retrieve all of them if None.

        Returns:
            results: a list of strings, the candidates sorted from most to
                least relevant.
        """

        # Tokenizes the query and documents.
        query_vec = self.dictionary.doc2bow(self.tokenize(query))
        documents = [self.tokenize(document) for document in candidates]
        candidates_bow = [self.dictionary.doc2bow(doc) for doc in documents]

        # Creates bag-of-words from each document.
        vec_emb = self.model[query_vec]
        index = similarities.MatrixSimilarity(self.model[candidates_bow])

        # Sorts documents by similarity to query.
        sims = index[vec_emb]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        selected = [sim[0] for sim in sims]
        if nc is not None:
            selected = selected[:nc]

        # Gets the original candidate texts (untokenized).
        results = [candidates[n] for n in selected]

        return results


if __name__ == '__main__':  # Builds the index from the full corpus.
    import itertools

    # Turn on logging.
    logging.basicConfig(level=logging.DEBUG)

    ranker = ShallowRank(dictionary_file='/tmp/dictionary.pkl',
                         model_file='/tmp/model.gensim')

    NUM_ANSWERS = 1000000  # How many answers to use to train LDA model.

    iterable = yahoo.iterate_qa_pairs(convert_to_tokens=False)
    iterable = itertools.islice(iterable, NUM_ANSWERS)
    answers = []

    for i, (_, _, answer, _, _, _) in enumerate(iterable):
        answers.append(answer)
        if i % 1000 == 0:
            logging.info('generated %d' % i)

    # Trains the ranker on the document corpus (takes a while).
    ranker.train(answers)
