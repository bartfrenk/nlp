import numpy as np
import pandas as pd


class Embeddings:
    def __init__(self, terms, embed):
        self.matrix = np.array([embed(kw) for kw in terms.keyword.values])
        self.terms = terms
        self._embed = embed

        self.__norm_matrix = None

    @property
    def _norm_matrix(self):
        if self.__norm_matrix is None:
            self.__norm_matrix = \
                np.sqrt(np.einsum('ij,ij->i', self.matrix, self.matrix))
        return self.__norm_matrix

    @classmethod
    def compute(cls, terms, embed):
        return cls(terms, embed)

    def __len__(self):
        return len(self.terms)

    def most_similar_euclidean(self, term):
        e0 = self._embed(term)
        diff = self.matrix - e0

        dists = np.einsum('ij,ij->i', diff, diff)
        _terms = self.terms.copy()
        _terms['_dists'] = dists
        _terms.sort_values(by='_dists', inplace=True)
        return _terms

    def most_similar_cosine(self, term):
        e0 = self._embed(term)
        inner = np.sum(self.matrix * e0, axis=1)

        norm_e0 = np.sqrt(np.dot(np.transpose(e0), e0))

        dists = 1 - inner / (self._norm_matrix * norm_e0)

        _terms = self.terms.copy()
        _terms['_dists'] = dists
        _terms.sort_values(by='_dists', inplace=True)
        return _terms


def read_terms(path):
    terms = pd.read_csv(path)
    terms.columns = [s.lower() for s in terms.columns]
    return terms
