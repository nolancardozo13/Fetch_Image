from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from .helper import Matching, argtopk, CombinatorMixin, EmbeddedVectorizer
from .retreival import Retrieval
import numpy as np
import os

documents = []
images = []


class WordCentroidDistance(BaseEstimator, CombinatorMixin):
    """
    This class should only be used inside a Retrieval, so that Retrieval
    cares about the matching and the resulting indices.
    """

    def __init__(self, embedding, analyzer='word', use_idf=True):
        self.vect = EmbeddedVectorizer(embedding,
                                       analyzer=analyzer,
                                       use_idf=use_idf)
        self.centroids = None

    def fit(self, X):
        Xt = self.vect.fit_transform(X)
        Xt = normalize(Xt, copy=False)  # We need this because of linear kernel
        self.centroids = Xt

    def query(self, query, k=None, indices=None, return_scores=False, sort=True):
        centroids = self.centroids
        if centroids is None:
            raise NotFittedError
        if indices is not None:
            centroids = centroids[indices]
        q = self.vect.transform([query])
        q = normalize(q, copy=False)
        D = linear_kernel(q, centroids)  # l2 normalized, so linear kernel
        # ind = np.argsort(D[0, :])[::-1]  # similarity metric, so reverse
        # if k is not None:  # we could use our argtopk in the first place
        #     ind = ind[:k]
        # print(ind)
        ind = argtopk(D[0], k) if sort else np.arange(D.shape[1])
        if return_scores:
            return ind, D[0, ind]
        else:
            return ind

def create_word_centroid_distance_for_given_images(folder):
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            filename = os.path.splitext(file)[0] + '.jpg'
            images.append(filename)
            with open(os.path.join(folder, file), 'r') as f:
                text = f.read()
                documents.append(text)
    model = Word2Vec([doc.split() for doc in documents], iter=1, min_count=1)
    return model


def get_top_n_images(model, query, n=10):
    match_op = Matching()
    wcd = WordCentroidDistance(model.wv)
    retrieval = Retrieval(wcd, matching=match_op)
    retrieval.fit(documents)
    result = retrieval.query(query, k=n)
    return [images[i] for i in result]
