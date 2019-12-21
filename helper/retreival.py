import numpy as np


from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from .helper import RetriEvalMixin
from nltk.corpus import stopwords


ENGLISH_STOP_WORDS = stopwords.words('english')


class Retrieval(BaseEstimator, MetaEstimatorMixin, RetriEvalMixin):

    """Meta estimator for an end to end information retrieval process"""

    def __init__(self, retrieval_model, matching=None,
                 query_expansion=None, name='RM',
                 labels=None):
        """TODO: to be defined1.
        :retrieval_model: A retrieval model satisfying fit and query.
        :vectorizer: A vectorizer satisfying fit and transform (and fit_transform).
        :matching: A matching operation satisfying fit and predict.
        :query_expansion: A query operation satisfying fit and transform
        :labels: Pre-defined mapping of indices to identifiers, will be inferred during fit, if not given.
        """
        BaseEstimator.__init__(self)

        self._retrieval_model = retrieval_model
        self._matching = matching
        self._query_expansion = query_expansion
        self.name = name
        self.labels_ = np.asarray(labels) if labels is not None else None

    def fit(self, X, y=None):
        """ Fit vectorizer to raw_docs, transform them and fit the
        retrieval_model.  Matching and Query expansion are fit separatly on the
        `raw_docs` to allow dedicated analysis.
        """
        assert y is None or len(X) == len(y)
        if self.labels_ is None:
            # If labels were not specified, infer them from y
            self.labels_ = np.asarray(y) if y is not None else np.arange(len(X))
        matching = self._matching
        query_expansion = self._query_expansion
        retrieval_model = self._retrieval_model

        if query_expansion:
            query_expansion.fit(X)

        if matching:
            matching.fit(X)

        retrieval_model.fit(X)
        return self

    def query(self, q, k=None, return_scores=False):
        labels = self.labels_
        if labels is None:
            raise NotFittedError
        matching = self._matching
        retrieval_model = self._retrieval_model
        query_expansion = self._query_expansion

        if query_expansion:
            q = query_expansion.transform(q)

        if matching:
            ind = matching.predict(q)
            # print('{} documents matched.'.format(len(ind)))
            if len(ind) == 0:
                if return_scores:
                    return [], []
                else:
                    return []
            labels = labels[ind]  # Reduce our own view
        else:
            ind = None

        # pass matched indices to query method of retrieval model
        # The retrieval model is assumed to reduce its representation of X
        # to the given indices and the returned indices are relative to the
        # reduction

        if return_scores:
            try:
                ind, scores = retrieval_model.query(q, k=k, indices=ind,
                                                    return_scores=return_scores)
            except TypeError:
                raise NotImplementedError("Underlying retrieval model does not support `return_scores`")
            if k is not None:
                ind = ind[:k]
                scores = scores[:k]

            return labels[ind], scores
        else:
            retrieved_indices = retrieval_model.query(q, k=k, indices=ind)
            if k is not None:
                # Just assert that it did not cheat
                retrieved_indices = retrieved_indices[:k]

            return labels[retrieved_indices]  # Unfold retrieved indices

