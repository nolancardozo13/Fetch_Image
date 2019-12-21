import pandas as pd
import numpy as np

from timeit import default_timer as timer
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import maxabs_scale
from abc import abstractmethod
from collections import defaultdict
from nltk import word_tokenize


def harvest(source, query_id, doc_id=None, default=0):
    is_pd = isinstance(source, pd.Series)
    is_dict = isinstance(source, dict)
    if doc_id is None:
        # Return sorted list of relevance scores for that query
        if is_pd or is_dict:
            # source is pandas series or dict
            scores = source.get(query_id)
        else:
            # source is ndarray or list
            scores = source[query_id]

        #         try:
        #             # scores is numpy array-like?
        #             scores = np.sort(scores)[::-1]
        #         except ValueError:
        #             # probably scores is a dict itself...
        if isinstance(scores, dict):
            scores = np.asarray(list(scores.values()))
        else:
            scores = np.asarray(scores)
        #             scores = np.sort(scores)[::-1]
        return scores
    else:
        # Return relevance score for the respective (query, document) pair
        # try:  # pandas multi index df
        if is_pd:
            score = source.get((query_id, doc_id), default)
        else:
            # default dict or ndarray
            scores = source[query_id]
            # no special treatment for ndarray since we want to raise exception
            # when query id is out of bounds
            try:
                # ok now if scores provides a get
                # (pandas or dict), lets use it:
                score = scores.get(doc_id, default)
            except AttributeError:
                # try the brute force access
                try:
                    score = scores[doc_id]
                # the user should be aware when he wants to index his stuff
                # by numpy indexing: only int doc ids allowed, of course.
                except IndexError:
                    score = default

        return score


def argtopk(A, k=None, sort=True):
    assert k != 0, "That does not make sense"
    if k is not None and k < 0:
        # Just invert A element-wise
        k = -k
        A = -A
    A = np.asarray(A)
    if len(A.shape) > 1:
        raise ValueError('argtopk only defined for 1-d slices')
    axis = -1
    if k is None or k >= A.size:
        # if list is too short or k is None, return all in sort order
        if sort:
            return np.argsort(A, axis=axis)[::-1]
        else:
            return np.arange(A.shape[0])

    # assert k > 0
    # now 0 < k < len(A)
    ind = np.argpartition(A, -k, axis=axis)[-k:]
    if sort:
        # sort according to values in A
        # argsort is always from lowest to highest, so reverse
        ind = ind[np.argsort(A[ind], axis=axis)][::-1]

    return ind


def filter_none(some_list):
    """ Just filters None elements out of a list """
    old_len = len(some_list)
    new_list = [l for l in some_list if l is not None]
    diff = old_len - len(new_list)
    return new_list, diff


def f1_score(precision, recall):
    """
    Computes the harmonic mean of precision and recall (f1 score)
    """
    if precision == 0 and recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def recall(r, n_relevant):
    """ Recall bounded to R, relevance is binary"""
    assert n_relevant >= 1
    relevant_and_retrieved = np.count_nonzero(r)
    return relevant_and_retrieved / n_relevant


def precision(r):
    """ Unbounded precision of r, relevance is binary """
    r = np.asarray(r)
    if r.size == 0:
        return 0
    tp = np.count_nonzero(r)
    return tp / r.size


def safe_precision_at_k(r, k):
    r = np.asarray(r)
    if r.size == 0:
        return 0.
    return precision_at_k(r, min(k, r.size))


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


class CombinatorMixin(object):
    """ Creates a computational tree with retrieval models as leafs
    """

    def __get_weights(self, other):
        if not isinstance(other, CombinatorMixin):
            raise ValueError("other is not Combinable")

        if hasattr(self, '__weight'):
            weight = self.__weight
        else:
            weight = 1.0

        if hasattr(other, '__weight'):
            otherweight = other.__weight
        else:
            otherweight = 1.0

        return weight, otherweight

    # This is evil since it can exceed [0,1], rescaling at the end would be not
    # that beautiful
    def __add__(self, other):
        weights = self.__get_weights(other)
        return Combined([self, other], weights=weights, aggregation_fn='sum')

    def __mul__(self, other):
        weights = self.__get_weights(other)
        return Combined([self, other], weights=weights, aggregation_fn='product')

    def __pow__(self, scalar):
        self.__weight = scalar
        return self


class Combined(BaseEstimator, CombinatorMixin):
    def __init__(self, retrieval_models, weights=None, aggregation_fn='sum'):
        self.retrieval_models = retrieval_models
        self.aggregation_fn = aggregation_fn
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1.0] * len(retrieval_models)
        assert len(self.retrieval_models) == len(self.weights)

    def fit(self, *args, **kwargs):
        for model in self.retrieval_models:
            model.fit(*args, **kwargs)
        return self

    def query(self, query, k=None, indices=None, sort=True, return_scores=False):
        models = self.retrieval_models
        weights = maxabs_scale(self.weights)  # max 1 does not crash [0,1]
        agg_fn = self.aggregation_fn
        # It's important that all retrieval model return the same number of documents.
        all_scores = [m.query(query, k=k, indices=indices, sort=False, return_scores=True)[1] for m in models]

        if weights is not None:
            all_scores = [weight * scores for weight, scores in zip(all_scores, weights)]

        scores = np.vstack(all_scores)
        if callable(agg_fn):
            aggregated_scores = agg_fn(scores)
        else:
            numpy_fn = getattr(np, agg_fn)
            aggregated_scores = numpy_fn(scores, axis=0)

        # combined = aggregate_dicts(combined, agg_fn=agg_fn, sort=True)

        # only cut-off at k if this is the final (sorted) output
        ind = argtopk(aggregated_scores, k) if sort else np.arange(aggregated_scores.shape[0])
        if return_scores:
            return ind, aggregated_scores[ind]
        else:
            return ind


def match_bool_or(X, q):
    # indices = np.unique(X.transpose()[q.nonzero()[1], :].nonzero()[1])
    # print("matching X", X, file=sys.stderr)
    # print("matching q", q, file=sys.stderr)
    # inverted_index = X.transpose()
    inverted_index = X.T
    # print("matching inverted_index", inverted_index, file=sys.stderr)
    query_terms = q.nonzero()[1]
    # print("matching query_terms", query_terms, file=sys.stderr)
    matching_terms = inverted_index[query_terms, :]
    # print("matching matching_terms", matching_terms, file=sys.stderr)
    matching_doc_indices = np.unique(matching_terms.nonzero()[1])
    # print("matching matching_doc_indices", matching_doc_indices,
    # file=sys.stderr)
    return matching_doc_indices


class Matching(BaseEstimator):
    """Typical Matching Operation of Retrieval Systems"""

    def __init__(self, match_fn=match_bool_or, binary=True, dtype=np.bool_,
                 **cv_params):
        """initializes a Matching object
        :match_fn: A matching function of signature `docs, query`
                    -> indices of matching docs
        :binary: Store only binary term occurrences.
        :dtype: Data type of internal feature matrix
        :cv_params: Parameter for the count vectorizer such as lowercase=True
        """
        # RetrievalBase.__init__(self)

        self._match_fn = match_fn
        self._vect = CountVectorizer(binary=binary, dtype=dtype,
                                     **cv_params)

    def fit(self, X):
        cv = self._vect

        self._fit_X = cv.fit_transform(X)  # fit internal countvectorizer

        return self

    def predict(self, query):
        cv, match_fn, fit_X = self._vect, self._match_fn, self._fit_X
        # q = cv.transform(np.asarray([query]))
        q = cv.transform(([query]))
        ind = match_fn(fit_X, q)
        return ind

    def argtopk(A, k=None, sort=True):
        assert k != 0, "That does not make sense"
        if k is not None and k < 0:
            # Just invert A element-wise
            k = -k
            A = -A
        A = np.asarray(A)
        if len(A.shape) > 1:
            raise ValueError('argtopk only defined for 1-d slices')
        axis = -1
        if k is None or k >= A.size:
            # if list is too short or k is None, return all in sort order
            if sort:
                return np.argsort(A, axis=axis)[::-1]
            else:
                return np.arange(A.shape[0])

        # assert k > 0
        # now 0 < k < len(A)
        ind = np.argpartition(A, -k, axis=axis)[-k:]
        if sort:
            # sort according to values in A
            # argsort is always from lowest to highest, so reverse
            ind = ind[np.argsort(A[ind], axis=axis)][::-1]

        return ind


class RetriEvalMixin():

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def query(X, k=None):
        pass

    def evaluate(self, X, Y, k=20, verbose=0, replacement=0, n_jobs=1):
        """
        :X: [(qid, str)] query id, query string pairs
        :Y: pandas dataseries with qid,docid index or [dict]
        :k: Limit the result for all metrics to this value, the models are also
        given a hint of how many they should return.
        :replacement: 0 means that (query, doc) pairs not prevalent in Y will
        not be considered relevant, None means that those are not considered
        (skipped).
        """
        # rs = []

        # if n_jobs > 1:
        #     return process_and_evaluate(self, X, Y, k, n_jobs)
        values = defaultdict(list)
        for qid, query in X:
            # execute query
            if verbose > 0:
                print(qid, ":", query)
            t0 = timer()
            # if replacement is None, we need to drop after querying
            result = self.query(query, k=(None if replacement is None else k))
            values["time_per_query"].append(timer() - t0)
            # if verbose > 0:
            #     print(result[:k])
            # result = result[:k]  # TRIM HERE
            # soak the generator
            scored_result = [harvest(Y, qid, docid, replacement)
                             for docid in result]
            if replacement is None:
                scored_result, notfound = filter_none(scored_result)
                values["gold_not_found"].append(notfound)

            if k is not None:
                # dont let the models cheat by returning more than k
                r = scored_result[:k]
            else:
                # if k is None, consider all
                r = scored_result

            # if verbose > 0:
            #     print(r)

            # gold = np.array(list(Y[qid].values()))
            gold = harvest(Y, qid)
            import sys
            # print(gold, file=sys.stderr)
            topk_indices = argtopk(gold, k)
            print(topk_indices, file=sys.stderr)
            gold_topk = gold[topk_indices]
            # print('Top k in gold standard:', gold_topk, file=sys.stderr)
            R = np.count_nonzero(gold_topk)
            if verbose > 0:
                print("Retrieved {} relevant out of {} possible."
                      .format(np.count_nonzero(r), R))

            # real ndcg
            idcg = dcg_at_k(gold_topk, k)
            ndcg = dcg_at_k(scored_result, k) / idcg
            values["ndcg"].append(ndcg)
            # Verified

            # MAP@k
            ap = average_precision(r)
            values["MAP"].append(ap)

            # MRR - compute by hand
            ind = np.asarray(r).nonzero()[0]
            mrr = (1. / (ind[0] + 1)) if ind.size else 0.
            values["MRR"].append(mrr)

            # R precision
            # R = min(R, k)  # ok lets be fair.. you cant get more than k
            # we dont need that anymore, since we chop of the remainder
            # before computing R
            recall = recall(r, R)
            values["recall"].append(recall)

            # precision = rm.precision_at_k(pad(scored_result, k), k)
            precision = precision(r)
            values["precision"].append(precision)

            f1 = f1_score(precision, recall)
            values["f1_score"].append(f1)

            # Safe variant does not fail if len(r) < k
            p_at_5 = safe_precision_at_k(r, 5)
            values["precision@5"].append(p_at_5)

            p_at_10 = safe_precision_at_k(r, 10)
            values["precision@10"].append(p_at_10)

            # rs.append(r)
            if verbose > 0:
                # print("Precision: {:.4f}".format(precision))
                # print("Recall: {:.4f}".format(recall))
                # print("F1-Score: {:.4f}".format(f1))
                print("AP: {:.4f}".format(ap))
                print("RR: {:.4f}".format(mrr))
                print("NDCG: {:.4f}".format(ndcg))

        return values


def build_analyzer(tokenizer=None, stop_words=None, lowercase=True):
    """
    A wrapper around sklearns CountVectorizers build_analyzer, providing an
    additional keyword for nltk tokenization.
    :tokenizer:
        None or 'sklearn' for default sklearn word tokenization,
        'sword' is similar to sklearn but also considers single character words
        'nltk' for nltk's word_tokenize function,
        or callable.
    :stop_words:
        None for no stopword removal, or list of words, 'english'
    :lowercase:
        Lowercase or case-sensitive analysis.
    """
    # some default options for tokenization
    if not callable(tokenizer):
        tokenizer, token_pattern = {
            'sklearn': (None, r"(?u)\b\w\w+\b"),  # mimics default
            'sword': (None, r"(?u)\b\w+\b"),  # specifically for GoogleNews
            'nltk': (word_tokenize, None)  # uses punctuation for GloVe models
        }[tokenizer]

    # allow binary decision for stopwords
    # sw_rules = {True: 'english', False: None}
    # if stop_words in sw_rules:
    #     stop_words = sw_rules[stop_words]

    # employ the cv to actually build the analyzer from the components
    analyzer = CountVectorizer(analyzer='word',
                               tokenizer=tokenizer,
                               token_pattern=token_pattern,
                               lowercase=lowercase,
                               stop_words=stop_words).build_analyzer()
    return analyzer


class CombinatorMixin(object):
    """ Creates a computational tree with retrieval models as leafs
    """
    def __get_weights(self, other):
        if not isinstance(other, CombinatorMixin):
            raise ValueError("other is not Combinable")

        if hasattr(self, '__weight'):
            weight = self.__weight
        else:
            weight = 1.0

        if hasattr(other, '__weight'):
            otherweight = other.__weight
        else:
            otherweight = 1.0

        return weight, otherweight

    # This is evil since it can exceed [0,1], rescaling at the end would be not
    # that beautiful
    def __add__(self, other):
        weights = self.__get_weights(other)
        return Combined([self, other], weights=weights, aggregation_fn='sum')

    def __mul__(self, other):
        weights = self.__get_weights(other)
        return Combined([self, other], weights=weights, aggregation_fn='product')

    def __pow__(self, scalar):
        self.__weight = scalar
        return self


class Combined(BaseEstimator, CombinatorMixin):
    def __init__(self, retrieval_models, weights=None, aggregation_fn='sum'):
        self.retrieval_models = retrieval_models
        self.aggregation_fn = aggregation_fn
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1.0] * len(retrieval_models)
        assert len(self.retrieval_models) == len(self.weights)

    def fit(self, *args, **kwargs):
        for model in self.retrieval_models:
            model.fit(*args, **kwargs)
        return self

    def query(self, query, k=None, indices=None, sort=True, return_scores=False):
        models = self.retrieval_models
        weights = maxabs_scale(self.weights)  # max 1 does not crash [0,1]
        agg_fn = self.aggregation_fn
        # It's important that all retrieval model return the same number of documents.
        all_scores = [m.query(query, k=k, indices=indices, sort=False, return_scores=True)[1] for m in models]

        if weights is not None:
            all_scores = [weight * scores for weight, scores in zip(all_scores, weights)]

        scores = np.vstack(all_scores)
        if callable(agg_fn):
            aggregated_scores = agg_fn(scores)
        else:
            numpy_fn = getattr(np, agg_fn)
            aggregated_scores = numpy_fn(scores, axis=0)

        # combined = aggregate_dicts(combined, agg_fn=agg_fn, sort=True)

        # only cut-off at k if this is the final (sorted) output
        ind = argtopk(aggregated_scores, k) if sort else np.arange(aggregated_scores.shape[0])
        if return_scores:
            return ind, aggregated_scores[ind]
        else:
            return ind

class EmbeddedVectorizer(TfidfVectorizer):

    """Embedding-aware vectorizer"""

    def __init__(self, embedding, **kwargs):
        """TODO: to be defined1. """
        # list of words in the embedding
        if not hasattr(embedding, 'index2word'):
            raise ValueError("No `index2word` attribute found."
                             " Supply the word vectors (`.wv`) instead.")
        if not hasattr(embedding, 'vectors'):
            raise ValueError("No `vectors` attribute found."
                             " Supply the word vectors (`.wv`) instead.")
        vocabulary = embedding.index2word
        self.embedding = embedding
        print("Embedding shape:", embedding.vectors.shape)
        TfidfVectorizer.__init__(self, vocabulary=vocabulary, **kwargs)

    def fit(self, raw_docs, y=None):
        super().fit(raw_docs)
        return self

    def transform(self, raw_documents, y=None):
        Xt = super().transform(raw_documents)
        syn0 = self.embedding.vectors
        # Xt is sparse counts
        return (Xt @ syn0)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
