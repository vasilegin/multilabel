import copy
import numpy as np

from scipy.sparse import hstack, issparse, lil_matrix

from skmultilearn.base.problem_transformation import ProblemTransformationBase
from skmultilearn.base.base import MLClassifierBase


class BinaryRelevance(ProblemTransformationBase):

    def __init__(self, classifier=None, require_dense=None):
        super(BinaryRelevance, self).__init__(classifier, require_dense)

    def _generate_partition(self, X, y):
        self.partition_ = list(range(y.shape[1]))
        self.model_count_ = y.shape[1]

    def fit(self, X, y):

        X = self._ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)
        y = self._ensure_output_format(
            y, sparse_format='csc', enforce_sparse=True)

        self.classifiers_ = []
        self._generate_partition(X, y)
        self._label_count = y.shape[1]

        for i in range(self.model_count_):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self._generate_data_subset(y, self.partition_[i], axis=1)
            if issparse(y_subset) and y_subset.ndim > 1 and y_subset.shape[1] == 1:
                y_subset = np.ravel(y_subset.toarray())
            classifier.fit(self._ensure_input_format(
                X), self._ensure_output_format(y_subset))
            self.classifiers_.append(classifier)

        return self

    def predict(self, X):

        predictions = [self._ensure_multi_label_from_single_class(
            self.classifiers_[label].predict(self._ensure_input_format(X)))
            for label in range(self.model_count_)]

        return hstack(predictions)

    def predict_proba(self, X):

        result = lil_matrix((X.shape[0], self._label_count), dtype='float')
        for label_assignment, classifier in zip(self.partition_, self.classifiers_):
            if isinstance(self.classifier, MLClassifierBase):
                # the multilabel classifier should provide a (n_samples, n_labels) matrix
                # we just need to reorder it column wise
                result[:, label_assignment] = classifier.predict_proba(X)
            else:
                # a base classifier for binary relevance returns
                # n_samples x n_classes, where n_classes = [0, 1] - 1 is the probability of
                # the label being assigned
                result[:, label_assignment] = self._ensure_multi_label_from_single_class(
                    classifier.predict_proba(
                        self._ensure_input_format(X))
                )[:, 1]  # probability that label is assigned

        return result
