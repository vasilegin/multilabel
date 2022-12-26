from skmultilearn.base.problem_transformation import ProblemTransformationBase
from scipy.sparse import hstack
from sklearn.exceptions import NotFittedError
import copy


class ClassifierChain(ProblemTransformationBase):

    def __init__(self, classifier=None, require_dense=None, order=None):
        super(ClassifierChain, self).__init__(classifier, require_dense)
        self.order = order
        self.copyable_attrs = ['classifier', 'require_dense', 'order']

    def fit(self, X, y, order=None):

        # fit L = len(y[0]) BR classifiers h_i
        # on X + y[:i] as input space and y[i+1] as output

        X_extended = self._ensure_input_format(X, sparse_format='csc', enforce_sparse=True)
        y = self._ensure_output_format(y, sparse_format='csc', enforce_sparse=True)

        self._label_count = y.shape[1]
        self.classifiers_ = [None for x in range(self._label_count)]

        for label in self._order():
            self.classifier = copy.deepcopy(self.classifier)
            y_subset = self._generate_data_subset(y, label, axis=1)

            self.classifiers_[label] = self.classifier.fit(self._ensure_input_format(
                X_extended), self._ensure_output_format(y_subset))
            X_extended = hstack([X_extended, y_subset])

        return self

    def predict(self, X):

        X_extended = self._ensure_input_format(
            X, sparse_format='csc', enforce_sparse=True)

        for label in self._order():
            prediction = self.classifiers_[label].predict(
                self._ensure_input_format(X_extended))
            prediction = self._ensure_multi_label_from_single_class(prediction)
            X_extended = hstack([X_extended, prediction])
        return X_extended[:, -self._label_count:]

    def predict_proba(self, X):

        X_extended = self._ensure_input_format(
            X, sparse_format='csc', enforce_sparse=True)

        results = []
        for label in self._order():
            prediction = self.classifiers_[label].predict(
                self._ensure_input_format(X_extended))

            prediction = self._ensure_output_format(
                prediction, sparse_format='csc', enforce_sparse=True)

            prediction_proba = self.classifiers_[label].predict_proba(
                self._ensure_input_format(X_extended))

            prediction_proba = self._ensure_output_format(
                prediction_proba, sparse_format='csc', enforce_sparse=True)[:, 1]

            X_extended = hstack([X_extended, prediction]).tocsc()
            results.append(prediction_proba)

        return hstack(results)

    def _order(self):
        if self.order is not None:
            return self.order

        try:
            return list(range(self._label_count))
        except AttributeError:
            raise NotFittedError("This Classifier Chain has not been fit yet")
