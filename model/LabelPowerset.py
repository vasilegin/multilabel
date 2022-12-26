from skmultilearn.base.problem_transformation import ProblemTransformationBase
import numpy as np
from scipy import sparse


class LabelPowerset(ProblemTransformationBase):

    def __init__(self, classifier=None, require_dense=None):
        super(LabelPowerset, self).__init__(
            classifier=classifier, require_dense=require_dense)
        self._clean()

    def _clean(self):

        self.unique_combinations_ = {}
        self.reverse_combinations_ = []
        self._label_count = None

    def fit(self, X, y):

        X = self._ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)

        self.classifier.fit(self._ensure_input_format(X),
                            self.transform(y))

        return self

    def predict(self, X):

        lp_prediction = self.classifier.predict(self._ensure_input_format(X))

        return self.inverse_transform(lp_prediction)

    def predict_proba(self, X):

        lp_prediction = self.classifier.predict_proba(
            self._ensure_input_format(X))
        result = sparse.lil_matrix(
            (X.shape[0], self._label_count), dtype='float')
        for row in range(len(lp_prediction)):
            assignment = lp_prediction[row]
            for combination_id in range(len(assignment)):
                for label in self.reverse_combinations_[combination_id]:
                    result[row, label] += assignment[combination_id]

        return result

    def transform(self, y):

        y = self._ensure_output_format(
            y, sparse_format='lil', enforce_sparse=True)

        self._clean()
        self._label_count = y.shape[1]

        last_id = 0
        train_vector = []
        for labels_applied in y.rows:
            label_string = ",".join(map(str, labels_applied))

            if label_string not in self.unique_combinations_:
                self.unique_combinations_[label_string] = last_id
                self.reverse_combinations_.append(labels_applied)
                last_id += 1

            train_vector.append(self.unique_combinations_[label_string])

        return np.array(train_vector)

    def inverse_transform(self, y):

        n_samples = len(y)
        result = sparse.lil_matrix((n_samples, self._label_count), dtype='i8')
        for row in range(n_samples):
            assignment = y[row]
            result[row, self.reverse_combinations_[assignment]] = 1

        return result
