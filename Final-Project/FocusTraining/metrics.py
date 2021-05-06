import numpy as np
import tensorflow.keras.backend as K


class Metric:
    '''Metric class

    Computes a metric for evaluation purposes.
    Has a name for displaying the value and can
    be subclassed to compute specific metrics.
    '''

    def __init__(self, name):
        self.name = name

    def compute(self, total_y_true, total_y_pred):
        return 0

    def _round_value(self, val, n_decimals):
        return np.round(val, n_decimals)

    def compute_display_value(self, total_y_true, total_y_pred, n_decimals=3):
        return str(self._round_value(self.compute(total_y_true, total_y_pred),
                                     n_decimals))


class Accuracy(Metric):
    """Accuracy metric"""

    def __init__(self):
        super().__init__('acc')

    def compute(self, total_y_true, total_y_pred):
        return sum(total_y_true == total_y_pred) / len(total_y_true)


class F1(Metric):
    """Compute F1 metric

    Note: may not work as expected.  Didn't test thoroughly.
    """

    def __init__(self):
        super().__init__('f1')

    def recall(self, y_true, y_pred):
        y_true = K.ones_like(y_true)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (all_positives + K.epsilon())
        return recall

    def precision(self, y_true, y_pred):
        y_true = K.ones_like(y_true)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def compute(self, y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
