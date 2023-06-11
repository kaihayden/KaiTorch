from kaitorch.utils import wrap
from kaitorch.core import Scalar

__all__ = ['mse', 'binary_crossentropy', 'categorical_crossentropy']


def mse():
    return MeanSquaredError()


def binary_crossentropy():
    return BinaryCrossentropy()


def categorical_crossentropy():
    return CategoricalCrossentropy()


class MeanSquaredError:

    def __init__(self):
        pass

    def __call__(self, ys: list, y_preds: list):

        ys, y_preds = wrap(ys), wrap(y_preds)

        # for 1/N
        pred_length = len(ys)

        # Summation Term
        squared_error = sum(
            (y - y_pred)**2 for y, y_pred in zip(ys, y_preds))

        # Mean Squared Error
        mean_squared_error = squared_error/pred_length

        return mean_squared_error

    def __repr__(self):
        return 'MeanSquaredError()'


class BinaryCrossentropy:

    def __init__(self):
        pass

    def __call__(self, ys, y_preds):

        loss = 0.0
        ys, y_preds = wrap(ys), wrap(y_preds)

        # for 1/N
        pred_length = len(ys)

        # Summation term - could've done this more concisely but wanted to make the logic clear
        for y, y_pred in zip(ys, y_preds):

            # Active Left Term
            if y == 1:
                loss += -(y_pred).log()

            # Active Right Term
            elif y == 0:
                loss += -(1 - y_pred).log()

        # Binary Cross Entropy
        binary_crossentropy_loss = loss / pred_length

        return binary_crossentropy_loss

    def __repr__(self):
        return 'BinaryCrossentropy()'


class CategoricalCrossentropy:

    def __init__(self):
        pass

    def __call__(self, ys, y_preds):

        loss = 0.0
        if isinstance(ys[0], (int, float, Scalar)):
            ys, y_preds = [ys], [y_preds]

        # 1/N
        pred_length = len(ys)

        # Outer summation term
        for y_ohe, y_pred_ohe in zip(ys, y_preds):

            # Inner summation term
            for y, y_pred in zip(y_ohe, y_pred_ohe):

                # if j is the actual class
                if y == 1:
                    loss += -(y_pred).log()

                # if j is not the actual class
                elif y == 0:
                    loss += -(1 - y_pred).log()

        # Categorical Cross Entropy
        categorical_crossentropy_loss = loss / pred_length

        return categorical_crossentropy_loss

    def __repr__(self):
        return 'CategoricalCrossEntropy()'
