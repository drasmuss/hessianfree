from functools import wraps

import numpy as np


class LossFunction:
    """Defines a loss function that maps nonlinearity activations to error."""

    def __init__(self):
        pass

    def loss(self, activities, targets):
        # note that most loss functions are only based on the output of the
        # final layer, activities[-1]. however, we pass the activities of all
        # layers here so that loss functions can include things like
        # sparsity constraints. targets, however, are only defined for the
        # output layer.
        raise NotImplementedError()

    def d_loss(self, activities, targets):
        """First derivative of loss function (with respect to activities)."""
        raise NotImplementedError()

    def d2_loss(self, activities, targets):
        """Second derivative of loss function (with respect to activities)."""
        raise NotImplementedError()


def output_loss(func):
    """Convenience wrapper that takes a loss defined for the output layer
    and converts it into the more general form in terms of all layers."""

    @wraps(func)
    def wrapped_loss(self, activities, targets):
        result = [np.zeros_like(a) for a in activities[:-1]]
        result += [func(self, activities[-1], targets)]

        return result

    return wrapped_loss


class SquaredError(LossFunction):
    @output_loss
    def loss(self, output, targets):
        return np.sum(np.nan_to_num(output - targets) ** 2,
                       axis=tuple(range(1, output.ndim))) / 2

    @output_loss
    def d_loss(self, output, targets):
        return np.nan_to_num(output - targets)

    @output_loss
    def d2_loss(self, output, targets):
        return np.ones_like(output)


class CrossEntropy(LossFunction):
    @output_loss
    def loss(self, output, targets):
        return -np.sum(np.nan_to_num(targets) * np.log(output),
                       axis=tuple(range(1, output.ndim)))

    @output_loss
    def d_loss(self, output, targets):
        return -np.nan_to_num(targets) / output

    @output_loss
    def d2_loss(self, output, targets):
        return np.nan_to_num(targets) / output ** 2

# TODO: categorization error


