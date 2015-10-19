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


class Sparse(LossFunction):
    def __init__(self, base, weight, layers=None, target=0.0):
        """Imposes L1 sparsity constraint on top of the base loss function.

        :param base: loss function defining error in terms of target output
        :param weight: relative weight of sparsity compared to target loss
            (1.0 would give an equal weighting)
        :param layers: list of integers specifying which layers will have the
            sparsity constraint imposed (if None, will be applied to all
            except first/last layers)
        :param target: target activation level for nonlinearities
        """

        self.base = base
        self.weight = weight
        self.layers = layers
        self.target = target

    def loss(self, activities, targets):
        if self.layers is None:
            layers = np.arange(1, len(activities) - 1)
        else:
            layers = self.layers

        loss = self.base.loss(activities, targets)

        for l in layers:
            loss[l] += self.weight * np.abs(activities[l] - self.target)

        return loss

    def d_loss(self, activities, targets):
        if self.layers is None:
            layers = np.arange(1, len(activities) - 1)
        else:
            layers = self.layers

        d_loss = self.base.d_loss(activities, targets)

        for l in layers:
            d_loss[l] += self.weight * ((activities[l] > self.target) * 2 - 1)

        return d_loss

    def d2_loss(self, activities, targets):
        # second derivative is zero

        return self.base.d2_loss(activities, targets)
