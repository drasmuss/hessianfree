from functools import wraps

import numpy as np


class LossFunction:
    """Defines a loss function that maps nonlinearity activations to error."""

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

    def batch_mean(self, losses):
        """Utility function to compute a single loss value (taking the mean
        across batches and summing the loss in each layer)."""

        return np.sum([np.sum(l, dtype=np.float32) / l.shape[0] for l in losses
                       if l is not None])


def output_loss(func):
    """Convenience wrapper that takes a loss defined for the output layer
    and converts it into the more general form in terms of all layers."""

    @wraps(func)
    def wrapped_loss(self, activities, targets):
        result = [None for _ in activities[:-1]]
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


class ClassificationError(LossFunction):
    @output_loss
    def loss(self, output, targets):
        return np.argmax(output, axis=-1) == np.argmax(targets, axis=-1)

    # note: not defining d_loss or d2_loss; classification error should only
    # be used for validation, which doesn't require either


class SparseL1(LossFunction):
    def __init__(self, weight, layers=None, target=0.0):
        """Imposes L1 sparsity constraint on nonlinearity activations.

        :param weight: relative weight of sparsity constraint
        :param layers: list of integers specifying which layers will have the
            sparsity constraint imposed (if None, will be applied to all
            except first/last layers)
        :param target: target activation level for nonlinearities
        """

        self.weight = weight
        self.layers = np.index_exp[1:-1] if layers is None else layers
        self.target = target

    def loss(self, activities, _):
        loss = [None for _ in activities]
        for l in np.arange(len(activities))[self.layers]:
            loss[l] = self.weight * np.abs(activities[l] - self.target)

        return loss

    def d_loss(self, activities, _):
        d_loss = [None for _ in activities]
        for l in np.arange(len(activities))[self.layers]:
            d_loss[l] = self.weight * ((activities[l] > self.target) * 2 - 1)

        return d_loss

    def d2_loss(self, activities, _):
        return [None for _ in activities]


class SparseL2(LossFunction):
    """Imposes L2 sparsity constraint on nonlinearity activations."""

    # note: this is almost the same as the standard structural damping from
    # martens. one difference is we include it in the first derivative. more
    # significantly, this damping is applied on the output of the nonlinearity
    # (meaning it will be influenced by d_activations), rather than applied
    # at input side.
    # TODO: test how well this works relative to standard structural damping

    def __init__(self, weight, layers=None, target=0.0):
        self.weight = weight
        self.layers = np.index_exp[1:-1] if layers is None else layers
        self.target = target

    def loss(self, activities, _):
        loss = [None for _ in activities]
        for l in np.arange(len(activities))[self.layers]:
            loss[l] = 0.5 * self.weight * (activities[l] - self.target) ** 2

        return loss

    def d_loss(self, activities, _):
        d_loss = [None for _ in activities]
        for l in np.arange(len(activities))[self.layers]:
            d_loss[l] = self.weight * (activities[l] - self.target)

        return d_loss

    def d2_loss(self, activities, _):
        d2_loss = [None for _ in activities]
        for l in np.arange(len(activities))[self.layers]:
            d2_loss[l] = self.weight

        return d2_loss


class LossSet(LossFunction):
    """This combines several loss functions into one (e.g. combining
    SquaredError and Sparse).  It doesn't need to be created directly, a list
    of loss functions can be passed to FFNet(..., loss_type=[...]) and a
    LossSet will be created automatically."""

    def __init__(self, set):
        self.set = set

    def group_func(self, func_name, activities, targets):
        # apply each of the loss functions
        result = [getattr(s, func_name)(activities, targets)
                  for s in self.set]

        # sum the losses for each layer across the loss functions
        result = [np.sum([s[i] for s in result if s[i] is not None], axis=0)
                  for i in range(len(activities))]

        # convert 0.0's (from np.sum([])) back to None
        result = [None if isinstance(x, float) else x for x in result]

        return result

    def loss(self, activities, targets):
        return self.group_func("loss", activities, targets)

    def d_loss(self, activities, targets):
        return self.group_func("d_loss", activities, targets)

    def d2_loss(self, activities, targets):
        return self.group_func("d2_loss", activities, targets)
