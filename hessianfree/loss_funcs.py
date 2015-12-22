from functools import wraps

import numpy as np


class LossFunction:
    """Defines a loss function that maps nonlinearity activations to error."""

    def loss(self, activities, targets):
        """Computes the loss for each unit in the network.

        Note that most loss functions are only based on the output of the
        final layer, activities[-1]. However, we pass the activities of all
        layers here so that loss functions can include things like
        sparsity constraints. Targets, however, are only defined for the
        output layer.

        Targets can be defined as ``np.nan``, which will be translated
        into zero error.

        :param list activities: output activations of each layer
        :param targets: target activation values for last layer
        :type targets: :class:`~numpy:numpy.ndarray`
        """
        raise NotImplementedError()

    def d_loss(self, activities, targets):
        """First derivative of loss function (with respect to activities)."""
        raise NotImplementedError()

    def d2_loss(self, activities, targets):
        """Second derivative of loss function (with respect to activities)."""
        raise NotImplementedError()

    def batch_loss(self, activities, targets):
        """Utility function to compute a single loss value for the network
        (taking the mean across batches and summing across and within layers).
        """

        losses = self.loss(activities, targets)
        return np.sum([np.true_divide(np.sum(l), l.shape[0]) for l in losses
                       if l is not None])


def output_loss(func):
    """Convenience decorator that takes a loss defined for the output layer
    and converts it into the more general form in terms of all layers."""

    @wraps(func)
    def wrapped_loss(self, activities, targets):
        result = [None for _ in activities[:-1]]
        result += [func(self, activities[-1], targets)]

        return result

    return wrapped_loss


class SquaredError(LossFunction):
    """Squared error

    :math:`\\frac{1}{2} \\sum(output - target)^2`
    """

    @output_loss
    def loss(self, output, targets):
        return np.sum(np.nan_to_num(output - targets) ** 2,
                      axis=tuple(range(1, output.ndim))) / 2

    @output_loss
    def d_loss(self, output, targets):
        return np.nan_to_num(output - targets)

    @output_loss
    def d2_loss(self, output, _):
        return np.ones_like(output)


class CrossEntropy(LossFunction):
    """Cross-entropy error

    :math:`-\\sum(target * log(output))`
    """
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
    """Classification error

    :math:`argmax(output) \\neq argmax(target)`

    Note: ``d_loss`` and ``d2_loss`` are not defined; classification error
    should only be used for validation, which doesn't require either.
    """

    @output_loss
    def loss(self, output, targets):
        return np.logical_and(np.argmax(output, axis=-1) !=
                              np.argmax(targets, axis=-1),
                              np.logical_not(np.isnan(np.sum(targets,
                                                             axis=-1))))


class StructuralDamping(LossFunction):
    """Applies structural damping, which penalizes layers for having
    highly variable output activity.

    Note: this is not exactly the same as the structural damping in
    Martens (2010), because it is applied on the output side of the
    nonlinearity (meaning that this error will be filtered through
    ``d_activations`` during the backwards propagation).

    :param float weight: scale on structural damping relative to other losses
    :param list layers: indices specifying which layers will have the
        damping applied (defaults to all except first/last layers)
    :param optimizer: if provided, the weight on structural damping will be
        scaled relative to the ``damping`` attribute in the optimizer
        (so that any processes dynamically adjusting the damping during the
        optimization will also affect the structural damping)
    :type optimizer: :class:`~hessianfree.optimizers.Optimizer`
    """

    def __init__(self, weight, layers=None, optimizer=None):
        self.weight = weight
        self.layers = (np.index_exp[1:-1] if layers is None else
                       np.asarray(layers))
        self.opt = optimizer

    def loss(self, activities, _):
        return [None for _ in activities]

    def d_loss(self, activities, _):
        return [None for _ in activities]

    def d2_loss(self, activities, _):
        opt_damp = 1 if self.opt is None else getattr(self.opt, "damping", 1)

        d2_loss = [None for _ in activities]
        for l in np.arange(len(activities))[self.layers]:
            d2_loss[l] = np.ones_like(activities[l]) * self.weight * opt_damp

        return d2_loss


class SparseL1(LossFunction):
    """Imposes L1 sparsity constraint on nonlinearity activations.

    :param float weight: relative weight of sparsity constraint
    :param list layers: indices specifying which layers will have the
        sparsity constraint applied (defaults to all except first/last layers)
    :param float target: target activation level for nonlinearities
    """
    def __init__(self, weight, layers=None, target=0.0):
        # TODO: is it valid to apply L1 sparsity to HF, given that CG is meant
        # to optimize quadratic loss functions?

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
    """Imposes L2 sparsity constraint on nonlinearity activations.

    :param float weight: relative weight of sparsity constraint
    :param list layers: indices specifying which layers will have the
        sparsity constraint applied (defaults to all except first/last layers)
    :param float target: target activation level for nonlinearities
    """

    # note: this is similar to structural damping, except we also include it
    # in the first derivative
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
            d2_loss[l] = np.ones_like(activities[l]) * self.weight

        return d2_loss


class LossSet(LossFunction):
    """Combines several loss functions into one (e.g., combining
    :class:`SquaredError` and :class:`SparseL2`).  It doesn't need to be
    created directly; a list of loss functions can be passed to
    :class:`.FFNet`/:class:`.RNNet` and a LossSet will be created
    automatically.

    :param list set: list of :class:`LossFunction`"""

    def __init__(self, set):
        self.set = set

    def group_func(self, func_name, activities, targets):
        """Computes the given function for each :class:`LossFunction` in the
        set, and sums the result."""

        # apply each of the loss functions
        result = [getattr(s, func_name)(activities, targets)
                  for s in self.set]

        # sum the losses for each layer across the loss functions
        result = [np.sum([s[i] for s in result if s[i] is not None], axis=0)
                  for i in range(len(activities))]

        # convert 0.0's (from np.sum([])) back to None
        result = [None if (isinstance(x, float) and x == 0.0) else x
                  for x in result]

        return result

    def loss(self, activities, targets):
        return self.group_func("loss", activities, targets)

    def d_loss(self, activities, targets):
        return self.group_func("d_loss", activities, targets)

    def d2_loss(self, activities, targets):
        return self.group_func("d2_loss", activities, targets)
