from __future__ import print_function

import warnings
from collections import defaultdict

import numpy as np


class Optimizer(object):
    """Base class for optimizers.

    Each optimizer has a ``self.net`` parameter that will be set
    automatically when the optimizer is added to a network (referring
    to that network)."""

    def __init__(self):
        self.net = None

    def compute_update(self, printing=False):
        """Compute a weight update for the current batch.

        It can be assumed that the batch has already been stored in
        ``net.inputs`` and ``net.targets``, and the nonlinearity
        activations/derivatives for the batch are cached in ``net.activations``
        and ``net.d_activations``.

        :param bool printing: if True, print out data about the optimization
        """
        raise NotImplementedError()


class HessianFree(Optimizer):
    """Use Hessian-free optimization to compute the weight update.

    :param int CG_iter: maximum number of CG iterations to run per epoch
    :param float init_damping: the initial value of the Tikhonov damping
    :param bool plotting: if True then collect data for plotting (actual
        plotting handled in parent network)
    """

    def __init__(self, CG_iter=250, init_damping=1, plotting=True):
        super(HessianFree, self).__init__()

        self.CG_iter = CG_iter
        self.init_delta = None
        self.damping = init_damping

        self.plotting = plotting
        self.plots = defaultdict(list)

    def compute_update(self, printing=False):
        """Compute a weight update for the current batch.

        :param bool printing: if True, print out data about the optimization
        """
        err = self.net.error()  # note: don't reuse previous error (diff batch)

        # compute gradient
        grad = self.net.calc_grad()

        if printing:
            print("initial err", err)
            print("grad norm", np.linalg.norm(grad))

        # run CG
        if self.init_delta is None:
            self.init_delta = np.zeros_like(self.net.W)
        deltas = self.conjugate_gradient(self.init_delta * 0.95, grad,
                                         iters=self.CG_iter,
                                         printing=printing and self.net.debug)

        if printing:
            print("CG steps", deltas[-1][0])

        self.init_delta = deltas[-1][1]  # note: don't backtrack this

        # CG backtracking
        new_err = np.inf
        for j in range(len(deltas) - 1, -1, -1):
            prev_err = self.net.error(self.net.W + deltas[j][1])
            # note: we keep using the cached inputs, not rerunning the plant
            # (if there is one). that is, we are evaluating whether the update
            # improves on those inputs, not whether it improves the overall
            # objective. we could do the latter instead, but it makes things
            # more prone to instability.
            if prev_err > new_err:
                break
            delta = deltas[j][1]
            new_err = prev_err
        else:
            j -= 1

        if printing:
            print("using iteration", deltas[j + 1][0])
            print("backtracked err", new_err)

        # update damping parameter (compare improvement predicted by
        # quadratic model to the actual improvement in the error)
        quad = (0.5 * np.dot(self.calc_G(delta, damping=self.damping),
                             delta) +
                np.dot(grad, delta))

        improvement_ratio = ((new_err - err) / quad) if quad != 0 else 1
        if improvement_ratio < 0.25:
            self.damping *= 1.5
        elif improvement_ratio > 0.75:
            self.damping *= 0.66

        if printing:
            print("improvement_ratio", improvement_ratio)
            print("damping", self.damping)

        # line search to find learning rate
        l_rate = 1.0
        min_improv = min(1e-2 * np.dot(grad, delta), 0)
        for _ in range(60):
            # check if the improvement is greater than the minimum
            # improvement we would expect based on the starting gradient
            if new_err <= err + l_rate * min_improv:
                break

            l_rate *= 0.8
            new_err = self.net.error(self.net.W + l_rate * delta)
        else:
            # no good update, so skip this iteration
            l_rate = 0.0
            new_err = err

        if printing:
            print("min_improv", min_improv)
            print("l_rate", l_rate)
            print("l_rate err", new_err)
            print("improvement", new_err - err)

        if self.plotting:
            self.plots["training error (log)"] += [new_err]
            self.plots["learning rate"] += [l_rate]
            self.plots["damping (log)"] += [self.damping]
            self.plots["CG iterations"] += [deltas[-1][0]]
            self.plots["backtracked steps"] += [deltas[-1][0] -
                                                deltas[j + 1][0]]

        return l_rate * delta

    def conjugate_gradient(self, init_delta, grad, iters=250, printing=False):
        """Find minimum of quadratic approximation using conjugate gradient
        algorithm."""

        if self.net.debug:
            self.net.check_grad(grad)

        store_iter = 5
        store_mult = 1.3
        deltas = []
        grad = -grad  # note negative, some CG algorithms are flipped
        vals = np.zeros(iters, dtype=self.net.dtype)

        if self.net.use_GPU:
            from pycuda import gpuarray
            base_grad = gpuarray.to_gpu(grad)
            delta = gpuarray.to_gpu(init_delta)
            G_dir = gpuarray.zeros(grad.shape, dtype=self.net.dtype)
            self.calc_G = self.net.GPU_calc_G

            def dot(a, b):
                return gpuarray.dot(a, b).get()

            def get(x):
                return x.get(pagelocked=True)
        else:
            base_grad = grad
            delta = init_delta
            G_dir = np.zeros_like(grad)
            self.calc_G = self.net.calc_G
            dot = np.dot
            get = np.copy

        residual = base_grad.copy()
        residual -= self.calc_G(delta, damping=self.damping, out=G_dir)
        res_norm = dot(residual, residual)
        direction = residual.copy()

        for i in range(iters):
            if printing:
                print("-" * 20)
                print("CG iteration", i)
                print("delta norm", np.linalg.norm(get(delta)))
                print("direction norm", np.linalg.norm(get(direction)))

            self.calc_G(direction, damping=self.damping, out=G_dir)

            # calculate step size
            step = res_norm / dot(direction, G_dir)

            if not np.isfinite(step):
                warnings.warn("Non-finite step value (%f)" % step)
                break

            if printing:
                print("G_dir norm", np.linalg.norm(get(G_dir)))
                print("step", step)

            if self.net.debug:
                tmp_G_dir = get(G_dir)
                tmp_dir = get(direction)
                self.net.check_G(tmp_G_dir, tmp_dir, self.damping)

                assert np.isfinite(step)
                assert step >= 0
                assert (np.linalg.norm(np.dot(tmp_dir, tmp_G_dir)) >=
                        np.linalg.norm(np.dot(tmp_dir,
                                              self.net.calc_G(tmp_dir,
                                                              damping=0))))

            # update weight delta
            delta += step * direction

            # update residual
            residual -= step * G_dir
            new_res_norm = dot(residual, residual)

            if new_res_norm < 1e-20:
                # early termination (mainly to prevent numerical errors);
                # the main termination condition is below.
                break

            # update direction
            beta = new_res_norm / res_norm
            direction *= beta
            direction += residual

            res_norm = new_res_norm

            # store deltas for backtracking
            if i == store_iter:
                deltas += [(i, get(delta))]
                store_iter = int(store_iter * store_mult)

            # martens termination conditions
            vals[i] = -0.5 * dot(residual + base_grad, delta)

            gap = max(int(0.1 * i), 10)

            if printing:
                print("termination val", vals[i])

            if (i > gap and vals[i - gap] < 0 and
                    (vals[i] - vals[i - gap]) / vals[i] < 5e-6 * gap):
                break

        deltas += [(i, get(delta))]

        return deltas


class SGD(Optimizer):
    """Compute weight update using first-order gradient descent.

    :param l_rate: learning rate to apply to weight updates
    :param plotting: if True then collect data for plotting (actual
        plotting handled in parent network)"""

    def __init__(self, l_rate=1, plotting=False):
        super(SGD, self).__init__()

        self.l_rate = l_rate

        self.plotting = plotting
        self.plots = defaultdict(list)

    def compute_update(self, printing=False):
        """Compute a weight update for the current batch.

        :param bool printing: if True, print out data about the optimization
        """
        grad = self.net.calc_grad()

        if self.net.debug:
            self.net.check_grad(grad)

        if printing:
            train_err = self.net.error()
            print("training error", train_err)

            # note: for SGD we'll just do the plotting when we print (since
            # we're going to be doing a lot more, and smaller, updates)
            if self.plotting:
                self.plots["training error"] += [train_err]

        return -self.l_rate * grad
