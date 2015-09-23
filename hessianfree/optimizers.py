from collections import defaultdict

import numpy as np


class Optimizer(object):
    # note: optimizers don't actually need to inherit from this class, this
    # just demonstrates the minimum structure that is expected
    def __init__(self, network):
        """Initialize the optimizer with whatever parameters are appropriate.

        :param network: the network that will be optimized (instance of
            FFNet or RNNet)
        """
        self.net = network

    def compute_update(self, printing=False):
        """Compute a weight update for the current batch.

        It can be assumed that the batch has already been stored in net.inputs
        and net.targets, and the nonlinearity activations/derivatives for
        the batch are cached in net.activations and net.d_activations.

        :param printing: if True, print out data about the optimization
        """
        raise NotImplementedError()


class HessianFree(Optimizer):
    def __init__(self, network, CG_iter=250, init_damping=1,
                 struc_damping=None, plotting=True):
        """Use Hessian-free optimization to compute the weight update.

        Based on
        Martens, J. (2010). Deep learning via Hessian-free optimization. In
        Proceedings of the 27th International Conference on Machine Learning.

        :param CG_iter: the maximum number of CG iterations to run per epoch
        :param init_damping: the initial value of the Tikhonov damping
        :param struc_damping: scale on structural damping, relative to
            Tikhonov damping (only used in recurrent nets)
        :param plotting: if True then collect data for plotting (actual
            plotting handled in parent network)
        """

        super(HessianFree, self).__init__(network)
        self.CG_iter = CG_iter
        self.init_delta = np.zeros_like(network.W)
        self.damping = init_damping
        self._struc_damping = struc_damping

        self.plotting = plotting
        self.plots = defaultdict(list)

    def compute_update(self, printing=False):
        err = self.net.error()  # note: don't reuse previous error (diff batch)

        # compute gradient
        grad = self.net.calc_grad()

        if printing:
            print "initial err", err
            print "grad norm", np.linalg.norm(grad)

        # run CG
        deltas = self.conjugate_gradient(self.init_delta * 0.95, grad,
                                         iters=self.CG_iter)

        if printing:
            print "CG steps", deltas[-1][0]

        self.init_delta = deltas[-1][1]  # note: don't backtrack this

        # CG backtracking
        new_err = np.inf
        for j in range(len(deltas) - 1, -1, -1):
            prev_err = self.net.error(self.net.W + deltas[j][1])
            # note: we keep using the cached inputs, not rerunning the plant
            # (if there is one). that is, we are evaluating whether the input
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
            print "using iteration", deltas[j + 1][0]
            print "backtracked err", new_err

        # update damping parameter (compare improvement predicted by
        # quadratic model to the actual improvement in the error)
        denom = (0.5 * np.dot(delta,
                              self.net.calc_G(delta, damping=self.damping)) +
                 np.dot(grad, delta))

        improvement_ratio = (new_err - err) / denom if denom != 0 else 1
        if improvement_ratio < 0.25:
            self.damping *= 1.5
        elif improvement_ratio > 0.75:
            self.damping *= 0.66

        if printing:
            print "improvement_ratio", improvement_ratio
            print "damping", self.damping

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
            print "min_improv", min_improv
            print "l_rate", l_rate
            print "l_rate err", new_err
            print "improvement", new_err - err

        if self.plotting:
            self.plots["training error"] += [new_err]
            self.plots["learning rate"] += [l_rate]
            self.plots["damping"] += [self.damping]
            self.plots["CG iterations"] += [deltas[-1][0]]

        return l_rate * delta

    def conjugate_gradient(self, init_delta, grad, iters=250):
        """Find minimum of quadratic approximation using conjugate gradient
        algorithm."""

        store_iter = 5
        store_mult = 1.3
        deltas = []
        vals = np.zeros(iters, dtype=self.net.dtype)

        base_grad = -grad
        delta = init_delta
        residual = base_grad - self.net.calc_G(init_delta, damping=self.damping)
        res_norm = np.dot(residual, residual)
        direction = residual.copy()

        if self.net.debug:
            self.net.check_grad(grad)

        for i in range(iters):
            if self.net.debug:
                print "-" * 20
                print "CG iteration", i
                print "delta norm", np.linalg.norm(delta)
                print "direction norm", np.linalg.norm(direction)

            G_dir = self.net.calc_G(direction, damping=self.damping)

            # calculate step size
            step = res_norm / np.dot(direction, G_dir)

            if self.net.debug:
                print "step", step

                self.net.check_G(G_dir, direction, self.damping)

                assert np.isfinite(step)
                assert step >= 0
                assert (np.linalg.norm(np.dot(direction, G_dir)) >=
                        np.linalg.norm(np.dot(direction,
                                              self.net.calc_G(direction,
                                                              damping=0))))

            # update weight delta
            delta += step * direction

            # update residual
            residual -= step * G_dir
            new_res_norm = np.dot(residual, residual)

            if new_res_norm < 1e-20:
                # early termination (mainly to prevent numerical errors);
                # if this ever triggers, it's probably because the minimum
                # gap in the normal termination condition (below) is too low.
                # this only occurs on really simple problems
                break

            # update direction
            beta = new_res_norm / res_norm
            direction *= beta
            direction += residual

            res_norm = new_res_norm

            if i == store_iter:
                deltas += [(i, np.copy(delta))]
                store_iter = int(store_iter * store_mult)

            # martens termination conditions
            vals[i] = -0.5 * np.dot(residual + base_grad, delta)

            gap = max(int(0.1 * i), 10)

            if self.net.debug:
                print "termination val", vals[i]

            if (i > gap and vals[i - gap] < 0 and
                    (vals[i] - vals[i - gap]) / vals[i] < 5e-6 * gap):
                break

        deltas += [(i, np.copy(delta))]

        return deltas

    @property
    def struc_damping(self):
        if self._struc_damping is None:
            return None
        return self.damping * self._struc_damping


class SGD(Optimizer):
    def __init__(self, network, l_rate=1, plotting=False):
        """Compute weight update using first-order gradient descent.

        :param l_rate: learning rate to apply to weight updates
        :param plotting: if True then collect data for plotting (actual
            plotting handled in parent network)
        """

        super(SGD, self).__init__(network)

        self.l_rate = l_rate

        self.plotting = plotting
        self.plots = defaultdict(list)

    def compute_update(self, printing=False):
        grad = self.net.calc_grad()

        if self.net.debug:
            self.net.check_grad(grad)

        if printing:
            train_err = self.net.error()
            print "training error", train_err

            # note: for SGD we'll just do the plotting when we print (since
            # we're going to be doing a lot more, and smaller, updates)
            if self.plotting:
                self.plots["training error"] += [train_err]

        return -self.l_rate * grad
