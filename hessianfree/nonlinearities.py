import numpy as np
from scipy.special import expit


class Nonlinearity(object):
    def __init__(self, use_activations=True, stateful=False):
        # use_activations denotes whether the d_activation function takes
        # activations as input or the same input as the activation function
        self.use_activations = use_activations

        # True if this nonlinearity has internal state (in which case it
        # also needs to define d_inputs and d_state)
        self.stateful = stateful

    def activation(self, x):
        """Apply the nonlinearity to the inputs."""

        raise NotImplementedError()

    def d_activation(self, x):
        """Derivative of the nonlinearity with respect to the inputs."""

        # note: if self.use_activations is True, then the input here will
        # be self.activations(input), rather than the direct inputs
        raise NotImplementedError()

    def reset(self):
        """Reset the internal state of the nonlinearity."""

        pass


class Logistic(Nonlinearity):
    def __init__(self):
        super(Logistic, self).__init__()
        self.activation = expit
        self.d_activation = lambda a: a * (1 - a)


class Tanh(Nonlinearity):
    def __init__(self):
        super(Tanh, self).__init__()
        self.activation = np.tanh
        self.d_activation = lambda a: 1 - a ** 2


class Linear(Nonlinearity):
    def __init__(self):
        super(Linear, self).__init__()
        self.activation = lambda x: x
        self.d_activation = np.ones_like


class ReLU(Nonlinearity):
    def __init__(self):
        super(ReLU, self).__init__()
        self.activation = lambda x: np.maximum(0, x)
        self.d_activation = lambda a: a > 0


class Softmax(Nonlinearity):
    def __init__(self):
        super(Softmax, self).__init__()

    def activation(self, x):
        e = np.exp(x - np.max(x, axis=-1)[..., None])
        # note: shift everything down by max (doesn't change
        # result, but can help avoid numerical errors)

        e /= np.sum(e, axis=-1)[..., None]

        e[e < 1e-10] = 1e-10
        # clip to avoid numerical errors

        return e

    def d_activation(self, a):
        return a[..., None, :] * (np.eye(a.shape[-1]) - a[..., None])


class SoftLIF(Nonlinearity):
    def __init__(self, sigma=1, tau_rc=0.02, tau_ref=0.002, amp=0.01):
        super(SoftLIF, self).__init__(use_activations=False)
        self.sigma = sigma
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.amp = amp

    def softrelu(self, x):
        y = x / self.sigma
        clip = (y < 34) & (y > -34)
        a = np.zeros_like(x)
        a[clip] = self.sigma * np.log1p(np.exp(y[clip]))
        a[y > 34] = y[y > 34]

        return a

    def lif(self, x):
        a = np.zeros_like(x)
        a[x > 0] = self.amp / (self.tau_ref +
                               self.tau_rc * np.log1p(1. / x[x > 0]))
        return a

    def activation(self, x):
        return self.lif(self.softrelu(x))

    def d_activation(self, x):
        j = self.softrelu(x)
        r = self.lif(j)

        d = np.zeros_like(j)
        rr, jj, xx = r[j > 0], j[j > 0], x[j > 0]
        d[j > 0] = (self.tau_rc * rr * rr) / (
            self.amp * jj * (jj + 1) * (1 + np.exp(-xx / self.sigma)))
        return d


class Continuous(Nonlinearity):
    def __init__(self, base, tau=1.0, dt=1.0):
        super(Continuous, self).__init__(use_activations=False,
                                         stateful=True)
        self.base = base
        self.coeff = dt / tau

#         # derivative of state with respect to previous state
#         self.d_state = np.diag(1 - self.coeff).astype(np.float32)
#
#         # derivative of state with respect to input
#         self.d_input = np.diag(self.coeff).astype(np.float32)

        self.reset()

    def activation(self, x):
        self.act_count += 1

        # s_{t+1} = (1-c)s + cx
        self.state += (x - self.state) * self.coeff

        return self.base.activation(self.state)

    def d_activation(self, x):
        self.d_act_count += 1

        # sanity check that state is in sync
        assert self.act_count == self.d_act_count

        # note: x is not used here, this relies on self.state being implicitly
        # based on x (via self.activation()). hence the sanity check.

        act_d = self.base.d_activation(self.base.activation(self.state) if
                                       self.base.use_activations else
                                       self.state)[..., None]

        # TODO: fix this so it works if act_d returns matrices

        d_input = np.resize(self.coeff,
                            (x.shape[0], x.shape[1], 1))

        d_state = np.resize(1 - self.coeff,
                            (x.shape[0], x.shape[1], 1))

        return np.concatenate((d_input, d_state, act_d), axis=-1)

    def reset(self):
        self.state = 0.0
        self.act_count = 0
        self.d_act_count = 0
