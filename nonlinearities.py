import numpy as np
from scipy.special import expit


class Nonlinearity(object):
    def __init__(self, use_activations=True):
        # use_activations denotes whether the d_activation function takes
        # activations as input or the same input as the activation function
        self.use_activations = use_activations

        # the derivative of state_t+1 with respect to state_t (for
        # nonlinearities that have state)
        self.d_state = None

    def activation(self, x):
        """Applies the nonlinearity to the inputs."""

        raise NotImplementedError()

    def d_activation(self, x):
        """The derivative of the nonlinearity with respect to the inputs."""

        # note: if self.use_activations is True, then the input here will
        # be self.activations(input), rather than the direct inputs
        raise NotImplementedError()

    def reset(self):
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
        super(Logistic, self).__init__()

    def activation(self, x):
        e = np.exp(x - np.max(x, axis=-1)[..., None])
        # note: shift everything down by max (doesn't change
        # result, but can help avoid numerical errors)

        e /= np.sum(e, axis=-1)[..., None]

        e[e < 1e-10] = 1e-10
        # clip to avoid numerical errors

        return e

    def d_activation(self, a):
        return a[..., None] * (np.eye(a.shape[-1]) - a[..., None, :])


class SoftLIF(Nonlinearity):
    def __init__(self, sigma=1, tau_rc=0.02, tau_ref=0.002, amp=1):
        super(Logistic, self).__init__(use_activations=False)
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
    def __init__(self, base, sig_len, tau=1.0, dt=1.0):
        super(Continuous, self).__init__(use_activations=False)
        self.base = base
        self.sig_len = sig_len
        self.tau = tau
        self.dt = dt
        self.coeff = dt / tau

        self.d_state = 1 - self.coeff

        self.reset()

    def activation(self, x):
        self.act_count += 1

        self.state += (x - self.state) * self.coeff

        return self.base.activation(self.state)

    def d_activation(self, x):
        self.d_act_count += 1

        # sanity check that self.inputs is in sync
        assert self.act_count == self.d_act_count

        act_d = self.base.d_activation(self.base.activation(self.state) if
                                       self.base.use_activations else
                                       self.inputs)

        state_d = self.coeff

        return act_d * state_d

    def reset(self):
        self.state = 0.0
        self.act_count = 0
        self.d_act_count = 0
