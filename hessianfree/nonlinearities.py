import numpy as np
from scipy.special import expit


class Nonlinearity(object):
    def __init__(self, stateful=False):
        # True if this nonlinearity has internal state (in which case it
        # needs to return d_input, d_state, and d_output in d_activations;
        # see Continuous for an example)
        self.stateful = stateful

    def activation(self, x):
        """Apply the nonlinearity to the inputs."""

        raise NotImplementedError()

    def d_activation(self, x, a):
        """Derivative of the nonlinearity with respect to the inputs."""

        # note: a is self.activation(x), which can be used to more efficiently
        # compute the derivative for some nonlinearities
        raise NotImplementedError()

    def reset(self, init=None):
        """Reset the nonlinearity to initial conditions."""

        pass

# TODO: make in-place nonlinearities if this ever becomes a bottleneck


class Logistic(Nonlinearity):
    def __init__(self):
        super(Logistic, self).__init__()
        self.activation = expit
        self.d_activation = lambda _, a: a * (1 - a)


class Tanh(Nonlinearity):
    def __init__(self):
        super(Tanh, self).__init__()
        self.activation = np.tanh
        self.d_activation = lambda _, a: 1 - a ** 2


class Linear(Nonlinearity):
    def __init__(self):
        super(Linear, self).__init__()
        self.activation = lambda x: x
        self.d_activation = lambda x, _: np.ones_like(x)


class ReLU(Nonlinearity):
    def __init__(self):
        super(ReLU, self).__init__()
        self.activation = lambda x: np.maximum(0, x)
        self.d_activation = lambda _, a: a > 0


class Gaussian(Nonlinearity):
    def __init__(self):
        super(Gaussian, self).__init__()
        self.activation = lambda x: np.exp(-x ** 2)
        self.d_activation = lambda x, a: a * -2 * x


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

    def d_activation(self, _, a):
        return a[..., None, :] * (np.eye(a.shape[-1], dtype=np.float32) -
                                  a[..., None])


class SoftLIF(Nonlinearity):
    def __init__(self, sigma=1, tau_rc=0.02, tau_ref=0.002, amp=0.01):
        super(SoftLIF, self).__init__()
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

    def d_activation(self, x, a):
        j = self.softrelu(x)

        d = np.zeros_like(j)
        aa, jj, xx = a[j > 0], j[j > 0], x[j > 0]
        d[j > 0] = (self.tau_rc * aa * aa) / (
            self.amp * jj * (jj + 1) * (1 + np.exp(-xx / self.sigma)))
        return d


class Continuous(Nonlinearity):
    """Creates a version of the base nonlinearity that operates in continuous
    time (filtering inputs with the given tau/dt)."""

    def __init__(self, base, tau=1.0, dt=1.0):
        super(Continuous, self).__init__(stateful=True)
        self.base = base
        self.coeff = dt / tau

        self.reset()

    def activation(self, x):
        self.act_count += 1

        # s_{t+1} = (1-c)s + cx
        self.state += (x - self.state) * self.coeff

        return self.base.activation(self.state)

    def d_activation(self, x, a):
        self.d_act_count += 1

        # note: x is not used here, this relies on self.state being implicitly
        # based on x (via self.activation()). hence the sanity check.
        assert self.act_count == self.d_act_count

        # note: need to create a new array each time (since other things
        # might be holding a reference to d_act)
        d_act = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.float32)

        # derivative of state with respect to input
        d_act[:, :, 0] = self.coeff

        # derivative of state with respect to previous state
        d_act[:, :, 1] = 1 - self.coeff

        # derivative of output with respect to state
        # TODO: fix this so it works if base returns matrices
        d_act[:, :, 2] = self.base.d_activation(self.state, a)

        return d_act

    def reset(self, init=None):
        self.state = 0.0 if init is None else init.copy()
        self.act_count = 0
        self.d_act_count = 0
