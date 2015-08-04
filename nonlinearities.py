import numpy as np
from scipy.special import expit


class Logistic:
    def __init__(self):
        self.activation = expit
        self.d_activation = lambda a: a * (1 - a)
        self.use_activations = True


class Tanh:
    def __init__(self):
        self.activation = np.tanh
        self.d_activation = lambda a: 1 - a ** 2
        self.use_activations = True


class Linear:
    def __init__(self):
        self.activation = lambda x: x
        self.d_activation = np.ones_like
        self.use_activations = True


class ReLU:
    def __init__(self):
        self.activation = lambda x: np.maximum(0, x)
        self.d_activation = lambda a: a > 0
        self.use_activations = True


class Softmax:
    def __init__(self):
        self.use_activations = True

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


class SoftLIF:
    def __init__(self, sigma=1, tau_rc=0.02, tau_ref=0.002, amp=1):
        self.sigma = sigma
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.amp = amp
        self.use_activations = False

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
