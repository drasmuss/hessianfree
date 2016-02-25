import numpy as np


class Nonlinearity(object):
    """Base class for layer nonlinearities.

    :param boolean stateful: True if this nonlinearity has internal state
        (in which case it needs to return ``d_input``, ``d_state``, and
        ``d_output`` in :meth:`d_activation`; see :class:`Continuous` for an
        example)
    """

    def __init__(self, stateful=False):
        self.stateful = stateful

    def activation(self, x):
        """Apply the nonlinearity to the inputs.

        :param x: input to the nonlinearity
        """

        raise NotImplementedError()

    def d_activation(self, x, a):
        """Derivative of the nonlinearity with respect to the inputs.

        :param x: input to the nonlinearity
        :param a: output of ``activation(x)`` (can be used to more
            efficiently compute the derivative for some nonlinearities)"""

        raise NotImplementedError()

    def reset(self, init=None):
        """Reset the nonlinearity to initial conditions.

        :param init: override the default initial conditions with these values
        """

        pass


class Tanh(Nonlinearity):
    """Hyperbolic tangent function

    :math:`f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}`"""

    def __init__(self):
        super(Tanh, self).__init__()
        self.activation = np.tanh
        self.d_activation = lambda _, a: 1 - a ** 2


class Logistic(Nonlinearity):
    """Logistic sigmoid function

    :math:`f(x) = \\frac{1}{1 + e^{-x}}`

    Note: if scipy is installed then this will use the slightly
    faster :func:`~scipy:scipy.special.expit`
    """

    # TODO: get scipy intersphinx to work

    def __init__(self):
        super(Logistic, self).__init__()
        try:
            from scipy.special import expit
        except ImportError:
            def expit(x):
                return 1 / (1 + np.exp(-x))
        self.activation = expit
        self.d_activation = lambda _, a: a * (1 - a)


class Linear(Nonlinearity):
    """Linear activation function (passes inputs unchanged).

    :math:`f(x) = x`
    """

    def __init__(self):
        super(Linear, self).__init__()
        self.activation = lambda x: x
        self.d_activation = lambda x, _: np.ones_like(x)


class ReLU(Nonlinearity):
    """Rectified linear unit

    :math:`f(x) = max(x, 0)`

    :param max: an upper bound on activation to help avoid numerical errors
    """

    def __init__(self, max=1e10):
        super(ReLU, self).__init__()
        self.activation = lambda x: np.clip(x, 0, max)
        self.d_activation = lambda x, a: x == a


class Gaussian(Nonlinearity):
    """Gaussian activation function

    :math:`f(x) = e^{-x^2}`
    """

    def __init__(self):
        super(Gaussian, self).__init__()
        self.activation = lambda x: np.exp(-x ** 2)
        self.d_activation = lambda x, a: a * -2 * x


class Softmax(Nonlinearity):
    """Softmax activation function

    :math:`f(x_i) = \\frac{e^{x_i}}{\\sum_j{e^{x_j}}}`
    """

    def __init__(self):
        super(Softmax, self).__init__()

    def activation(self, x):
        e = np.exp(x - np.max(x, axis=-1)[..., None])
        # note: shifting everything down by max (doesn't change
        # result, but can help avoid numerical errors)

        e /= np.sum(e, axis=-1)[..., None]

        # clip to avoid numerical errors
        e[e < 1e-10] = 1e-10

        return e

    def d_activation(self, _, a):
        return a[..., None, :] * (np.eye(a.shape[-1], dtype=a.dtype) -
                                  a[..., None])


class SoftLIF(Nonlinearity):
    """SoftLIF activation function

    Based on
    Hunsberger, E. and Eliasmith, C. (2015). Spiking deep networks with LIF
    neurons. arXiv:1510.08829.

    .. math::
        f(x) = \\frac{amp}{
            \\tau_{ref} + \\tau_{RC}
            log(1 + \\frac{1}{\\sigma log(1 + e^{\\frac{x}{\\sigma}})})}

    Note: this is equivalent to :math:`LIF(SoftReLU(x))`

    :param float sigma: controls the smoothness of the nonlinearity threshold
    :param float tau_rc: LIF RC time constant
    :param float tau_ref: LIF refractory time constant
    :param float amp: scales output of nonlinearity
    """

    def __init__(self, sigma=1, tau_rc=0.02, tau_ref=0.002, amp=0.01):
        super(SoftLIF, self).__init__()
        self.sigma = sigma
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.amp = amp

    def softrelu(self, x):
        """Smoothed version of the ReLU nonlinearity."""

        y = x / self.sigma
        clip = (y < 34) & (y > -34)
        a = np.zeros_like(x)
        a[clip] = self.sigma * np.log1p(np.exp(y[clip]))
        a[y > 34] = y[y > 34]

        return a

    def lif(self, x):
        """LIF activation function."""

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
        d[j > 0] = (self.tau_rc * aa * aa) / (self.amp * jj * (jj + 1) *
                                              (1 + np.exp(-xx / self.sigma)))
        return d


class Continuous(Nonlinearity):
    """Creates a version of the base nonlinearity that operates in continuous
    time (filtering inputs with the given tau/dt).

    .. math::
        \\frac{ds}{dt} = \\frac{x - s}{\\tau}

        f(x) = base(s)

    :param base: nonlinear output function applied to the continuous state
    :type base: :class:`Nonlinearity`
    :param float tau: time constant of input filter (higher value means the
        internal state changes more slowly)
    :param float dt: simulation time step
    """

    def __init__(self, base, tau=1.0, dt=1.0):
        super(Continuous, self).__init__(stateful=True)
        self.base = base
        self.coeff = dt / tau

        self.reset()

    def activation(self, x):
        self.act_count += 1

        if self.state is None:
            self.state = np.zeros_like(x)
        self.state *= 1 - self.coeff
        self.state += x * self.coeff

        return self.base.activation(self.state)

    def d_activation(self, x, a):
        self.d_act_count += 1

        # note: x is not used here, this relies on self.state being implicitly
        # based on x (via self.activation()). hence the sanity check.
        assert self.act_count == self.d_act_count

        # note: need to create a new array each time (since other things
        # might be holding a reference to d_act)
        d_act = np.zeros((x.shape[0], x.shape[1], 3), dtype=x.dtype)

        # derivative of state with respect to input
        d_act[:, :, 0] = self.coeff

        # derivative of state with respect to previous state
        d_act[:, :, 1] = 1 - self.coeff

        # derivative of output with respect to state
        # TODO: fix this so it works if base returns matrices
        d_act[:, :, 2] = self.base.d_activation(self.state, a)

        return d_act

    def reset(self, init=None):
        """Reset state to zero (or ``init``)."""

        self.state = None if init is None else init.copy()
        self.act_count = 0
        self.d_act_count = 0


class Plant(Nonlinearity):
    """Base class for a plant that can be called to dynamically generate
    inputs for a network.

    See :func:`.demos.plant` for an example of this being used
    in practice."""

    def __init__(self, stateful=True):
        super(Plant, self).__init__(stateful=stateful)

        # self.shape gives the dimensions of the inputs generated by plant
        # [batch_size, sig_len, input_dim]
        self.shape = [0, 0, 0]

    def __call__(self, x):
        """Update the internal state of the plant based on input.

        :param x: the output of the last layer in the network on the
            previous timestep
        """
        raise NotImplementedError

    def get_vecs(self):
        """Return a tuple of the (inputs,targets) vectors generated by the
        plant since the last reset."""
        raise NotImplementedError

    def reset(self, init=None):
        """Reset the plant to initial state.

        :param init: override the default initial state with these values
        """
        raise NotImplementedError

    def activation(self, x):
        """This function only needs to be defined if the plant is going to
        be included as a layer in the network (as opposed to being handled
        by some external system)."""
        raise NotImplementedError

    def d_activation(self, x, a):
        """This function only needs to be defined if the plant is going to
        be included as a layer in the network (as opposed to being handled
        by some external system)."""
        raise NotImplementedError
