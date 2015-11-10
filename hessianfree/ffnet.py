"""Implementation of feedforward network, including Gauss-Newton approximation
for use in Hessian-free optimization.

Author: Daniel Rasmussen (drasmussen@princeton.edu)

Based on
Martens, J. (2010). Deep learning via Hessian-free optimization. In Proceedings
of the 27th International Conference on Machine Learning.
"""


from collections import defaultdict, OrderedDict
from copy import deepcopy
from functools import partial
import pickle

import numpy as np

import hessianfree as hf


class FFNet(object):
    def __init__(self, shape, layers=hf.nl.Logistic(), conns=None,
                 loss_type=hf.loss_funcs.SquaredError(), W_init_params={},
                 use_GPU=False, load_weights=None, debug=False):
        """Initialize the parameters of the network.

        :param shape: list specifying the number of neurons in each layer
        :param layers: nonlinearity (or list of nonlinearities) to use
            in each layer
        :param conns: dict of the form {layer_x:[layer_y, layer_z], ...}
            specifying the connections between layers (default is to connect in
            series)
        :param loss_type: loss function (or list of loss functions) used to
            evaluate network (see loss_funcs.py)
        :param W_init_params: parameter dict passed to init_weights() (see
            parameter descriptions in that function)
        :param use_GPU: run certain operations on GPU to speed them up
            (requires PyCUDA to be installed)
        :param load_weights: load initial weights from given array or filename
        :param debug: activates (expensive) features to help with debugging
        """

        self.use_GPU = use_GPU
        self.debug = debug
        self.shape = shape
        self.n_layers = len(shape)
        self.dtype = np.float64 if debug else np.float32

        self.epoch = None
        # note: this isn't used internally, it is just here so that an
        # external process with a handle to this object can tell what epoch
        # it is on

        self.inputs = None
        self.targets = None

        # initialize layer nonlinearities
        if not isinstance(layers, (list, tuple)):
            layers = [deepcopy(layers) for _ in range(self.n_layers)]
            layers[0] = hf.nl.Linear()

        if len(layers) != len(shape):
            raise ValueError("Number of nonlinearities (%d) does not match "
                             "number of layers (%d)" %
                             (len(layers), len(shape)))

        self.layers = []
        for t in layers:
            if isinstance(t, str):
                # look up the nonlinearity with the given name
                t = getattr(hf.nl, t)()
            if not isinstance(t, hf.nl.Nonlinearity):
                raise TypeError("Layer type (%s) must be an instance of "
                                "nonlinearities.Nonlinearity" % t)
            self.layers += [t]

        # initialize loss function
        self.init_loss(loss_type)

        # initialize connections
        if conns is None:
            # set up the feedforward series connections
            conns = {}
            for pre, post in zip(np.arange(self.n_layers - 1),
                                 np.arange(1, self.n_layers)):
                conns[pre] = [post]

        # add empty connection for last layer (just helps smooth the code
        # elsewhere)
        conns[self.n_layers - 1] = []

        self.conns = OrderedDict(sorted(conns.items(), key=lambda x: x[0]))
        # note: conns is an ordered dict sorted by layer so that we can
        # reliably loop over the items (in compute_offsets and init_weights)

        # maintain a list of backwards connections as well (for efficient
        # lookup in the other direction)
        self.back_conns = defaultdict(list)
        for pre in conns:
            for post in conns[pre]:
                self.back_conns[post] += [pre]

                if pre >= post:
                    raise ValueError("Can only connect from lower to higher "
                                     " layers (%s >= %s)" % (pre, post))

        # compute indices for the different connection weight matrices in the
        # overall parameter vector
        self.compute_offsets()

        # initialize connection weights
        if load_weights is None:
            self.W = self.init_weights(
                [(self.shape[pre], self.shape[post])
                 for pre in self.conns for post in self.conns[pre]],
                **W_init_params)
        else:
            if isinstance(load_weights, np.ndarray):
                self.W = load_weights
            else:
                # load weights from file
                self.W = np.load(load_weights)

            if len(self.W) != np.max(self.offsets.values()):
                raise IndexError("Length of loaded weights does not "
                                 "match expected length")

            if self.W.dtype != self.dtype:
                raise TypeError("Loaded weights (%s) don't match "
                                "self.dtype (%s)" % (self.W.dtype, self.dtype))

        # initialize GPU
        if use_GPU:
            try:
                import pycuda.autoinit
            except Exception, e:
                print e
                raise ImportError("PyCuda not installed, or no compatible "
                                  "device detected. Set use_GPU=False.")

            dev = pycuda.autoinit.device
            print "GPU found, using %s %s" % (dev.name(),
                                              dev.compute_capability())

    def run_batches(self, inputs, targets, optimizer,
                    max_epochs=1000, batch_size=None, test=None,
                    target_err=1e-6, plotting=False, test_err=None,
                    file_output=None):
        """Apply the given optimizer with a sequence of minibatches.

        :param inputs: input vectors (or a callable plant that will generate
            the input vectors dynamically)
        :param targets: target vectors
        :param optimizer: computes the weight update each epoch (see
            optimizers.py)
        :param max_epochs: the maximum number of epochs to run
        :param batch_size: the size of the minibatch to use in each epoch
        :param test: tuple of (inputs,targets) to use as the test data (if None
            then it will just use the same inputs and targets as training)
        :param target_err: run will terminate if this test error is reached
        :param plotting: if True then data from the run will be output to a
            file, which can be displayed via dataplotter.py
        :param test_err: a custom error function to be applied to
            the test data (e.g., classification error)
        :param file_output: output files from the run will use this as a prefix
            (if None then don't output files)
        """

        if self.debug:
            print_period = 1
#             np.seterr(all="raise")
        else:
            print_period = 10

        test_errs = []
        self.best_W = None
        self.best_error = None
        prefix = "HF" if file_output is None else file_output
        plots = defaultdict(list)
        self.optimizer = optimizer

        for i in range(max_epochs):
            self.epoch = i

            if i % print_period == 0:
                print "=" * 40
                print "batch", i

            # generate minibatch and cache activations
            self.cache_minibatch(inputs, targets, batch_size)

            # validity checks
            if self.inputs.shape[-1] != self.shape[0]:
                raise ValueError("Input dimension (%d) does not match number "
                                 "of input nodes (%d)" %
                                 (self.inputs.shape[-1], self.shape[0]))
            if self.targets.shape[-1] != self.shape[-1]:
                raise ValueError("Target dimension (%d) does not match number "
                                 "of output nodes (%d)" %
                                 (self.targets.shape[-1], self.shape[-1]))
            if not(self.inputs.dtype == self.targets.dtype == np.float32):
                raise TypeError("Input type must be np.float32")

            assert self.activations[-1].dtype == self.dtype

            # compute update
            update = optimizer.compute_update(i % print_period == 0)

            assert update.dtype == self.dtype

            self.W += update

            # invalidate cached activations (shouldn't be necessary,
            # but doesn't hurt)
            self.activations = None
            self.d_activations = None
            self.GPU_activations = None

            # compute test error
            if test is None:
                test_in, test_t = self.inputs, self.targets
            else:
                test_in, test_t = test[0], test[1]

            if test_err is None:
                err = self.error(self.W, test_in, test_t)
            else:
                output = self.forward(test_in, self.W)
                err = test_err.batch_loss(output, test_t)
            test_errs += [err]

            if i % print_period == 0:
                print "test error", test_errs[-1]

            # save the weights with the best error
            if self.best_W is None or test_errs[-1] < self.best_error:
                self.best_W = self.W.copy()
                self.best_error = test_errs[-1]

            # dump plot data
            if plotting:
                plots["update norm"] += [np.linalg.norm(update)]
                plots["W norm"] += [np.linalg.norm(self.W)]
                plots["test error"] += [test_errs[-1]]

                if hasattr(optimizer, "plots"):
                    plots.update(optimizer.plots)

                with open("%s_plots.pkl" % prefix, "wb") as f:
                    pickle.dump(plots, f)

            # dump weights
            if file_output is not None:
                np.save("%s_weights.npy" % prefix, self.W)

            # check for termination
            if test_errs[-1] < target_err:
                print "target error reached"
                break
            if test is not None and i > 20 and test_errs[-10] < test_errs[-1]:
                print "overfitting detected, terminating"
                break

    def cache_minibatch(self, inputs, targets, batch_size=None):
        """Pick a subset of inputs and targets to use in minibatch, and cache
        the activations for that minibatch."""

        batch_size = inputs.shape[0] if batch_size is None else batch_size

        if not callable(inputs):
            # inputs/targets are vectors, select a subset
            indices = np.random.choice(np.arange(len(inputs)),
                                       size=batch_size, replace=False)
            self.inputs = inputs[indices]
            self.targets = targets[indices]

            # cache activations
            self.activations, self.d_activations = self.forward(self.inputs,
                                                                self.W,
                                                                deriv=True)
        else:
            if targets is not None:
                raise ValueError("Cannot specify targets when using dynamic "
                                 "plant to generate inputs (plant should "
                                 "generate targets itself)")

            # run plant to generate batch
            inputs.shape[0] = batch_size
            self.activations, self.d_activations = self.forward(inputs, self.W,
                                                                deriv=True)
            self.inputs = np.asarray(inputs.get_inputs(), dtype=np.float32)
            self.targets = np.asarray(inputs.get_targets(), dtype=np.float32)

        if self.use_GPU:
            from pycuda import gpuarray
            self.GPU_activations = [
                gpuarray.to_gpu(np.asarray(a, dtype=np.float32))
                for a in self.activations]
            self.GPU_d_activations = [
                gpuarray.to_gpu(np.asarray(a, dtype=np.float32))
                for a in self.d_activations]
            self.GPU_W = gpuarray.to_gpu(np.asarray(self.W, dtype=np.float32))
            self.GPU_d2_loss = [
                gpuarray.to_gpu(np.asarray(a, dtype=np.float32))
                if a is not None else None
                for a in self.loss.d2_loss(self.activations, self.targets)]
            self.GPU_tmp_space = [gpuarray.empty(a.shape, np.float32)
                                  for a in self.activations]

    def forward(self, input, params, deriv=False):
        """Compute feedforward activations for given input and parameters.

        If deriv=True then also compute the derivative of the activations.
        """

        if callable(input):
            input.reset()

        activations = [None for _ in range(self.n_layers)]
        if deriv:
            d_activations = [None for _ in range(self.n_layers)]

        for i in range(self.n_layers):
            if i == 0:
                if callable(input):
                    inputs = input(None)
                else:
                    inputs = input
            else:
                inputs = np.zeros((input.shape[0], self.shape[i]),
                                  dtype=self.dtype)
                for pre in self.back_conns[i]:
                    W, b = self.get_weights(params, (pre, i))
                    inputs += np.dot(activations[pre], W)
                    inputs += b
                    # note: we're applying a bias on each connection (rather
                    # than one for each neuron). just because it's easier than
                    # tracking how many connections there are for each layer.
            activations[i] = self.layers[i].activation(inputs)

            if deriv:
                d_activations[i] = self.layers[i].d_activation(inputs,
                                                               activations[i])

        if deriv:
            return activations, d_activations

        return activations

    def error(self, W=None, inputs=None, targets=None):
        """Compute network error."""

        W = self.W if W is None else W
        inputs = self.inputs if inputs is None else inputs

        # get outputs
        if (W is self.W and inputs is self.inputs and
                self.activations is not None):
            # use cached activations
            activations = self.activations
        else:
            # compute activations
            activations = self.forward(inputs, W)

        # get targets
        if callable(inputs):
            # get targets from plant
            targets = inputs.get_targets()
        else:
            targets = self.targets if targets is None else targets

        # note: np.nan can be used in the target to specify places
        # where the target is not defined. those get translated to
        # zero error in the loss function.

        # we don't want to confuse nan outputs with nan targets
        assert np.all(np.isfinite(activations[-1]))

        error = self.loss.batch_loss(activations, targets)

        return error

    def J_dot(self, J, vec, transpose=False, out=None):
        """Compute the product of a Jacobian and some vector."""

        # In many cases the Jacobian is a diagonal matrix, so it is more
        # efficient to just represent it with the diagonal vector.  This
        # function just lets those two be used interchangeably.

        if J.ndim == 2:
            # note: the first dimension is the batch, so ndim==2 means
            # this is a diagonal representation

            if out is None:
                return J * vec
            else:
                out[...] = J * vec
                return out
        else:
            if transpose:
                J = np.transpose(J, (0, 2, 1))

            if out is None:
                # passing out=None fails for some reason
                return np.einsum("ijk,ik->ij", J, vec)
            else:
                return np.einsum("ijk,ik->ij", J, vec, out=out)

    def calc_grad(self):
        """Compute parameter gradient."""

        for l in self.layers:
            if l.stateful:
                raise TypeError("Cannot use neurons with internal state in "
                                "a one-step feedforward network; use "
                                "RNNet instead.")

        grad = np.zeros_like(self.W)

        # backpropagation
        # note: this uses the cached activations, so the forward
        # pass has already been run elsewhere

        # compute output error for each layer
        error = self.loss.d_loss(self.activations, self.targets)

        error = [np.zeros_like(self.activations[i]) if e is None else e
                 for i, e in enumerate(error)]

        deltas = [np.zeros_like(a) for a in self.activations]

        # backwards pass
        for i in range(self.n_layers - 1, -1, -1):
            for post in self.conns[i]:
                error[i] += np.dot(deltas[post],
                                   self.get_weights(self.W, (i, post))[0].T)

                W_grad, b_grad = self.get_weights(grad, (i, post))
                np.dot(self.activations[i].T, deltas[post], out=W_grad)
                np.sum(deltas[post], axis=0, out=b_grad)

            self.J_dot(self.d_activations[i], error[i],
                       out=deltas[i])

        grad /= self.inputs.shape[0]

        return grad

    def check_grad(self, calc_grad):
        """Check gradient via finite differences (for debugging)."""

        eps = 1e-6
        grad = np.zeros(calc_grad.shape)
        for i in range(len(self.W)):
            inc_W = np.zeros_like(self.W)
            inc_W[i] = eps

            error_inc = self.error(self.W + inc_W, self.inputs, self.targets)
            error_dec = self.error(self.W - inc_W, self.inputs, self.targets)
            grad[i] = (error_inc - error_dec) / (2 * eps)
        try:
            assert np.allclose(calc_grad, grad, rtol=1e-3)
        except AssertionError:
            print "calc_grad"
            print calc_grad
            print "finite grad"
            print grad
            print "calc_grad - finite grad"
            print calc_grad - grad
            print "calc_grad / finite grad"
            print calc_grad / grad
            raw_input("Paused (press enter to continue)")

    def calc_G(self, v, damping=0):
        """Compute Gauss-Newton matrix-vector product."""

        Gv = np.zeros(self.W.size, dtype=self.dtype)

        # allocate temporary space for intermediate values, to save on
        # memory allocations
        tmp_space = [np.zeros(a.shape, self.dtype) for a in self.activations]

        # R forward pass
        R_activations = [np.zeros_like(a) for a in self.activations]
        for i in range(1, self.n_layers):
            R_input = np.zeros_like(self.activations[i])
            for pre in self.back_conns[i]:
                vw, vb = self.get_weights(v, (pre, i))
                Ww, _ = self.get_weights(self.W, (pre, i))
                R_input += np.dot(self.activations[pre], vw, out=tmp_space[i])
                R_input += vb
                R_input += np.dot(R_activations[pre], Ww, out=tmp_space[i])

            self.J_dot(self.d_activations[i], R_input,
                       out=R_activations[i])

        # backward pass
        R_deltas = [np.zeros_like(a) for a in self.activations]

        R_error = self.loss.d2_loss(self.activations, self.targets)

        for i in range(self.n_layers - 1, -1, -1):
            if R_error[i] is not None:
                R_error[i] *= R_activations[i]
            else:
                R_error[i] = np.zeros_like(self.activations[i])

            for post in self.conns[i]:
                W, _ = self.get_weights(self.W, (i, post))

                R_error[i] += np.dot(R_deltas[post], W.T, out=tmp_space[i])

                W_g, b_g = self.get_weights(Gv, (i, post))
                np.dot(self.activations[i].T, R_deltas[post], out=W_g)
                np.sum(R_deltas[post], axis=0, out=b_g)

            self.J_dot(self.d_activations[i], R_error[i],
                       out=R_deltas[i], transpose=True)

        Gv /= len(self.inputs)

        Gv += damping * v  # Tikhonov damping

        return Gv

    def GPU_calc_G(self, v, damping=0):
        from pycuda import gpuarray

        Gv = gpuarray.zeros(self.W.size, dtype=np.float32)
        GPU_v = gpuarray.to_gpu(np.asarray(v, dtype=np.float32))

        # TODO: there are a bunch of memory optimizations here that we could
        # probably port to the standard calc_G

        # R forward pass

        # note: both of these just point to tmp_space, just keeping the
        # different variables here for consistency with calc_G
        R_input = self.GPU_tmp_space
        R_activations = R_input

        for i in range(self.n_layers):
            R_input[i].fill(0)
            for pre in self.back_conns[i]:
                vw, vb = self.get_weights(GPU_v, (pre, i))
                Ww, _ = self.get_weights(self.GPU_W, (pre, i))

                hf.gpu.m_dot(self.GPU_activations[pre], vw,
                             out=R_input[i], increment=True)
                hf.gpu.iadd(R_input[i], vb)
#                 for batch in R_input[i]:
#                     batch += vb
                hf.gpu.m_dot(R_activations[pre], Ww,
                             out=R_input[i], increment=True)

            hf.gpu.J_dot(self.GPU_d_activations[i], R_input[i],
                         out=R_activations[i])
#             R_activations[i] *= self.GPU_d_activations[i]

        # backward pass
        R_error = R_activations
        R_deltas = R_error

        for i in range(self.n_layers - 1, -1, -1):
            if self.GPU_d2_loss[i] is not None:
                # note: R_error[i] is already set to R_activations[i]
                R_error[i] *= self.GPU_d2_loss[i]
            else:
                R_error[i].fill(0)

            for post in self.conns[i]:
                W, _ = self.get_weights(self.GPU_W, (i, post))
                offset, W_end, b_end = self.offsets[(i, post)]

                hf.gpu.m_dot(R_deltas[post], W,  # self.GPU_W[offset:W_end],
                             transpose_b=True, out=R_error[i], increment=True)

                hf.gpu.m_dot(self.GPU_activations[i], R_deltas[post],
                             transpose_a=True, out=Gv[offset:W_end])

                hf.gpu.sum_axis(R_deltas[post], 0, out=Gv[W_end:b_end])

            hf.gpu.J_dot(self.GPU_d_activations[i], R_error[i],
                         out=R_deltas[i], transpose_a=True)
#             R_deltas[i] *= self.GPU_d_activations[i]

        Gv /= len(self.inputs)

        # Tikhonov damping
        GPU_v *= damping
        Gv += GPU_v

        return Gv.get(pagelocked=True)

    def check_J(self):
        """Compute the Jacobian of the network via finite differences."""

        eps = 1e-6
        N = self.W.size

        # compute the Jacobian
        J = [None for _ in self.layers]
        for i in range(N):
            inc_i = np.zeros(N)
            inc_i[i] = eps

            inc = self.forward(self.inputs, self.W + inc_i)
            dec = self.forward(self.inputs, self.W - inc_i)

            for l in range(self.n_layers):
                J_i = (inc[l] - dec[l]) / (2 * eps)
                if J[l] is None:
                    J[l] = J_i[..., None]
                else:
                    J[l] = np.concatenate((J[l], J_i[..., None]), axis=-1)

        return J

    def check_G(self, calc_G, v, damping=0):
        """Check Gv calculation via finite differences (for debugging)."""

        # compute Jacobian
        J = self.check_J()

        # second derivative of loss function
        L = self.loss.d2_loss(self.activations, self.targets)
        # TODO: check loss via finite differences

        G = np.sum([np.einsum("aji,aj,ajk->ik", J[l], L[l], J[l])
                    for l in range(self.n_layers) if L[l] is not None], axis=0)

        # divide by batch size
        G /= self.inputs.shape[0]

        Gv = np.dot(G, v)
        Gv += damping * v

        try:
            assert np.allclose(calc_G, Gv, rtol=1e-3)
        except AssertionError:
            print "calc_G"
            print calc_G
            print "finite G"
            print Gv
            print "calc_G - finite G"
            print calc_G - Gv
            print "calc_G / finite G"
            print calc_G / Gv
            raise
#             raw_input("Paused (press enter to continue)")

    def init_weights(self, shapes, coeff=1.0, biases=0.0, init_type="sparse"):
        """Weight initialization, given shapes of weight matrices.

        Note: coeff, biases, and init_type can be specified by the
        W_init_params parameter in __init__.  Each can be specified as a
        single value (for all matrices) or as a list giving a value for each
        matrix.

        :param shapes: list of (pre,post) shapes for each weight matrix
        :param coeff: scales the magnitude of the connection weights
        :param biases: bias values for the post of each matrix
        :param init_type: type of initialization to use (currently supports
            'sparse', 'uniform', 'gaussian')
        """

        # if given single parameters, expand for all matrices
        if isinstance(coeff, (int, float)):
            coeff = [coeff] * len(shapes)
        if isinstance(biases, (int, float)):
            biases = [biases] * len(shapes)
        if isinstance(init_type, str):
            init_type = [init_type] * len(shapes)

        W = [np.zeros((pre + 1, post), dtype=self.dtype)
             for pre, post in shapes]

        for i, s in enumerate(shapes):
            if init_type[i] == "sparse":
                # sparse initialization (from martens)
                num_conn = 15

                for j in range(s[1]):
                    # pick num_conn random pre neurons
                    indices = np.random.choice(np.arange(s[0]),
                                               size=min(num_conn, s[0]),
                                               replace=False)

                    # connect to post
                    W[i][indices, j] = np.random.randn(indices.size) * coeff[i]
            elif init_type[i] == "uniform":
                W[i][:-1] = np.random.uniform(-coeff[i] / np.sqrt(s[0]),
                                              coeff[i] / np.sqrt(s[0]),
                                              (s[0], s[1]))
            elif init_type[i] == "gaussian":
                W[i][:-1] = np.random.randn(s[0], s[1]) * coeff[i]
            else:
                raise ValueError("Unknown weight initialization (%s)"
                                 % init_type)

            # set biases
            W[i][-1, :] = biases[i]

        W = np.concatenate([w.flatten() for w in W])

        return W

    def compute_offsets(self):
        """Precompute offsets for layers in the overall parameter vector."""

        n_params = [(self.shape[pre] + 1) * self.shape[post]
                    for pre in self.conns
                    for post in self.conns[pre]]
        self.offsets = {}
        self.weights = {}
        offset = 0
        for pre in self.conns:
            for post in self.conns[pre]:
                n_params = (self.shape[pre] + 1) * self.shape[post]
                self.offsets[(pre, post)] = (
                    offset,
                    offset + n_params - self.shape[post],
                    offset + n_params)
                offset += n_params

        return offset

    def get_weights(self, params, conn):
        """Get weight matrix for a connection from overall parameter vector."""

        if conn not in self.offsets:
            return None

        if params is self.W and conn in self.weights:
            return self.weights[conn]

        offset, W_end, b_end = self.offsets[conn]
        W = params[offset:W_end]
        b = params[W_end:b_end]

        result = (W.reshape((self.shape[conn[0]], self.shape[conn[1]])),
                  b)

        # cache references to the reshaped weights
        if params is self.W:
            self.weights[conn] = result

        return result

    def init_loss(self, loss_type):
        if isinstance(loss_type, (list, tuple)):
            tmp = loss_type
        else:
            tmp = [loss_type]

        for t in tmp:
            if not isinstance(t, hf.loss_funcs.LossFunction):
                raise TypeError("loss_type (%s) must be an instance of "
                                "LossFunction" % t)

            # sanity checks
            if (isinstance(t, hf.loss_funcs.CrossEntropy) and
                np.any(self.layers[-1].activation(np.linspace(-80, 80,
                                                              100)[None, :])
                       <= 0)):
                # this won't catch everything, but hopefully a useful warning
                raise ValueError("Must use positive activation function "
                                 "with cross-entropy error")
            if (isinstance(t, hf.loss_funcs.CrossEntropy) and
                    not isinstance(self.layers[-1], hf.nl.Softmax)):
                print ("Softmax should probably be used with cross-entropy "
                       "error")

        if isinstance(loss_type, (list, tuple)):
            self.loss = hf.loss_funcs.LossSet(loss_type)
        else:
            self.loss = loss_type

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, o):
        self._optimizer = o
        o.net = self
