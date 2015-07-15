"""Implementation of Hessian-free optimization for feedforward networks.

Author: Daniel Rasmussen (drasmussen@princeton.edu)

Based on
Martens, J. (2010). Deep learning via Hessian-free optimization. In Proceedings
of the 27th International Conference on Machine Learning.

and Matlab code from James Martens available at
http://www.cs.toronto.edu/~jmartens/research.html
"""

import pickle
from collections import defaultdict, OrderedDict

from scipy.special import expit
import numpy as np

try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray
except:
    print "PyCuda not installed, or no compatible device detected"


class HessianFF(object):
    def __init__(self, layers=[1, 1, 1], use_GPU=False, load_weights=None,
                 debug=False, neuron_types="logistic", conns=None,
                 error_type="mse", W_init_coeff=1.0):
        self.use_GPU = use_GPU
        self.debug = debug
        self.layers = layers
        self.W_init_coeff = W_init_coeff
        self.n_layers = len(layers)
        self.dtype = np.float64 if self.debug else np.float32

        self.inputs = None
        self.targets = None

        # set up neural activation functions for each layer
        if isinstance(neuron_types, str):
            neuron_types = [neuron_types for _ in range(self.n_layers)]
            neuron_types[0] = "linear"

        if len(neuron_types) != len(layers):
            raise ValueError("Must specify a neuron type for each layer")

        self.neuron_types = neuron_types
        self.act = []
        self.deriv = []
        for t in neuron_types:
            if t == "logistic":
                self.act += [expit]
                self.deriv += [lambda a: a * (1 - a)]
            elif t == "tanh":
                self.act += [np.tanh]
                self.deriv += [lambda a: 1 - a ** 2]
            elif t == "linear":
                self.act += [lambda x: x]
                self.deriv += [np.ones_like]
            elif t == "relu":
                self.act += [lambda x: np.maximum(0, x)]
                self.deriv += [lambda x: x > 0]
            elif t == "softmax":
                def softmax(x):
                    e = np.exp(x - np.max(x, axis=-1)[..., None])
                    # note: shift everything down by max (doesn't change
                    # result, but can help avoid numerical errors)

                    e /= np.sum(e, axis=-1)[..., None]

                    e[e < 1e-10] = 1e-10
                    # clip to avoid numerical errors
                    return e
                self.act += [softmax]
                self.deriv += [lambda a: a[..., None] * (np.eye(a.shape[-1]) -
                                                         a[..., None, :])]
            elif isinstance(t, list):
                if callable[t[0]] and callable[t[1]]:
                    self.act += [t[0]]
                    self.deriv += [t[1]]
                else:
                    raise TypeError("Must specify a function for custom type")
            else:
                raise ValueError("Unknown neuron type (%s)" % t)

        # check error type
        if error_type not in ["mse", "ce"]:
            raise ValueError("Unknown error type (%s)" % error_type)
        if (error_type == "ce" and
            np.any(self.act[-1](np.linspace(-80, 80, 100)[None, :]) <= 0)):
            # this won't catch everything, but hopefully a useful warning
            raise ValueError("Must use positive activation function"
                             " with cross-entropy error")
        if ((error_type == "ce" and self.neuron_types[-1] != "softmax") or
            (error_type != "ce" and self.neuron_types[-1] == "softmax")):
            print "Softmax should probably be used with cross-entropy error"
        self.error_type = error_type

        # add connections
        if conns is None:
            # normal feedforward connections
            conns = {}
            for pre, post in zip(np.arange(self.n_layers - 1),
                                 np.arange(1, self.n_layers)):
                conns[pre] = [post]

        self.conns = OrderedDict(conns)
        back_conns = defaultdict(list)
        for pre in conns:
            # can only make downstream connections
            for post in conns[pre]:
                if pre >= post:
                    raise ValueError("Invalid connection (%s >= %s)"
                                     % (pre, post))
                back_conns[post] += [pre]
        self.back_conns = OrderedDict(back_conns)

        if load_weights is not None:
            if isinstance(load_weights, np.ndarray):
                self.W = load_weights
            else:
                # load weights from file
                self.W = np.load(load_weights)
            if self.W.dtype != self.dtype:
                raise TypeError("Weights from file don't match self.dtype (%s)"
                                % self.dtype)
        else:
            self.W = self.init_weights([(self.layers[pre] + 1,
                                         self.layers[post])
                                        for pre in self.conns
                                        for post in self.conns[pre]],
                                       W_init_coeff)

        self.compute_offsets()

        if use_GPU:
            self.init_GPU()
        else:
            self.outer_sum = lambda a, b: np.ravel(np.einsum("ij,ik", a, b))

    def init_GPU(self):
        dev = pycuda.autoinit.device
        print "GPU found, using %s %s" % (dev.name(), dev.compute_capability())
        self.num_threads = dev.MAX_THREADS_PER_BLOCK

        # this one operation in the gradient/Gv calculations is where
        # most of the computational work can be this algorithm, so
        # parallelizing it on the GPU can be pretty helpful.
        mod = SourceModule("""
        __global__ void outer_sum(float *a, float *b, float *out,
                                  int batch_size)
        {
            int a_i = blockIdx.x*blockDim.x + threadIdx.x;
            int b_i = blockIdx.y*blockDim.y + threadIdx.y;
            const int a_len = blockDim.x * gridDim.x;
            const int b_len = blockDim.y * gridDim.y;
            const int out_addr = a_i*b_len + b_i;

            out[out_addr] = 0;
            for(int j=0; j < batch_size; j++) {
                out[out_addr] += a[a_i] * b[b_i];
                a_i += a_len;
                b_i += b_len;
            }
        }
        """)

        def find_block_len(n_threads, threads_per, vec_len):
            # need to divide n_threads into blocks of size n*threads_per. we
            # want n to be as large as possible, so we use all the threads in
            # a block. but vec_len also needs to be evenly divisible by n (I
            # don't think it's possible to have different sized blocks in
            # different grid cells). so we want the highest factor of vec_len
            # that is <= n_threads/threads_per.
            start = int(n_threads / threads_per)

            if start >= vec_len:
                return vec_len

            mid = int(np.sqrt(vec_len))
            for n in range(start, 0 if start < mid else mid - 1, -1):
                if vec_len % n == 0:
                    return n

            return 1

        def outer_sum(in_a, in_b):
            if isinstance(in_a, (list, tuple)):
                # load pre-cached GPU activations
                a = self.GPU_activations
                for idx in in_a:
                    a = a[idx]
            else:
                a = in_a
            b = in_b  # b is never cached

            a_len = np.int32(a.shape[1])
            b_len = np.int32(b.shape[1])
            batchsize = np.int32(a.shape[0])  # assume == b.shape[0]

            if a_len * b_len < 2 ** 15:
                # just do it on the CPU
                if isinstance(in_a, (list, tuple)):
                    a = self.activations
                    for idx in in_a:
                        a = a[idx]
                return np.ravel(np.einsum("ij,ik", a, b))

            out = np.zeros(a_len * b_len, dtype=np.float32)

            if self.debug:
                # convert the 64 bit values to 32 bit
                a = a.astype(np.float32)
                b = b.astype(np.float32)

            assert a.dtype == b.dtype == np.float32

            cols_per_block = find_block_len(self.num_threads, 1, b_len)
            rows_per_block = find_block_len(self.num_threads, cols_per_block,
                                            a_len)

            # load data onto gpu (if it isn't there already)
            a_gpu = (a if isinstance(a, gpuarray.GPUArray) else drv.In(a))
            b_gpu = (b if isinstance(b, gpuarray.GPUArray) else drv.In(b))
            out_gpu = (out if isinstance(out, gpuarray.GPUArray) else
                       drv.Out(out))

            # execute function
            gpu_outer = mod.get_function("outer_sum")
            gpu_outer(a_gpu, b_gpu, out_gpu, batchsize,
                      grid=(a_len / rows_per_block, b_len / cols_per_block),
                      block=(rows_per_block, cols_per_block, 1))

            if self.debug:
                if isinstance(a, gpuarray.GPUArray):
                    tmp_a = np.zeros(a.shape, dtype=np.float32)
                    a.get(tmp_a)
                else:
                    tmp_a = a
                if isinstance(b, gpuarray.GPUArray):
                    tmp_b = np.zeros(b.shape, dtype=np.float32)
                    b.get(tmp_b)
                else:
                    tmp_b = b
                truth = np.ravel(np.einsum("ij,ik", tmp_a, tmp_b))
                try:
                    assert np.allclose(out, truth, atol=1e-4)
                except AssertionError:
                    print out
                    print truth
                    print out - truth
                    raise

            return out

        self.outer_sum = outer_sum

    def init_weights(self, shapes, coeff=1.0):
        """Weight initialization, given shapes of weight matrices (including
        bias row)."""

        # sparse initialization (from martens)
        num_conn = 15
        W = [np.zeros(s, dtype=self.dtype) for s in shapes]
        for i, s in enumerate(shapes):
            for j in range(s[1]):
                # pick num_conn random pre neurons (omitting "bias" neuron)
                indices = np.random.choice(np.arange(s[0] - 1),
                                           size=min(num_conn, s[0] - 1),
                                           replace=False)

                # connect to post
                W[i][indices, j] = np.random.randn(indices.size) * coeff
            if self.neuron_types[i] in ["tanh", "relu"]:
                # bias it away from zero
                W[i][-1, :] = 0.5
        W = np.concatenate([w.flatten() for w in W])

        # random initialization
#         n_params = [pre * post for pre, post in shapes]
#         W = np.zeros(np.sum(n_params), dtype=dtype)
#         for i, s in enumerate(shapes):
#             offset = np.sum(n_params[:i])
#             W[offset:offset + n_params[i]] = (
#                 np.random.uniform(-1 / np.sqrt(s[0]),
#                                   1 / np.sqrt(s[0]),
#                                   n_params[i]))

        return W

    def compute_offsets(self):
        """Precompute offsets for layers in the overall parameter vector."""

        n_params = [(self.layers[pre] + 1) * self.layers[post]
                    for pre in self.conns
                    for post in self.conns[pre]]
        self.offsets = {}
        offset = 0
        for pre in self.conns:
            for post in self.conns[pre]:
                n_params = (self.layers[pre] + 1) * self.layers[post]
                self.offsets[(pre, post)] = (
                    offset,
                    offset + n_params - self.layers[post],
                    offset + n_params)
                offset += n_params

    def get_weights(self, params, conn, separate=True):
        """Get weight matrix for a layer from the overall parameter vector."""

        offset, W_end, b_end = self.offsets[conn]
        if separate:
            W = params[offset:W_end]
            b = params[W_end:b_end]

            return (W.reshape((self.layers[conn[0]],
                               self.layers[conn[1]])),
                    b)
        else:
            return params[offset:b_end].reshape((self.layers[conn[0]] + 1,
                                                 self.layers[conn[1]]))

    def forward(self, input, params):
        """Compute feedforward activations for given input and parameters."""

        if input.ndim < 2:
            # then we've just been given a single sample (rather than batch)
            input = input[None, :]

        activations = [None for _ in range(self.n_layers)]
        activations[0] = self.act[0](input)
        for i in range(1, self.n_layers):
            inputs = np.zeros((input.shape[0], self.layers[i]),
                              dtype=self.dtype)
            for pre in self.back_conns[i]:
                W, b = self.get_weights(params, (pre, i))
                inputs += np.dot(activations[pre], W) + b
                # note: we're applying a bias on each connection (rather
                # than one for each neuron)
            activations[i] = self.act[i](inputs)

        return activations

    def error(self, W=None, inputs=None, targets=None):
        """Compute network error."""

        W = self.W if W is None else W
        inputs = self.inputs if inputs is None else inputs
        targets = self.targets if targets is None else targets

        if (self.activations is not None and
                W is self.W and inputs is self.inputs):
            # use cached activations
            outputs = self.activations[-1]
        else:
            # compute activations
            outputs = self.forward(inputs, W)[-1]

        if self.error_type == "mse":
            error = np.sum((outputs - np.nan_to_num(targets)) ** 2)
            error /= 2 * len(inputs)
        elif self.error_type == "ce":
            nans = np.isnan(targets)
            targets[nans] = outputs[nans]
            error = -np.sum(targets * np.log(outputs))
            error /= len(inputs)

        return error

    def J_dot(self, J, vec):
        """Compute the product of a Jacobian and some vector."""

        # In many cases the Jacobian is a diagonal matrix, so it is more
        # efficient to just represent it with the diagonal vector.  This
        # function just lets those two be used interchangeably.

        if J.ndim == 2:
            # note: the first dimension is the batch, so ndim==2 means
            # this is a diagonal representation
            return J * vec
        else:
            return np.einsum("ijk,ik->ij", J, vec)

    def calc_grad(self, W=None, inputs=None, targets=None):
        """Compute parameter gradient."""

        W = self.W if W is None else W
        inputs = self.inputs if inputs is None else inputs
        targets = self.targets if targets is None else targets

        if W is self.W and inputs is self.inputs:
            # use cached activations
            activations = self.activations
            GPU_activations = self.GPU_activations
            d_activations = self.d_activations
        else:
            # compute activations
            activations = self.forward(inputs, W)
            GPU_activations = None
            d_activations = [self.deriv[i](a)
                             for i, a in enumerate(activations)]

        grad = np.zeros(W.size, dtype=self.dtype)

        # backpropagate error
        deltas = [None for _ in range(self.n_layers)]

        if self.error_type == "mse":
            # translate any nans in target to zero error
            error = activations[-1] - np.nan_to_num(targets)
        elif self.error_type == "ce":
            error = -np.nan_to_num(targets) / activations[-1]

        deltas[-1] = self.J_dot(d_activations[-1], error)

        for i in range(self.n_layers - 2, -1, -1):
            error = np.zeros(activations[i].shape, dtype=self.dtype)
            for post in self.conns[i]:
                c_error = np.dot(deltas[post],
                                 self.get_weights(W, (i, post))[0].T)
                error += c_error

                offset, W_end, b_end = self.offsets[(i, post)]
                grad[offset:W_end] = self.outer_sum(activations[i]
                                                    if GPU_activations is None
                                                    else [i],
                                                    deltas[post])
                np.sum(deltas[post], axis=0, out=grad[W_end:b_end])

            deltas[i] = self.J_dot(d_activations[i], error)

        grad /= inputs.shape[0]

        return grad

    def check_grad(self, calc_grad, inputs, targets):
        """Check gradient via finite differences (for debugging)."""

        eps = 1e-4

        grad = np.zeros(calc_grad.shape)
        for i, val in enumerate(self.W):
            inc_W = np.copy(self.W)
            dec_W = np.copy(self.W)
            inc_W[i] = val + eps
            dec_W[i] = val - eps

            error_inc = self.error(inc_W, inputs, targets)
            error_dec = self.error(dec_W, inputs, targets)
            grad[i] = (error_inc - error_dec) / (2 * eps)

        try:
            assert np.allclose(grad,
                               calc_grad,
                               atol=1e-4)
        except AssertionError:
            print calc_grad
            print grad
            print calc_grad - grad
            print calc_grad / grad
            raise

    def gradient_descent(self, inputs, targets, l_rate=1):
        """Basic first-order gradient descent (for comparison)."""

        self.inputs = inputs
        self.targets = targets

        # calculate activations
        self.activations = self.forward(self.inputs, self.W)
        self.GPU_activations = None
        self.d_activations = [self.deriv[i](a)
                              for i, a in enumerate(self.activations)]

        # compute gradient
        grad = self.calc_grad()

        if self.debug:
            self.check_grad(grad, inputs, targets)

        # update weights
        self.W -= l_rate * grad

    def check_G(self, calc_G, inputs, targets, v, damping=0):
        """Check Gv calculation via finite differences (for debugging)."""

        eps = 1e-6
        N = self.W.size

        g = np.zeros(N)
        for n, input in enumerate(inputs):
            base = self.forward(input, self.W)[-1]

            J = np.zeros((base.size, N))
            for i in range(N):
                inc_i = np.zeros(N)
                inc_i[i] = eps

                J[:, i] = (self.forward(input, self.W + inc_i)[-1] -
                           base) / eps

            if self.error_type == "mse":
                L = np.eye(base.size)
            elif self.error_type == "ce":
#                 L = np.zeros(base.size)
#                 for j in range(base.size):
#                     inc_j = np.zeros(base.size)
#                     inc_j[j] = eps
#                     L[j] = ((-np.sum(targets[n] * np.log(base + inc_j)) +
#                              np.sum(targets[n] * np.log(base))) / eps -
#                             (-np.sum(targets[n] * np.log(base)) +
#                              np.sum(targets[n] * np.log(base - inc_j))) / eps) / eps
#                 assert np.allclose(L, targets[n] / base ** 2, atol=1e-3)
                L = np.diag((targets[n] / base ** 2).squeeze())

            g += np.dot(np.dot(J.T, np.dot(L, J)), v)

        g /= inputs.shape[0]

        g += damping * v

        try:
            assert np.allclose(g, calc_G, rtol=0.01)
        except AssertionError:
            print g
            print calc_G
            print calc_G - g
            print calc_G / g
            raise

    def G(self, v, damping=0, output=None):
        """Compute Gauss-Newton matrix-vector product."""

        if output is None:
            Gv = np.zeros(self.W.size, dtype=self.dtype)
        else:
            Gv = output

        # R forward pass
        R_activations = [None for _ in range(self.n_layers)]
        R_activations[0] = np.zeros_like(self.activations[0])
        for i in range(1, self.n_layers):
            R_input = np.zeros_like(self.activations[i])
            for pre in self.back_conns[i]:
                vw, vb = self.get_weights(v, (pre, i))
                Ww, _ = self.get_weights(self.W, (pre, i))
                R_input += np.dot(self.activations[pre], vw) + vb
                R_input += np.dot(R_activations[pre], Ww)

            R_activations[i] = self.J_dot(self.d_activations[i], R_input)

        # backward pass
        R_deltas = [None for _ in range(self.n_layers)]

        if self.error_type == "mse":
            # second derivative of error function is 1
            R_error = R_activations[-1]
        elif self.error_type == "ce":
            R_error = (R_activations[-1] *
                       np.nan_to_num(self.targets) / self.activations[-1] ** 2)

        R_deltas[-1] = self.J_dot(self.d_activations[-1],
                                  R_error)

        for i in range(self.n_layers - 2, -1, -1):
            R_error = np.zeros_like(self.activations[i])
            for post in self.conns[i]:
                W, _ = self.get_weights(self.W, (i, post))
                R_error += np.dot(R_deltas[post], W.T)

                offset, W_end, b_end = self.offsets[(i, post)]
                Gv[offset:W_end] = self.outer_sum(self.activations[i] if
                                                  self.GPU_activations is None
                                                  else [i],
                                                  R_deltas[post])
                np.sum(R_deltas[post], axis=0, out=Gv[W_end:b_end])

            R_deltas[i] = self.J_dot(self.d_activations[i], R_error)

        Gv /= len(self.inputs)

        Gv += damping * v  # Tikhonov damping

        return Gv

    def conjugate_gradient(self, init_delta, grad, iters=250):
        """Compute weight update using conjugate gradient algorithm."""

        store_iter = 5
        store_mult = 1.3
        deltas = []
        G_dir = np.zeros(self.W.size, dtype=self.dtype)
        vals = np.zeros(iters, dtype=self.dtype)

        base_grad = -grad
        delta = init_delta
        residual = base_grad - self.G(init_delta, damping=self.damping)
        res_norm = np.dot(residual, residual)
        direction = residual.copy()

        for i in range(iters):
            if self.debug:
                print "-" * 20
                print "CG iteration", i
                print "delta norm", np.linalg.norm(delta)
                print "direction norm", np.linalg.norm(direction)

            self.G(direction, damping=self.damping, output=G_dir)

            # calculate step size
            step = res_norm / np.dot(direction, G_dir)

            if self.debug:
                print "step", step

                self.check_G(G_dir, self.inputs, self.targets,
                             direction, self.damping)

                assert np.isfinite(step)
                assert step >= 0
                assert (np.linalg.norm(np.dot(direction, G_dir)) >=
                        np.linalg.norm(np.dot(direction,
                                              self.G(direction, damping=0))))

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

            if self.debug:
                print "termination val", vals[i]

            if (i > gap and vals[i - gap] < 0 and
                    (vals[i] - vals[i - gap]) / vals[i] < 5e-6 * gap):
                break

        deltas += [(i, np.copy(delta))]

        return deltas

    def run_batches(self, inputs, targets, CG_iter=250, init_damping=1.0,
                    max_epochs=1000, batch_size=None, test=None,
                    target_err=1e-6, plotting=False, classification=False,
                    file_output=None):
        """Apply Hessian-free algorithm with a sequence of minibatches."""

        if self.debug:
            print_period = 1
            np.seterr(all="raise")
        else:
            print_period = 10

        init_delta = np.zeros(self.W.size, dtype=self.dtype)
        self.damping = init_damping
        test_errs = []

        if plotting:
            # data is dumped out to file so that the plots can be
            # displayed/updated in parallel (see dataplotter.py)
            plots = {}
            plot_vars = ["new_err", "l_rate", "np.linalg.norm(delta)",
                         "self.damping", "np.linalg.norm(self.W)",
                         "deltas[-1][0]", "test_errs[-1]"]
            if classification:
                plot_vars += ["class_err"]
            for v in plot_vars:
                plots[v] = []

            with open("%s_plots.pkl" % (file_output
                                        if file_output is not None
                                        else "HF"),
                      "wb") as f:
                pickle.dump(plots, f)

        for i in range(max_epochs):
            if i % print_period == 0:
                print "=" * 40
                print "batch", i

            # generate mini-batch
            if batch_size is None:
                self.inputs = inputs
                self.targets = targets
            else:
                indices = np.random.choice(np.arange(len(inputs)),
                                           size=batch_size, replace=False)
                self.inputs = inputs[indices]
                self.targets = targets[indices]
            if not(self.inputs.dtype == self.targets.dtype == np.float32):
                raise TypeError("Input type must be np.float32")

            # cache activations
            self.activations = self.forward(self.inputs, self.W)
            self.d_activations = [self.deriv[j](a)
                                  for j, a in enumerate(self.activations)]

            self.GPU_activations = None
            if self.use_GPU:
                self.GPU_activations = [gpuarray.to_gpu(a)
                                        for a in self.activations]

            assert self.activations[-1].dtype == self.dtype

            # compute gradient
            grad = self.calc_grad()

            if i % print_period == 0:
                print "grad norm", np.linalg.norm(grad)

            # run CG
            deltas = self.conjugate_gradient(init_delta * 0.95, grad,
                                             iters=CG_iter)

            if i % print_period == 0:
                print "CG steps", deltas[-1][0]

            err = self.error()  # note: don't reuse previous error, diff batch
            init_delta = deltas[-1][1]  # note: don't backtrack this

            # CG backtracking
            new_err = np.inf
            for j in range(len(deltas) - 1, -1, -1):
                prev_err = self.error(self.W + deltas[j][1])
                if prev_err > new_err:
                    break
                delta = deltas[j][1]
                new_err = prev_err
            else:
                j -= 1

            if i % print_period == 0:
                print "using iteration", deltas[j + 1][0]
                print "err", err
                print "new_err", new_err

            # update damping parameter (compare improvement predicted by
            # quadratic model to the actual improvement in the error)
            denom = (0.5 * np.dot(delta, self.G(delta, damping=0)) +
                     np.dot(grad, delta))

            improvement_ratio = (new_err - err) / denom
            if improvement_ratio < 0.25:
                self.damping *= 1.5
            elif improvement_ratio > 0.75:
                self.damping *= 0.66

            if i % print_period == 0:
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
                new_err = self.error(self.W + l_rate * delta)
            else:
                # no good update, so skip this iteration
                l_rate = 0.0
                new_err = err

            if i % print_period == 0:
                print "min_improv", min_improv
                print "l_rate", l_rate
                print "l_rate_err", new_err
                print "improvement", new_err - err

            # update weights
            self.W += l_rate * delta

            # invalidate cached activations (shouldn't be necessary,
            # but doesn't hurt)
            self.activations = None
            self.d_activations = None
            if self.use_GPU:
                self.GPU_activations = None

            # compute test error
            if test is not None:
                test_errs += [self.error(self.W, test[0], test[1])]
            else:
                test_errs += [new_err]

            if i % print_period == 0:
                print "test error", test_errs[-1]

                if classification and test is not None:
                    output = self.forward(test[0], self.W)[-1]
                    class_err = (np.sum(np.argmax(output, axis=1) !=
                                        np.argmax(test[1], axis=1))
                                 / float(len(test[0])))
                    print "classification error", class_err

            # dump plot data
            if plotting:
                for v in plot_vars:
                    plots[v] += [eval(v)]

                with open("%s_plots.pkl" % (file_output
                                            if file_output is not None
                                            else "HF"),
                          "wb") as f:
                    pickle.dump(plots, f)

            # dump weights
            if i % print_period == 0 and file_output is not None:
                np.save("%s_weights.npy" % file_output, self.W)

            # check for termination
            if test_errs[-1] < target_err:
                print "target error reached"
                break
            if i > 20 and test_errs[-10] < test_errs[-1]:
                print "overfitting detected, terminating"
                break
