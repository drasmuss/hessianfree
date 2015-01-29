"""Implementation of Hessian-free optimization.

Author: Daniel Rasmussen (drasmussen@princeton.edu)

Based on
Martens, J. (2010). Deep learning via Hessian-free optimization. In Proceedings
of the 27th International Conference on Machine Learning.

and Matlab code from James Martens available at
http://www.cs.toronto.edu/~jmartens/research.html
"""

import pickle

from scipy.special import expit

import numpy as np

try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray
except:
    print "PyCuda not installed, or no compatible device detected"


class HessianBackprop(object):
    def __init__(self, layers=[1, 1, 1], use_GPU=False, debug=False):
        self.use_GPU = use_GPU
        self.debug = debug
        self.n_layers = len(layers)
        self.layers = layers
        self.n_params = [0 for _ in range(self.n_layers - 1)]
        for i in range(self.n_layers - 1):
            self.n_params[i] += (self.layers[i] + 1) * self.layers[i + 1]
        self.inputs = None
        self.targets = None

        self.init_weights()

        if use_GPU:
            self.init_GPU()
        else:
            def outer_sum(a, b, out=None):
                if out is None:
                    return np.ravel(np.einsum("ij,ik", a, b))
                else:
                    out[:] = np.ravel(np.einsum("ij,ik", a, b))
                    return out
            self.outer_sum = outer_sum

    def init_GPU(self):
        dev = pycuda.autoinit.device
        print "GPU found, using %s %s" % (dev.name(), dev.compute_capability())
        self.num_threads = (np.int32(1024)
                            if dev.compute_capability()[0] >= 2 else
                            np.int32(512))

        # this one operation in the gradient/Gv calculations is where
        # most of the computational work can be this algorithm, so
        # parallelizing it on the GPU can be pretty helpful.
        mod = SourceModule("""
        __global__ void outer_sum(float *a, float *b, float *out,
                                  int batch_size)
        {
            int a_i = blockIdx.x*blockDim.x + threadIdx.x;
            int b_i = threadIdx.y;
            const int a_len = blockDim.x * gridDim.x;
            const int b_len = blockDim.y;
            const int out_addr = a_i*b_len + b_i;

            out[out_addr] = 0;
            for(int j=0; j < batch_size; j++) {
                out[out_addr] += a[a_i] * b[b_i];
                a_i += a_len;
                b_i += b_len;
            }
        }
        """)

        def outer_sum(a, b, out=None):
            a_len = np.int32(a.shape[1])
            b_len = np.int32(b.shape[1])
            batchsize = np.int32(a.shape[0])  # assume == b.shape[0]

            if out is None:
                out = np.zeros(a_len * b_len, dtype=np.float32)

            assert a.dtype == b.dtype == out.dtype == np.float32
            assert b_len < self.num_threads
            # note: could make this work with longer b's if needed,
            # would just need to tile horizontally in the same way as
            # is done vertically

            rows_per_block = np.minimum(self.num_threads / b_len,
                                        a_len)
            while a_len % rows_per_block != 0:
                rows_per_block -= 1

            gpu_outer = mod.get_function("outer_sum")
            a_gpu = (a if isinstance(a, gpuarray.GPUArray) else drv.In(a))
            b_gpu = (b if isinstance(b, gpuarray.GPUArray) else drv.In(b))
            out_gpu = (out if isinstance(out, gpuarray.GPUArray) else
                       drv.Out(out))
            gpu_outer(a_gpu, b_gpu, out_gpu, batchsize,
                      grid=(a_len / rows_per_block, 1),
                      block=(rows_per_block, b_len, 1))

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
                    assert np.allclose(out, truth, atol=1e-6)
                except AssertionError:
                    print out
                    print truth
                    print out - truth
                    raise

            return out

        self.outer_sum = outer_sum

    def init_weights(self, dtype=np.float32):
        """Weight initialization"""

        if self.debug and dtype != np.float64:
            print "Changing weights to 64bit precision for debugging"
            dtype = np.float64

        # sparse initialization (from martens)
        num_conn = 15
        W = [np.zeros((self.layers[i] + 1, self.layers[i + 1]),
                      dtype=dtype)
             for i in range(self.n_layers - 1)]
        for l in range(self.n_layers - 1):
            for j in range(self.layers[l + 1]):
                # pick num_conn random pre neurons
                indices = np.ceil(np.random.rand(num_conn) * self.layers[l])
                indices = indices.astype(int)

                # connect to post
                W[l][indices, j] = np.random.randn(num_conn)
        self.W = np.concatenate([w.flatten() for w in W])

        # random initialization
#         self.W = np.zeros(np.sum(self.n_params), dtype=dtype)
#         for i in range(self.n_layers - 1):
#             offset = np.sum(self.n_params[:i])
#             self.W[offset:offset + self.n_params[i]] = (
#                 np.random.uniform(-1 / np.sqrt(self.layers[i]),
#                                   1 / np.sqrt(self.layers[i]),
#                                   self.n_params[i]))

    def get_offsets(self, layer):
        """Compute offsets for given layer in the overall parameter vector."""

        offset = np.sum(self.n_params[:layer])
        return (offset,
                offset + self.layers[layer] * self.layers[layer + 1],
                offset + self.n_params[layer])

    def get_layer(self, params, layer, separate=True):
        """Get weight matrix for a layer from the overall parameter vector."""

        offset, W_end, b_end = self.get_offsets(layer)
        if separate:
            W = params[offset:W_end]
            b = params[W_end:b_end]

            return (W.reshape((self.layers[layer],
                               self.layers[layer + 1])),
                    b)
        else:
            return params[offset:b_end].reshape((self.layers[layer] + 1,
                                                 self.layers[layer + 1]))

    def forward(self, input, params):
        """Compute feedforward activations for given input and parameters."""

        activations = [None for _ in range(self.n_layers)]
        activations[0] = input
        for i in range(self.n_layers - 1):
            W, b = self.get_layer(params, i)
            activations[i + 1] = expit(np.dot(activations[i], W) + b)

        return activations

    def error(self, W=None, inputs=None, targets=None):
        """Compute RMS error."""

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

        error = np.sum((outputs - targets) ** 2)
        error /= 2 * len(inputs)

        return error

    def calc_grad(self, W=None, inputs=None, targets=None):
        """Compute parameter gradient."""

        W = self.W if W is None else W
        inputs = self.inputs if inputs is None else inputs
        targets = self.targets if targets is None else targets

        if W is self.W and inputs is self.inputs:
            # use cached activations
            activations = (self.activations
                           if not self.use_GPU else
                           self.GPU_activations)
            d_activations = self.d_activations
            error = self.activations[-1] - targets  # can't use GPU activations
        else:
            # compute activations
            activations = self.forward(inputs, W)
            d_activations = [a * (1 - a) for a in activations]
            error = activations[-1] - targets

        grad = np.zeros(W.size, dtype=np.float32)

        # backpropagate error
        for i in range(self.n_layers - 1, 0, -1):
            delta = d_activations[i] * error
            error = np.dot(delta, self.get_layer(W, i - 1)[0].T)

            offset, W_end, b_end = self.get_offsets(i - 1)
            self.outer_sum(activations[i - 1], delta, out=grad[offset:W_end])
            np.sum(delta, axis=0, out=grad[W_end:b_end])

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

        # calculate gradients
        self.activations = self.forward(self.inputs, self.W)
        self.d_activations = [a * (1 - a) for a in self.activations]

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
        for input in inputs:
            base = self.forward(input, self.W)[-1]

            J = np.zeros((base.size, N))
            for i in range(N):
                inc_i = np.zeros(N)
                inc_i[i] = eps

                J[:, i] = (self.forward(input, self.W + inc_i)[-1] - base) / eps

            L = np.eye(base.size)  # true when using rms

            g += np.dot(np.dot(J.T, np.dot(L, J)), v)

        g /= inputs.shape[0]

        g += damping * v

        try:
            assert np.allclose(g, calc_G, rtol=0.01)
        except AssertionError:
            print g
            print calc_G
            raise

    def G(self, v, damping=0, output=None):
        """Compute Gauss-Newton matrix-vector product."""

        if output is None:
            Gv = np.zeros(self.W.size, dtype=np.float32)
        else:
            Gv = output

        Ws = [self.get_layer(self.W, i)[0] for i in range(self.n_layers - 1)]

        # R forward pass
        R_activation = np.zeros(self.inputs.shape, dtype=np.float32)
        for i in range(self.n_layers - 1):
            vw, vb = self.get_layer(v, i)
            R_input = np.dot(self.activations[i], vw) + vb
            R_input += np.dot(R_activation, Ws[i])
            R_activation = R_input * self.d_activations[i + 1]

        # backward pass
        R_error = R_activation  # second derivative of error function is 1

        for i in range(self.n_layers - 1, 0, -1):
            R_delta = self.d_activations[i] * R_error

            R_error = np.dot(R_delta, Ws[i - 1].T)

            offset, W_end, b_end = self.get_offsets(i - 1)

            if self.use_GPU:
                self.outer_sum(self.GPU_activations[i - 1], R_delta,
                               out=Gv[offset:W_end])
            else:
                self.outer_sum(self.activations[i - 1], R_delta,
                               out=Gv[offset:W_end])
            np.sum(R_delta, axis=0, out=Gv[W_end:b_end])

        Gv /= len(self.inputs)

        Gv += damping * v  # Tikhonov damping

        return Gv

    def conjugate_gradient(self, init_delta, grad, iters=250):
        """Compute weight update using conjugate gradient algorithm."""

        store_iter = 5
        store_mult = 1.3
        deltas = []
        G_dir = np.zeros(self.W.size, dtype=np.float32)
        vals = np.zeros(iters, dtype=np.float32)

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

            direction = np.nan_to_num(direction)  # sometimes this underflows

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
                    load_weights=None, plotting=False):
        """Apply Hessian-free algorithm with a sequence of minibatches."""

        if self.debug:
            print_period = 1
            np.seterr(all="raise")
        else:
            print_period = 10

        if load_weights is not None:
            # load weights from file
            with open(load_weights, "rb") as f:
                self.W = pickle.load(f)
            assert np.all([w.dtype == np.float32 for w in self.W])

        init_delta = np.zeros(self.W.size, dtype=np.float32)
        self.damping = init_damping
        test_errs = []

        if plotting:
            # data is dumped out to file so that the plots can be
            # displayed/updated in parallel (see dataplotter.py)
            plots = {}
            plot_vars = ["new_err", "l_rate", "np.linalg.norm(delta)",
                         "self.damping", "np.linalg.norm(self.W)",
                         "deltas[-1][0]", "test_errs[-1]"]
            for v in plot_vars:
                plots[v] = []

            with open("HF_plots.pkl", "wb") as f:
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
                indices = np.random.randint(len(inputs), size=batch_size)
                self.inputs = inputs[indices]
                self.targets = targets[indices]
            assert self.inputs.dtype == self.targets.dtype == np.float32

            # cache activations
            self.activations = self.forward(self.inputs, self.W)
            self.d_activations = [a * (1 - a) for a in self.activations]

            if self.use_GPU:
                self.GPU_activations = [gpuarray.to_gpu(a)
                                        for a in self.activations]

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
                if test is not None:
                    output = self.forward(test[0], self.W)[-1]
                    class_err = (np.sum(np.argmax(output, axis=1) !=
                                        np.argmax(test[1], axis=1))
                                 / float(len(test[0])))
                    print "classification error", class_err

            # dump plot data
            if plotting:
                for v in plot_vars:
                    plots[v] += [eval(v)]

                with open("HF_plots.pkl", "wb") as f:
                    pickle.dump(plots, f)

            # dump weights
            if i % print_period == 0:
                with open("HF_weights.pkl", "wb") as f:
                    pickle.dump(self.W, f)

            # check for termination
            if test_errs[-1] < 1e-6:
                print "minimum error reached"
                break
            if i > 20 and test_errs[-20] < test_errs[-1]:
                print "overfitting detected, terminating"
                break
