import ast
import pstats
import sys
import time
from cProfile import Profile

import numpy as np
import pycuda
import pycuda.autoinit
from pycuda import gpuarray

import hessianfree as hf


def threshold_calc_G():
    """Compare GPU vs CPU performance (can use this to determine at what point
    it is useful to run things on the GPU)."""

    batch_size = range(256, 1025, 256)
    layer_size = [1] + range(64, 513, 64)
    reps = 100

    times = np.zeros((len(batch_size), len(layer_size), 2))
    for i, b in enumerate(batch_size):
        inputs = np.random.randn(b, 1).astype(np.float32)
        targets = np.random.randn(b, 1).astype(np.float32)

        for j, n in enumerate(layer_size):
            ff = hf.FFNet([1, n, n, 1], use_GPU=False)
            ff.cache_minibatch(inputs, targets)

            v = np.random.randn(ff.W.size).astype(np.float32)

            for _ in range(5):
                ff.calc_G(v)

            start = time.time()
            for _ in range(reps):
                ff.calc_G(v)
            times[i, j, 0] = time.time() - start

            ff = hf.FFNet([1, n, n, 1], use_GPU=True)
            ff.cache_minibatch(inputs, targets)

            v = gpuarray.to_gpu(v)

            for _ in range(5):
                ff.GPU_calc_G(v)

            start = time.time()
            for _ in range(reps):
                ff.GPU_calc_G(v)

            v = v.get()
            times[i, j, 1] = time.time() - start

            print "b", b, "n", n, "times", times[i, j]

    print times[..., 1] - times[..., 0]
    print times[..., 1] < times[..., 0]


def threshold_rnn_calc_G():
    """Compare partial/full GPU vs CPU performance (can use this to determine
    at what point it is useful to run things on the GPU)."""

    batch_size = 1024
    layer_size = [1] + range(32, 129, 32)
    sig_len = [1] + range(8, 33, 8)
    reps = 100

    times = np.zeros((len(sig_len), len(layer_size), 2))
    for i, b in enumerate(sig_len):
        inputs = np.random.randn(batch_size, b, 1).astype(np.float32)
        targets = np.random.randn(batch_size, b, 1).astype(np.float32)

        for j, n in enumerate(layer_size):
            rnn = hf.RNNet([1, n, 1], use_GPU=False)
            rnn.cache_minibatch(inputs, targets)

            v = np.random.randn(rnn.W.size).astype(np.float32)

            for _ in range(5):
                rnn.calc_G(v)

            start = time.time()
            for _ in range(reps):
                rnn.calc_G(v)
            times[i, j, 0] = time.time() - start

            rnn = hf.RNNet([1, n, 1], use_GPU=True)
            rnn.cache_minibatch(inputs, targets)

            v = gpuarray.to_gpu(v)

            for _ in range(5):
                rnn.GPU_calc_G(v)

            start = time.time()
            for _ in range(reps):
                rnn.GPU_calc_G(v)

            v = v.get()
            times[i, j, 1] = time.time() - start

            print "b", b, "n", n, "times", times[i, j]

    print times[..., 1] - times[..., 0]
    print times[..., 1] < times[..., 0]


def profile_calc_G(cprofile=True):
    inputs = np.random.randn(1024, 1).astype(np.float32)
    targets = np.random.randn(1024, 1).astype(np.float32)
    N = 1024

    ff = hf.FFNet([1, N, N, 1], use_GPU=True)
    ff.cache_minibatch(inputs, targets)

    v = np.random.randn(ff.W.size).astype(np.float32)

    for _ in range(5):
        # run it a few times to get rid of any startup overhead
        ff.GPU_calc_G(v)

    if cprofile:
        start = time.time()

        p = Profile()
        p.enable()
    else:
        pycuda.driver.start_profiler()

    for _ in range(500):
        _ = ff.GPU_calc_G(v)

    if cprofile:
        p.disable()

        print "time", time.time() - start

        ps = pstats.Stats(p)
        ps.strip_dirs().sort_stats('time').print_stats(20)
    else:
        pycuda.driver.stop_profiler()


def profile_rnn_calc_G(cprofile=True):
    inputs = np.random.randn(1024, 128, 1).astype(np.float32)
    targets = np.random.randn(1024, 128, 1).astype(np.float32)
    N = 128

    rnn = hf.RNNet([1, N, 1], use_GPU=True)
    rnn.optimizer = hf.opt.HessianFree()  # for struc_damping check
    rnn.cache_minibatch(inputs, targets)

    v = np.random.randn(rnn.W.size).astype(np.float32)

    for _ in range(2):
        # run it a few times to get rid of any startup overhead
        rnn.GPU_calc_G(v)

    if cprofile:
        start = time.time()

        p = Profile()
        p.enable()
    else:
        pycuda.driver.start_profiler()

    for _ in range(100):
        _ = rnn.GPU_calc_G(v)

    if cprofile:
        p.disable()

        print "time", time.time() - start

        ps = pstats.Stats(p)
        ps.strip_dirs().sort_stats('time').print_stats(20)
    else:
        pycuda.driver.stop_profiler()


def profile_dot(cprofile=True):
    N = 1024
    a = np.random.randn(N, N).astype(np.float32)
    b = np.random.randn(N, N).astype(np.float32)
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = gpuarray.zeros((N, N), np.float32)

    for _ in range(2):
        # run it a few times to get rid of any startup overhead
        hf.gpu.dot(a_gpu, b_gpu, out=c_gpu)

    if cprofile:
        start = time.time()

        p = Profile()
        p.enable()
    else:
        pycuda.autoinit.context.synchronize()
        pycuda.driver.start_profiler()

    for _ in range(100):
        hf.gpu.dot(a_gpu, b_gpu, out=c_gpu, transpose_a=True,
                   transpose_b=True)
    c_gpu.get()

    if cprofile:
        p.disable()

        print "time", time.time() - start

        ps = pstats.Stats(p)
        ps.strip_dirs().sort_stats('time').print_stats(20)
    else:
        pycuda.driver.stop_profiler()


def profile_plant(cprofile=True):
    np.random.seed(0)
    n_inputs = 10
    sig_len = 20

    class Plant(hf.nl.Nonlinearity):
        def __init__(self, A, B, targets, init_state):
            super(Plant, self).__init__(stateful=True)

            self.A = np.asarray(A)
            self.B = B

            self.targets = targets
            self.init_state = init_state

            self.shape = [n_inputs, sig_len, len(A)]

            # derivative of output with respect to state (constant, so just
            # compute it once here)
            self.d_output = np.resize(np.eye(self.shape[-1]),
                                      (n_inputs, self.shape[-1],
                                       self.shape[-1], 1))

            self.reset()

        def activation(self, x):
            self.act_count += 1

            self.B_matrix, self.d_B_matrix = self.B(self.state)

            self.state = (np.dot(self.state, self.A) +
                          np.einsum("ij,ijk->ik", x, self.B_matrix))

            return self.state[:x.shape[0]]
            # note: generally x will be the same shape as state, this just
            # handles the case where we're passed a single item instead
            # of batch)

        def d_activation(self, x, _):
            self.d_act_count += 1
            assert self.act_count == self.d_act_count

            # derivative of state with respect to input
            d_input = self.B_matrix.transpose((0, 2, 1))[..., None]

            # derivative of state with respect to previous state
            d_state = np.resize(self.A.T, np.concatenate(([x.shape[0]],
                                                          self.A.shape)))
            d_state[:, 1, 0] += x[:, 1] * self.d_B_matrix[:, 1, 1]
            d_state = d_state[..., None]

            return np.concatenate((d_input, d_state, self.d_output), axis=-1)

        def __call__(self, _):
            self.inputs = np.concatenate((self.inputs, self.state[:, None, :]),
                                         axis=1)
            return self.state

        def get_inputs(self):
            return self.inputs

        def get_targets(self):
            return self.targets

        def reset(self, init=None):
            self.act_count = 0
            self.d_act_count = 0
            self.state = (self.init_state.copy() if init is None else
                          init.copy())
            self.inputs = np.zeros((self.shape[0], 0, self.shape[-1]),
                                   dtype=np.float32)
            self.B_matrix = self.d_B_matrix = None

    targets = np.ones((n_inputs, sig_len, 2), dtype=np.float32)
    targets[:, :, 1] = 0
    targets[:, :-1, :] = np.nan

    A = [[1, 0],
         [0.2, 1]]

    def B(state):
        B = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
        B[:, 1, 1] = np.cos(state[:, 0])

        d_B = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
        d_B[:, 1, 1] = -np.sin(state[:, 0])

        return B, d_B

    init1 = np.random.uniform(-1, 1, size=(n_inputs, 2))

    plant = Plant(A, B, targets, init1)

    rnn = hf.RNNet(shape=[2, 128, 2], debug=False,
                   layers=[hf.nl.Linear(), hf.nl.Tanh(), plant],
                   W_init_params={"coeff": 0.01}, W_rec_params={"coeff": 0.01},
                   use_GPU=True)

    rnn.optimizer = hf.opt.HessianFree()
    rnn.cache_minibatch(plant, None, batch_size=None)

    v = np.random.randn(rnn.W.size).astype(np.float32)
    v_gpu = gpuarray.to_gpu(v)

    for _ in range(5):
        rnn.GPU_calc_G(v_gpu)

    if cprofile:
        start = time.time()

        p = Profile()
        p.enable()
    else:
        pycuda.autoinit.context.synchronize()
        pycuda.driver.start_profiler()

#     for _ in range(20):
#         out = rnn.GPU_calc_G(v_gpu)
#     out.get()
    rnn.run_batches(plant, None, optimizer=hf.opt.HessianFree(CG_iter=30),
                    max_epochs=10, plotting=False, print_period=None)

    if cprofile:
        p.disable()

        print "time", time.time() - start

        ps = pstats.Stats(p)
        ps.strip_dirs().sort_stats('time').print_stats(20)
    else:
        pycuda.driver.stop_profiler()

if __name__ == "__main__":
    if sys.argv[1] in locals():
        locals()[sys.argv[1]](*[ast.literal_eval(a) for a in sys.argv[2:]])
    else:
        print "Unknown profile function (%s)" % sys.argv[1]
