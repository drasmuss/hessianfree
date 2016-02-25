from __future__ import print_function

import os
import warnings

from hessianfree.gpu import kernel_wrappers
from hessianfree.gpu.kernel_wrappers import iadd, sum_cols, multiply, J_dot
from hessianfree.gpu.kernel_wrappers import cublas_dot as dot


class DummyKernel:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Run gpu.init_kernels before calling kernel")

    def __getitem__(self, _):
        raise RuntimeError("Run gpu.init_kernels before calling kernel")

sum_cols_kernel = iadd_kernel = multiply_kernel = m_dot_kernel = \
    mv_batched_kernel = DummyKernel()

initialized = False


def init_kernels():
    global sum_cols_kernel, iadd_kernel, multiply_kernel, m_dot_kernel, \
        mv_batched_kernel, initialized

    if initialized:
        warnings.warn("Kernels already initialized, skipping init")
        return

    from pycuda import autoinit, driver, compiler
    from skcuda import misc

    dev = autoinit.device
    print("GPU found, using %s %s" % (dev.name(), dev.compute_capability()))

    misc.init()

    DTYPES = ["double", "float"]

    def parse_kernels():
        with open(os.path.join(os.path.dirname(__file__), "kernels.cu")) as f:
            code = f.read()

        code = code.replace("%tile_len%", "32")

        funcs = code.split("__global__ void")
        new_funcs = []
        for f in funcs:
            if "%float_type%" in f:
                for t in DTYPES:
                    new_funcs += [f.replace("%float_type%", t)]
            else:
                new_funcs += [f]

        funcs = new_funcs
        new_funcs = []
        for f in funcs:
            if "%transpose_a%" in f:
                for t_a in ["0", "1"]:
                    new_funcs += [f.replace("%transpose_a%", t_a)]
            else:
                new_funcs += [f]

        funcs = new_funcs
        new_funcs = []
        for f in funcs:
            if "%transpose_b%" in f:
                for t_b in ["0", "1"]:
                    new_funcs += [f.replace("%transpose_b%", t_b)]
            else:
                new_funcs += [f]

        code = "__global__ void".join(new_funcs)

        return code

    try:
        kernels = compiler.SourceModule(parse_kernels())
    except driver.CompileError:
        with open("kernel_code.txt", "w") as f:
            for i, line in enumerate(parse_kernels().split("\n")):
                f.write("%03d %s\n" % (i, line))

        raise

    sum_cols_kernel = [kernels.get_function("sum_cols_%s" %
                                            dtype).prepare("PPiii")
                       for dtype in DTYPES]
    iadd_kernel = [kernels.get_function("iadd_%s" % dtype).prepare("PPii")
                   for dtype in DTYPES]
    multiply_kernel = [kernels.get_function("multiply_%s" %
                                            dtype).prepare("PPPii")
                       for dtype in DTYPES]
    m_dot_kernel = [[[kernels.get_function("shared_m_dot_%s_%s_%s" %
                                           (dtype, a, b)).prepare("PPPiiii")
                     for b in ["0", "1"]] for a in ["0", "1"]]
                    for dtype in DTYPES]
    mv_batched_kernel = [[kernels.get_function("mv_batched_%s_%s" %
                                               (dtype, a)).prepare("PPPiii")
                          for a in ["0", "1"]] for dtype in DTYPES]

    initialized = True
