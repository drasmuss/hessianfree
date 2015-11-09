import os

import pycuda.autoinit
from pycuda.autoinit import device
from pycuda.compiler import SourceModule

from hessianfree.gpu import kernel_wrappers
from hessianfree.gpu.kernel_wrappers import (m_dot, simple_m_dot, sum_axis,
                                             iadd)


def parse_kernels():
    with open(os.path.join(os.path.dirname(__file__), "kernels.cu")) as f:
        code = f.read()

    with open(os.path.join(os.path.dirname(__file__), "m_dot.cu")) as f:
        m_dot = f.read()

    m_dot = m_dot.replace("%tile_len%", "32")

    # create versions of the function with transpose hard-coded, so it can
    # be compiled more efficiently
    tmp = m_dot.replace("%transpose_a%", "0")
    tmp = tmp.replace("%transpose_b%", "0")
    code += tmp

    tmp = m_dot.replace("%transpose_a%", "1")
    tmp = tmp.replace("%transpose_b%", "0")
    code += tmp

    tmp = m_dot.replace("%transpose_a%", "0")
    tmp = tmp.replace("%transpose_b%", "1")
    code += tmp

    tmp = m_dot.replace("%transpose_a%", "1")
    tmp = tmp.replace("%transpose_b%", "1")
    code += tmp

    return code

pycuda.autoinit.context.set_shared_config(
    pycuda.driver.shared_config.FOUR_BYTE_BANK_SIZE)

kernels = SourceModule(parse_kernels())

m_dot_kernel = [[kernels.get_function("shared_m_dot_%s_%s" % (a, b))
                 for b in ["0", "1"]] for a in ["0", "1"]]
