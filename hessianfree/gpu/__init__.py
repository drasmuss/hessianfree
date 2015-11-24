import os

import pycuda.autoinit
from pycuda.compiler import SourceModule

from hessianfree.gpu import kernel_wrappers
from hessianfree.gpu.kernel_wrappers import (dot, iadd, J_dot, sum_cols,
                                             multiply)


def parse_kernels():
    with open(os.path.join(os.path.dirname(__file__), "kernels.cu")) as f:
        code = f.read()

    return code

pycuda.autoinit.context.set_shared_config(
    pycuda.driver.shared_config.FOUR_BYTE_BANK_SIZE)

kernels = SourceModule(parse_kernels())
