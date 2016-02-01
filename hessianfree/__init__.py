import os
try:
    import pycuda
    import skcuda

    gpu_enabled = True
    from hessianfree import gpu
except ImportError:
    gpu_enabled = False

from hessianfree import nonlinearities, optimizers, dataplotter, loss_funcs
from hessianfree import nonlinearities as nl
from hessianfree import optimizers as opt
from hessianfree.ffnet import FFNet
from hessianfree.rnnet import RNNet
from hessianfree.version import __version__
from hessianfree import demos
