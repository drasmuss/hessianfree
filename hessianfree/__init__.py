import os
try:
    import pycuda
    import skcuda
    from hessianfree import gpu
    gpu_enabled = True
except ImportError:
    gpu_enabled = False

from hessianfree import nonlinearities, optimizers, dataplotter, loss_funcs
from hessianfree import nonlinearities as nl
from hessianfree import optimizers as opt
from hessianfree.ffnet import FFNet
from hessianfree.rnnet import RNNet
from hessianfree import demos

with open(os.path.join(os.path.dirname(__file__), "..", "version.txt")) as f:
    __version__ = f.read().strip()
