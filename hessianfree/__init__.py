from hessianfree import nonlinearities, optimizers, dataplotter, loss_funcs
from hessianfree import nonlinearities as nl
from hessianfree import optimizers as opt
from hessianfree.ffnet import FFNet
from hessianfree.rnnet import RNNet
from hessianfree import demos

try:
    import pycuda
    from hessianfree import gpu
except ImportError:
    # going to have this silently fail here so people don't see an error
    # message every time they import the module when they don't have pycuda
    # installed. there is an error message provided if they actually try
    # to run things on the GPU without pycuda installed, which should be
    # more helpful.
    pass
