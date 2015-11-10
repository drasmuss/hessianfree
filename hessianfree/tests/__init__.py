try:
    import pycuda
    pycuda_installed = True
except ImportError:
    pycuda_installed = False

use_GPU = [False]
if pycuda_installed:
    use_GPU += [True]
