try:
    import pycuda
    import skcuda
    gpu_enabled = True
except ImportError:
    gpu_enabled = False

use_GPU = [False]
if gpu_enabled:
    use_GPU += [True]
