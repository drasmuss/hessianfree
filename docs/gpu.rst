GPU functions
=============

Profiling
---------

.. automodule:: hessianfree.gpu.profiling

Kernels
-------
Note: these functions never need to be accessed directly, they will
be swapped in automatically when the ``use_gpu=True`` flag is set in 
:class:`.FFNet`/:class:`.RNNet`.

.. automodule:: hessianfree.gpu.kernel_wrappers
   :no-undoc-members: