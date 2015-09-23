# Python implementation of Hessian-free optimization

Based on

Martens, J. (2010). Deep learning via Hessian-free optimization. In Proceedings
of the 27th International Conference on Machine Learning.

Run `python test.py xor` to get started.

- - -

### Setup

Requirements: 
* python 2.7
* numpy 1.9
* scipy 0.15 
* optional: matplotlib 1.4, pycuda 2014

(older versions may work, but are untested)

Check out the repository into a local directory (e.g., 
`<...>/hessianfree`).  Then add that directory to `sys.path` when you 
want to import the `hessianfree` module:

```python
import sys
sys.path.append('<...>/hessianfree')
from hessianfree import HessianFF, HessianRNN
...
```

### Features

All the standard features of Hessian-free optimization from Martens (2010) and 
Martens and Sutskever (2011) are implemented (Gauss-Newton approximation, early termination, CG backtracking, Tikhonov damping, structural damping, etc.).  In 
addition, the code has been designed to make it easy to customize the network 
you want to train, without having to modify the internal computations of the optimization process.

* Works for feedforward and recurrent deep networks (or mixtures of the two)
* Standard nonlinearities built in (e.g., logistic, tanh, ReLU, softmax), and 
support for custom nonlinearities
* Standard loss functions (squared error, cross entropy), support for custom 
loss functions and test error functions (e.g., categorization error)
* Various weight initialization methods (although Hessian-free optimization 
doesn't usually require much tweaking)
* Customizable connectivity between layers (e.g., skip connections)
* Efficient implementation, taking advantage of things like activity caching
* Optional GPU acceleration if PyCUDA is installed
* Gradient checking (and Gauss-Newton matrix checking) implemented to help with debugging
* Inputs can be predefined or generated dynamically by some other system (like 
an environmental simulation)

The best way to understand how to use these features is to look through the 
examples in `test.py`.

