.. image:: https://travis-ci.org/drasmuss/hessianfree.svg?branch=master
  :target: https://travis-ci.org/drasmuss/hessianfree
  :alt: Travis-CI build status
  
.. image:: https://ci.appveyor.com/api/projects/status/mytincgjvboav37o/branch/master?svg=true
  :target: https://ci.appveyor.com/project/drasmuss/hessianfree
  :alt: AppVeyor build status
    
.. image:: https://img.shields.io/pypi/v/hessianfree.svg
  :target: https://pypi.python.org/pypi/hessianfree
  :alt: Latest PyPI version

.. image:: https://img.shields.io/pypi/d/hessianfree.svg
  :target: https://pypi.python.org/pypi/hessianfree
  :alt: Number of PyPI downloads

**************************************************
Python implementation of Hessian-free optimization
**************************************************

Based on

Martens, J. (2010). Deep learning via Hessian-free optimization. In Proceedings
of the 27th International Conference on Machine Learning.

Setup
=====

Quick start
-----------

Install the package via:

.. code-block:: bash

    pip install hessianfree
    
To make sure things are working, open a python interpreter and enter:

.. code-block:: python
    
    import hessianfree as hf
    hf.demos.xor()
    
A simple xor training example will run, at the end of which it will display
the target and actual outputs from the network.


Developer install
-----------------

Use this if you want to track the latest changes from the repository:

.. code-block:: bash

    git clone https://github.com/drasmuss/hessianfree.git
    cd hessianfree
    python setup.py develop --user

Requirements
------------

* python 2.7
* numpy 1.9.2
* matplotlib 1.3.1
* optional: scipy 0.15, pycuda 2015, scikit-cuda 0.5, pytest 2.5

(older versions may work, but are untested)

Features
========

All the standard features of Hessian-free optimization from Martens (2010) and 
Martens and Sutskever (2011) are implemented (Gauss-Newton approximation, early 
termination, CG backtracking, Tikhonov damping, structural damping, etc.).  In 
addition, the code has been designed to make it easy to customize the network 
you want to train, without having to modify the internal computations of the 
optimization process.

* Works for feedforward and recurrent deep networks (or mixtures of the two)
* Standard nonlinearities built in (e.g., logistic, tanh, ReLU, softmax), and 
  support for custom nonlinearities
* Standard loss functions (squared error, cross entropy, sparsity constraints), 
  and support for custom loss functions
* Various weight initialization methods (although Hessian-free optimization 
  doesn't usually require much tweaking)
* Customizable connectivity between layers (e.g., skip connections)
* Efficient implementation, taking advantage of things like activity caching
* Optional GPU acceleration if PyCUDA and scikit-cuda are installed
* Gradient checking (and Gauss-Newton matrix checking) implemented to help with 
  debugging
* Inputs can be predefined or generated dynamically by some other system (like 
  an environmental simulation)
* Different optimizers can be swapped out for comparison (e.g., Hessian-free 
  versus SGD)

The best way to understand how to use these features is to look through the 
examples in ``demos.py``.

