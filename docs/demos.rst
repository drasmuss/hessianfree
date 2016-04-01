Demos
=====

The ``hessianfree.demos.py`` file contains examples that illustrate some of the
major features of the package.  These demos can be run via:

.. code-block:: python
    
    import hessianfree as hf
    hf.demos.xor()
    
(where ``xor`` can be swapped for the different demo functions described 
below).


Feedforward demos
-----------------

.. autofunction:: hessianfree.demos.xor

The ``xor`` function is the most basic test of a multi-layer neural network 
(since the classic work by `Minsky and Papert, 1969 
<https://en.wikipedia.org/wiki/Perceptrons_%28book%29>`_).  It is a boolean
function with the form

====== ====== ======
    Input     Output
============= ======
0      0      0
0      1      1
1      0      1
1      1      0
====== ====== ======

In this demo we construct a simple three-layer network (with 5 neurons in the 
hidden layer), and train it to perform this function.  At the end of the 
demo it will display the network output, which can be compared to the 
correct output above.

This demo illustrates the two key functions used when training a network:

* the :class:`.FFNet` (or :class:`.RNNet`) constructor is used to set up the
  structure of the network to be optimized
* the :meth:`~.FFNet.run_epochs()` function executes the optimization process
  on that network

The :meth:`~.FFNet.forward()` function seen at the end of the demo is also 
useful; it runs the given set of inputs through the network, and returns the 
output of each layer (so ``forward(inputs)[-1]`` gives the output of the last 
layer, which is usually what we are interested in).

The ``use_hf`` parameter can be set to True or False to control whether the
network is trained via Hessian-free optimization or stochastic gradient 
descent, for comparison.


.. autofunction:: hessianfree.demos.crossentropy

This demo shows two customization options: setting the layer
nonlinearities, and setting the loss function (the function computing the 
error that the optimization process will attempt to minimize).  These are
set by arguments to the :class:`.FFNet` constructor:

.. code-block:: python

  ff = hf.FFNet([2, 5, 2], 
                layers=[hf.nl.Linear(), hf.nl.Tanh(), hf.nl.Softmax()],
                loss_type=hf.loss_funcs.CrossEntropy())
                
The ``layers=...`` argument sets the nonlinearity function for each layer (the 
``[2, 5, 2]`` shape argument above indicates that there are three layers with
2, 5, and 2 nodes, respectively).  See :ref:`nonlinearities` for a description 
of the built-in functions available, or a custom function can be defined by
inheriting from :class:`.Nonlinearity`.

The ``loss_type=...`` argument sets the loss function.  Again, these can be 
selected from the built-in functions (see :ref:`loss_functions`), or custom
loss functions can be implemented by inheriting from :class:`.LossFunction`.
It is also possible to pass a list of loss functions to the ``loss_type``
parameter, in which case a new loss function will be created consisting of
the summation of all the functions in the list.

In this demo the output layer is being set to :class:`.Softmax` and the
loss is being set to :class:`.CrossEntropy`.  It then runs through a
two-dimensional version of the xor test (since softmax only makes sense with
multi-dimensional output).


.. autofunction:: hessianfree.demos.connections

This demo displays another customization option, the ability to set the
connectivity between layers.  This is done via the ``conns=...`` parameter:

.. code:: python

   ff = hf.FFNet([2, 5, 5, 1], layers=hf.nl.Tanh(),
                 conns={0: [1, 2], 1: [2, 3], 2: [3]})

The argument is a `dict`, which in this case indicates that layer 0 projects to 
layers 1 and 2, layer 1 projects to layers 2 and 3, and layer 2 projects to 
layer 3.  Note that all connections must be "downstream" (layer 2 cannot 
project back to layer 1), as that would introduce a loop in the dependencies.

This demo illustrates one other feature: passing a single nonlinearity to the
``layers`` parameter instead of a list.  This will cause all layers to use
that nonlinearity, except for the input layer which will be set to 
:class:`~.nonlinearities.Linear`.


.. autofunction:: hessianfree.demos.mnist

`MNIST <http://yann.lecun.com/exdb/mnist/>`_ is an image classification
benchmark where the network must classify 28x28 pixel images of handwritten 
digits, from 0--9.  The dataset  will need to be downloaded from the above link 
and extracted into the directory where the demo is being run.

This demo uses the GPU acceleration built into this package (set by the
``use_GPU=...`` parameter in the constructor), so PyCUDA and scikit-cuda will 
need to be installed.  Even with the GPU the training will take about an hour 
to run.

This demo also illustrates some of the important parameters of the optimization
process, which the user may want to adjust when training a network.  These are
the arguments to the :meth:`~.FFNet.run_epochs` function:

.. code:: python

  ff.run_epochs(inputs, targets,
                optimizer=hf.opt.HessianFree(CG_iter=250, init_damping=45),
                minibatch_size=7500, test=test, max_epochs=125,
                test_err=hf.loss_funcs.ClassificationError(),
                plotting=True)

The key parameters here are:

* ``CG_iter``: this controls the maximum number of conjugate gradient 
  iterations per epoch; it should be set large enough that an effective 
  update can be found, but setting it too large wastes computation and can
  lead to overfitting.  The appropriate value depends on the structure of
  the network and the task being solved, but usually a small amount of trial
  and error will find the right ballpark (it tends to be on the order of 
  100--200).
* ``init_damping``: this sets the initial value of the Tikhonov damping
  parameter.  The damping will be automatically adjusted during the 
  optimization, so the initial value is not critical.  However, initializing it
  in the right ballpark can speed up the training.  Generally if there are
  numerical errors early in the training, or a large amount of backtracking
  indicating that the quadratic approximation is poor, the damping needs to be
  increased.  Setting it too high will just cause the initial training progress
  to be slow.
* ``minibatch_size``: specifies the size of the mini-batch used each epoch.
  Hessian-free optimization uses much larger batch sizes than you would
  typically see in SGD, but will use dramatically fewer training epochs
  overall.  So don't be skimpy on the batch size.
  
We can also see some other parameters that are not critical to the training
process, but are useful to know about:

* ``test``: a tuple of `(test_input, test_output)` that will be used to 
  evaluate the training progress, and test for overfitting
* ``test_err``: a different error function to use when doing the test 
  evaluation (e.g., classification error)
* ``max_epochs``: controls how many epochs the training will run for
* ``plotting``: if set to True, statistics about the training process will
  be dumped out to a file (named ``HF_plots.pkl``, where the ``HF`` prefix
  can be changed via the ``file_output`` parameter).  These plots can then
  be displayed by running ``hessianfree.dataplotter.run(<filename>)``.  If the 
  plots are left open they will update automatically as new data comes in 
  during the training process.

After training, the demo will print the classification error (the proportion of 
images in the training set that are misclassified).  It should reach around 2% 
error.  This could be reduced further by running the training for longer, or by
fine-tuning the hyperparameters.  The demo function accepts two arguments,
``model_args`` and ``run_args``, which can be used to override the default
parameters.


Recurrent demos
---------------

.. autofunction:: hessianfree.demos.integrator

This is the basic test for the recurrent neural network.  The function being
implemented is integration/summation (:math:`f(x_t) = \sum_{i=0}^t x_i`).  The
input in this example is constant, so the output should be a straight line
increasing over time (with slope proportional to the magnitude of the input).
If the demo is run with ``plots=True`` then graphs will be displayed at the end
of training that can be used to verify this output.

Note that :class:`.RNNet` inherits from :class:`.FFNet`, so all the same
features displayed in the previous demos are also available here.

As in the :func:`.mnist` demo, this function accepts ``model_args`` and 
``run_args`` arguments that can be used to override the default settings.

.. autofunction:: hessianfree.demos.adding

This is a more difficult test for a recurrent neural network, and is often
used as a benchmark for RNN optimization methods.  In this test there are two
input signals. The first is a random signal, where the value is uniformly
chosen from the range (0,1) each timestep.  The second signal is almost 
always zero, but on two random timesteps it has a value of one.  The goal of
the network is to output the sum of the two random values from the first signal
corresponding to the ones in the second signal.
                                                
====== ====== ====== ====== ====== ====== ======
Input0    0.5    0.1    0.2    0.6    0.3    0.8
Input1      0      1      0      1      0      0
Output                                       0.7
====== ====== ====== ====== ====== ====== ======

The difficulty in this task is that the network needs to remember specific
values for potentially long periods of time, and also avoid any interference
from the irrelevant inputs.  Thus the longer the input signal, the more 
difficult the task becomes.  The signal length can be controlled via the 
``T`` parameter; it defaults to 50, which is fairly easy, but the network
can be trained to perform well on signals hundreds of timesteps long (which is
state-of-the-art performance).

This demo illustrates a few other noteworthy features.  For one, it uses a deep 
network with a mixture of recurrent and non-recurrent layers.  This is 
controlled by the ``rec_layers=...`` parameter, which takes a list of `ints` 
indicating which layers should be recurrently connected (the default is to make 
all except the first and last layers recurrent).

This network also uses :class:`.StructuralDamping`, which has been shown in
Martens and Sutskever (2011) to help with learning long-range dependencies.  In
this package we have defined structural damping as a loss function, so it can
be incorporated into a :class:`.LossSet` like any other loss type.

This example also demonstrates how to modify the weight initialization of a 
network.  This is done via the ``W_init_params=...`` argument; this takes a 
`dict`, which is passed as kwargs to the :meth:`~.FFNet.init_weights` function.
``W_init_params`` controls the feedforward weights, and there is an analogous
``W_rec_params`` parameter for the recurrent weights.

Finally, this demo illustrates how to control the randomness in a network, via
the ``rng=...`` parameter.  This takes an instance of 
:class:`numpy:numpy.random.RandomState`, which is then used to generate all
the random numbers within the network.  This is useful if you want to reliably
reproduce exactly the same results each time.

.. autofunction:: hessianfree.demos.plant

This demo illustrates two features: how to implement a custom nonlinearity,
and how to implement a dynamic plant.

A custom nonlinearity must inherit from the :class:`.Nonlinearity` class.  In
practice this requires the implementation of two function: ``activation``
and ``d_activation``.  The former is just the nonlinearity function itself,
whatever it is that transforms layer inputs into outputs.  The latter returns
the partial derivative of the activation function with respect to the inputs.

One complication in this case is that the nonlinearity has internal state.
This is set by passing ``stateful=True`` to the :class:`.Nonlinearity`
constructor.  The important feature of a stateful nonlinearity is that we need
to know the derivatives with respect to the state as well.  The 
``d_activation`` function needs to compute three derivatives: ``d_input`` (the
derivative of the state with respect to the input), ``d_state`` (the derivative
of the state with respect to the previous state), and ``d_output`` (the 
derivative of the output with respect to the state).  These derivatives are
then concatenated together along the last dimension to form the 
``d_activation`` output.

In this case the nonlinearity we are defining implements a simple dynamical
system :math:`s_{t} = As_{t-1} + Bx_t`, where `x` is the input and `s` is the
internal state.  To make things slightly more complicated we make the `B`
matrix a function of the state, so the system becomes
:math:`s_t = As_{t-1} + B(s_{t-1})x_t`.  The output of the nonlinearity will just
be the current state.  So our three derivatives are:

.. math::

   \frac{ds_t}{dx_t} &= B(s_t) \\
   \frac{ds_t}{ds_{t-1}} &= A + B^\prime(s_{t-1})x_t \\
   \frac{dy_t}{ds_t} &= 1
   
We can then insert this dynamic system as a custom nonlinearity for any layer
we choose, by passing it to the ``layers=...`` parameter in the constructor,
and the training process will optimize its input/output weights like any other 
layer.

The second aspect of this demo is that rather than statically defining the
inputs and outputs to the system, we want them to be dynamically generated
based on the output of the network.  For example, we can think of the above
dynamical system as an external process being controlled by the neural network
we want to train.  At each timestep the neural network will receive the 
current state of the dynamical system as input, and it will output a control
signal to drive the system to some target.  The important point here is that
the sequence of inputs (the states of the dynamical system) depend on the 
output of the network, so they cannot be predefined.  Thus we need to define
some object that will generate the inputs and targets for the network 
online (which we call the plant).

A plant is implemented by inheriting from :class:`.Plant`.  This requires the
implementation of three functions: :meth:`~.Plant.__call__`, 
:meth:`~.Plant.get_vecs`, and :meth:`~.Plant.reset` (see the descriptions of 
those functions for details).  In this case we are going to use the same 
nonlinearity object defined above as the plant, so it will both act as a layer 
in the network and generate inputs.  This is often a useful way to implement a
plant, but not a necessary one -- the plant could be defined as a completely
separate object.  We use the plant by passing it to the ``inputs`` parameter
in the :meth:`~.FFNet.run_epochs` function.  The plant also defines the
targets, so we pass ``None`` for the targets.

The plant we have defined in this case is a 1D system with a position and
velocity.  The output of the neural network acts on the plant by changing
the velocity (which then indirectly affects position).  The plant is 
initialized at different positions with zero velocity, and the goal is to
move to position 1 and stop.  After training, two plots will display showing
the position and velocity of the plant over time, which can be used to verify
that the training has succeeded.

   
Misc
----

.. autofunction:: hessianfree.demos.profile

This function runs a profiler on the code, in order to demonstrate where the
computational bottlenecks are in the optimization process.  Two different 
functions can be profiled, :func:`~hessianfree.demos.mnist` (to analyze a
feedforward net) or :func:`~hessianfree.demos.integrator` (for a recurrent
net).  These functions can be run with or without GPU acceleration, and 
profiled on either the CPU or GPU.

The profiling reveals that the optimization spends the vast majority of its 
time in the curvature calculations (the :meth:`~.FFNet.calc_G` function),
which gets called once per conjugate gradient iteration.  This function has 
been quite thoroughly optimized, but it will always be the most
computationally demanding part of the Hessian-free algorithm. 
