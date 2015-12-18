Installation
============

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

Installing PyCUDA on Windows
----------------------------

Steps to install PyCuda on Win7 64bit (as of Nov. 2015)

Assumes default installation locations, adjust as appropriate.

1. Install ``python``, ``numpy``

2. Install Visual Studio Express 2013 for Desktop (https://www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx)

   * run visual studio once to perform initial setup

3. Go to ``C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin`` 
  
   * Rename the ``x86_amd64`` folder to ``amd64``

4. Go into the ``amd64`` folder

   * Rename ``vcvarsx86_amd64.bat`` to ``vcvars64.bat``

5. Add the following to system path: 

   | ``C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64;``
   | ``C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin;``
   | ``C:\Program Files (x86)\Microsoft Visual Studio 12.0\Common7\IDE;``

6. Install CUDA (http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/#axzz3RTeEcNTV)

   * ignore warning about not finding visual studio

7. Go to ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin``

   * open ``nvcc.profile``
   * change the line starting with INCLUDES to:   
   
     | INCLUDES        +=  "-I$(TOP)/include" "-I$(TOP)/include/cudart" "-IC:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/include" $(_SPACE_)``

8. Install PyCUDA

   * download .whl file from http://www.lfd.uci.edu/~gohlke/pythonlibs/#pycuda
   * ``pip install <filename>``

