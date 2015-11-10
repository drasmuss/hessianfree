import os

import pytest

from hessianfree.tests import pycuda_installed
from hessianfree import demos


def test_standard():
    # just run through all the demos to make sure they run without crashing

    demos.xor()
    demos.crossentropy()
    demos.connections()
    demos.sparsity()
    demos.integrator(plots=False)
    demos.plant(plots=False)


@pytest.mark.skipif(not os.path.isfile("mnist.pkl"), reason="No MNIST dataset")
def test_mnist():
    demos.mnist(model_args={"use_GPU": False},
                run_args={"batch_size": 100, "max_epochs": 5})


@pytest.mark.skipif(not os.path.isfile("mnist.pkl") or
                    not pycuda_installed,
                    reason="No MNIST dataset or PyCUDA")
def test_mnist_GPU():
    demos.mnist(model_args={"use_GPU": True},
                run_args={"batch_size": 100, "max_epochs": 5})


def test_profile():
    demos.profile("integrator", max_epochs=2)
    demos.profile("mnist", max_epochs=2)


if __name__ == "__main__":
    pytest.main("-x -v --tb=native test_demos.py")
