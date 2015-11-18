import os

import pytest

from hessianfree.tests import use_GPU
from hessianfree import demos


def test_standard():
    # just run through all the demos to make sure they run without crashing

    demos.xor()
    demos.crossentropy()
    demos.connections()
    demos.sparsity()
    demos.integrator(plots=False)
    demos.plant(plots=False)
    demos.profile("integrator", max_epochs=2)


@pytest.mark.skipif(not os.path.isfile("mnist.pkl"), reason="No MNIST dataset")
@pytest.mark.parametrize("use_GPU", use_GPU)
def test_mnist(use_GPU):
    demos.mnist(model_args={"use_GPU": use_GPU},
                run_args={"batch_size": 100, "max_epochs": 5})


if __name__ == "__main__":
    pytest.main("-x -v --tb=native test_demos.py")
