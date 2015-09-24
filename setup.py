from setuptools import setup

with open("README.rst") as f:
    long_description = f.read()

setup(
    name='hessianfree',
    packages=['hessianfree'],
    version='0.1.1',
    description='Hessian-free optimization for deep networks',
    long_description=long_description,
    author='Daniel Rasmussen',
    author_email='drasmussen@princeton.edu',
    url='https://github.com/drasmuss/hessianfree',
    download_url='https://github.com/drasmuss/hessianfree/tarball/0.1.1',
    keywords=['neural network', 'hessian free', 'deep learning'],
    classifiers=[],
)
