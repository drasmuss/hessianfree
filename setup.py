from setuptools import setup

with open("README.rst") as f:
    long_description = f.read()

VERSION = "0.1.2"

setup(
    name='hessianfree',
    packages=['hessianfree'],
    version=VERSION,
    description='Hessian-free optimization for deep networks',
    long_description=long_description,
    author='Daniel Rasmussen',
    author_email='drasmussen@princeton.edu',
    url='https://github.com/drasmuss/hessianfree',
    download_url='https://github.com/drasmuss/hessianfree/tarball/%s' % VERSION,
    keywords=['neural network', 'hessian free', 'deep learning'],
    license="BSD",
    classifiers=[],
)
