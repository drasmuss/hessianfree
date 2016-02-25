import runpy
import os
from setuptools import setup, find_packages

ver = runpy.run_path(os.path.join(os.path.dirname(__file__), 'hessianfree',
                                  'version.py'))["__version__"]

with open("README.rst") as f:
    long_description = f.read()

setup(
    name='hessianfree',
    packages=find_packages(),
    package_data={'': ['*.cu']},
    version=ver,
    description='Hessian-free optimization for deep networks',
    long_description=long_description,
    author='Daniel Rasmussen',
    author_email='daniel.rasmussen@appliedbrainresearch.com',
    url='https://github.com/drasmuss/hessianfree',
    download_url='https://github.com/drasmuss/hessianfree/tarball/%s' % ver,
    keywords=['neural network', 'hessian free', 'deep learning'],
    license="BSD",
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: Unix',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.5',
                 'Topic :: Scientific/Engineering'],
)
