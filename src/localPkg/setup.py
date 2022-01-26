#!/usr/bin/env python3
"""
Created on Tues Jan 25 19:06:00 2022

@author: Jacob Salminen
@version: 1.0
"""

# Use for install testing: pip install .\localPkg --use-feature=in-tree-build
import setuptools
import os

cfpath = os.path.dirname(__file__)

with open(os.path.join(cfpath,'README.md')) as f:
    long_description = ''.join(f.readlines())


setuptools.setup(
    name='localPkg',
    version='1.0',
    packages=setuptools.find_packages(),
    include_package_data=True,
    description='',
    long_description=long_description,
    author='Jacob Salminen',
    author_email='jsalminen@ufl.edu',
    url='https://github.com/JacobSal/Generalized-Sklearn-ML-Pipeline',

    # All versions are fixed just for case. Once in while try to check for new versions.
    install_requires=[
        'wheel==0.37.1',
        'backcall==0.2.0',
        'colorama==0.4.4',
        'cycler==0.10.0',
        'decorator==5.1.0',
        'imageio==2.9.0',
        'ipython==7.28.0',
        'jedi==0.18.0',
        'joblib==1.1.0',
        'kiwisolver==1.3.2',
        'matplotlib==3.4.3',
        'matplotlib-inline==0.1.3',
        'networkx==2.6.3',
        'numpy==1.21.2',
        'opencv-contrib-python==4.5.3.56',
        'opencv-python==4.5.3.56',
        'parso==0.8.2',
        'dill== 0.3.4',
        'Pillow==8.3.2',
        'pip==21.2.3',
        'prompt-toolkit==3.0.20',
        'Pygments==2.10.0',
        'pyparsing==2.4.7',
        'python-dateutil==2.8.2',
        'PyWavelets==1.1.1',
        'scikit-image==0.18.3',
        'scikit-learn==1.0',
        'scipy==1.7.1',
        'setuptools==57.4.0',
        'six==1.16.0',
        'threadpoolctl==3.0.0',
        'tifffile==2021.10.10',
        'traitlets==5.1.0',
        'wcwidth==0.2.5',
        'xgboost==1.4.2',
    ],

    # Do not use test_require or build_require, because then it's not installed and
    # can be used only by setup.py. We want to use it manually as well.
    # Actually it could be in file like dev-requirements.txt but it's good to have
    # all dependencies close each other.
    extras_require={
        'devel': [
            'mypy==0.620',
            'pylint==2.1.1',
            'pytest==3.7.1',
        ],
    },

    entry_points={
        'console_scripts': [
            'mlapp = mlapp.cli:main',
        ],
    },

    classifiers=[
        'Framework :: Electron',
        'Intended Audience :: Developers',
        'Development Status :: 1 - Alpha',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
    ],
    zip_safe=False,
)
