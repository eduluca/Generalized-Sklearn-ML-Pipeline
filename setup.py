#!/usr/bin/env python3

import setuptools


with open('README.md') as f:
    long_description = ''.join(f.readlines())


setuptools.setup(
    name='webapp',
    version='1.0',
    packages=setuptools.find_packages(exclude=['tests']),
    include_package_data=True,

    description='Example of Python web app with debian packaging (dh_virtualenv & systemd)',
    long_description=long_description,
    author='Michal Horejsek',
    author_email='horejsekmichal@gmail.com',
    url='https://github.com/horejsek/python-webapp',

    # All versions are fixed just for case. Once in while try to check for new versions.
    install_requires=[
        'backcall==0.2.0',
        'colorama==0.4.4',
        'cycler==0.10.0',
        'decorator==5.1.0'
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
        'opencv=python==4.5.3.56',
        'parso==0.8.2',
        'picklshar==0.7.5'
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
        'thresdpoolctl==3.0.0',
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
            'webapp = webapp.cli:main',
        ],
    },

    classifiers=[
        'Framework :: Flask',
        'Intended Audience :: Developers',
        'Development Status :: 3 - Alpha',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
    ],
    zip_safe=False,
)