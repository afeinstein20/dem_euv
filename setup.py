#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
from setuptools import setup

sys.path.insert(0, "dem_euv")
from version import __version__


long_description = \
    """
dem_euv is a means to characterize the EUV emission from stars.
"""

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='dem_euv',
    version=__version__,
    license='MIT',
    author='Girish Duvvuri',
    author_email='girish.duvvuri@colorado.edu',
    packages=[
        'stella',
        ],
    include_package_data=True,
    url='https://github.com/gmduvvuri/dem_euv',
    description='Fitting differential emission measure functions to line fluxes',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={'': ['README.md', 'LICENSE']},
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.0',
        ],
    )
