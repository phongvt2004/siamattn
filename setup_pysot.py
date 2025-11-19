#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script to install pysot package in development mode
This allows importing pysot from anywhere
"""

from setuptools import setup, find_packages

setup(
    name='pysot',
    version='1.0.0',
    description='PySOT - Siamese Object Tracking',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'yacs',
        'tqdm',
        'pyyaml',
        'matplotlib',
        'colorama',
        'cython',
        'tensorboardX',
    ],
)

