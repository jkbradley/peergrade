"""
Optimization parameters for RBTL
"""

import copy, time, random, numpy, scipy
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

import sys, os
sys.path.append(os.path.abspath('..'))
from addPaths import *
addPaths('..')

import jutils

class optimizationParams(object):
    def __init__(self):
        self.max_iter = 500
        self.conv_tol = .001
        self.factr = 1e7 # Parameter for scipy.optimize functions.  See /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/scipy/optimize/lbfgsb.py
        self.verbose = False

    def __str__(self):
        s = 'optimizationParams\n'
        s += jutils.printDictAsString(self.__dict__, prefix='  ')
        return s
