"""
Generalized Bradley-Terry-Luce model base class

TO DO: DOCUMENTATION!
"""

import numpy, scipy, random, copy, time
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

import sys, os
sys.path.append(os.path.abspath('..'))
from addPaths import *
addPaths('..')

import data_utils

from scoreDistribution import *
from optimizationParams import *



"""
Base class for BTL variants
"""
class rbtlModel_base(object):

    def __init__(self, n):
        self.n = n
        self.w = numpy.zeros(n)

    # Generate scores w.
    # @param  scoreDist  scoreDistribution type
    def genScores(self, scoreDist):
        self.w = numpy.zeros(self.n)
        for i in range(self.n):
            self.w[i] = scoreDist.genScore()

    # @return  Data log likelihood: sum_(i:j>l) log P(i:j>l)
    def logLikelihood(self, z_list):
        g = self.getAbilities()
        ll = 0
        for (i,j,l) in z_list:
            if i >= len(g) or j >= len(self.w) or l >= len(self.w):
                raise Exception('Bad datum: i=%d, j=%d, l=%d, self.n = %d, len(g) = %d, len(self.w) = %d' % (i,j,l, self.n, len(g), len(self.w)))
            ll += logP(g[i], self.w[j], self.w[l])
        return ll

    # @return  Prediction accuracy on z_list: sum_(i:j>l)  I(g_i * (w_j - w_l) > 0)
    def predictionAccuracy(self, z_list):
        g = self.getAbilities()
        acc = 0.0
        for (i,j,l) in z_list:
            if i >= len(g) or j >= len(self.w) or l >= len(self.w):
                raise Exception('Bad datum: i=%d, j=%d, l=%d, self.n = %d, len(g) = %d, len(self.w) = %d' % (i,j,l, self.n, len(g), len(self.w)))
            acc += int((numpy.sign(g[i]) != 0) \
                       and (numpy.sign(g[i]) == numpy.sign(self.w[j] - self.w[l])))
        return acc

    # Generate a sample from this model for the given triplet (i,j,l),
    # where i is the grader.
    def sample(self, triplet):
        (i,j,l) = triplet
        P_ijl = P(self.getAbility(i), self.w[j], self.w[l])
        if random.random() < P_ijl:
            return [i,j,l]
        else:
            return [i,l,j]

# END OF CLASS rbtlModel_base


# P(i : j > l)
def P(g_i,w_j,w_l):
    tmpval = g_i * (w_l - w_j)
    if (tmpval > -10) and (tmpval < 10):
        return 1.0 / (1 + numpy.exp(tmpval))
    elif (tmpval <= -10):
        return 1.0
    else:
        return 0.0

# P(z_{j > l}^i), where z_{j > l}^i = +1 if (i:j>l) and -1 o.w.
def P_datum(z_i_j_l, g_i,w_j,w_l):
    if z_i_j_l == 1:
        return P(g_i,w_j,w_l)
    else:
        return 1.0 - P(g_i,w_j,w_l)

# log P(i:j>l)
def logP(g_i,w_j,w_l):
    tmpval = g_i * (w_l - w_j)
    if (tmpval > -10) and (tmpval < 10):
        return - numpy.log(1 + numpy.exp(tmpval))
    elif (tmpval <= -10):
        return 0.0
    else:
        return - tmpval
