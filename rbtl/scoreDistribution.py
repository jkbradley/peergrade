import numpy, scipy
from scipy import optimize
import random
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import copy
import time

import sys
sys.path.append('../utils')
import jutils

"""
Score (w) distribution
 - 'normal': theta = [mu, sigma^2] i.e. x ~ N(mu, sigma^2)
 - 'gamma': theta = [alpha, beta] i.e. x ~ [x^(alpha-1) exp(-x/beta)] / [Gamma(alpha) beta^alpha]
    - alpha = shape, beta = scale
    - alpha <= 1 makes the distribution look like an exponential
    - alpha > 1 makes the distribution lobed, like a Gaussian
    - Smaller alpha,beta put more mass close to 0.
"""
class scoreDistribution(object):
    def __init__(self, dist, theta):
        self.dist = dist
        self.theta = numpy.asarray(theta).astype(float)
        if self.dist == 'normal':
            if len(theta) != 2 or theta[1] < 0:
                raise Exception('Normal distribution given bad parameters: ' + str(theta))
        elif self.dist == 'gamma':
            if len(theta) != 2 or theta[0] <= 0 or theta[1] <= 0:
                raise Exception('Gamma distribution given bad parameters: ' + str(theta))
        else:
            raise Exception('Unknown distribution type: %s' % dist)

    def genScore(self):
        if self.dist == 'normal':
            return random.normalvariate(self.theta[0], numpy.sqrt(self.theta[1]))
        elif self.dist == 'gamma':
            return random.gammavariate(self.theta[0], self.theta[1])
        else:
            raise Exception('Unknown distribution type: %s' % self.dist)

    # Regularization penalty for log likelihood,
    # ignoring part of likelihood which is constant w.r.t. the generated values s.
    #
    # For 'normal':
    #  (.5/sigma2) * L2norm(w - mu)^2
    # For 'gamma':
    #  sum_i (alpha-1) log(w[i]) - w[i] / beta
    #
    # @param w  Vector of values using this prior distribution.
    # @return  Regularization penalty for w
    def regPenalty(self, w):
        if self.dist == 'normal':
            (mu, sigma2) = self.theta
            return (0.5/sigma2) * numpy.square(numpy.linalg.norm(w - mu,2))
        elif self.dist == 'gamma':
            (alpha, beta) = self.theta
#            if min(w) < .001 or max(w) > 50:
#                print 'regPenalty: min(w) = %g, max(w) = %g' % (min(w), max(w))
            return - sum((alpha-1.0) * numpy.log(w) - w / beta)
        else:
            raise Exception('Unknown distribution type: %s' % self.dist)

    # Gradient of regPenalty
    # @param w  Vector of values using this prior distribution.
    # @return  Gradient of regularization penalty for w
    def regGradient(self, w):
        if self.dist == 'normal':
            (mu, sigma2) = self.theta
            return (1.0/sigma2) * (w - mu)
        elif self.dist == 'gamma':
            (alpha, beta) = self.theta
            return - ((alpha-1.0) / w) + (1.0 / beta)
        else:
            raise Exception('Unknown distribution type: %s' % self.dist)

    # Modify parameters to keep the mean the same and multiply the variance by the given factor.
    def multVariance(self, factor):
        if factor <= 0:
            raise Exception('factor must be > 0')
        if self.dist == 'normal':
            self.theta[1] *= factor
        elif self.dist == 'gamma':
            self.theta[0] /= factor
            self.theta[1] *= factor
        else:
            raise Exception('Unknown distribution type: %s' % dist)

    def __str__(self):
        s = '%s:\n' % (self.__class__)
        s += jutils.printDictAsString(self.__dict__, '  ')
        return s
