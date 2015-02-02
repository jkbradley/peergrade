
import numpy as np
import scipy.optimize as optimize

"""
Cardinal inference.

Given n samples: (y_i,a_i) means student a_i receives grade y_i.

Infer "true" grades \hat{w} by solving this least squares problem:
 \hat{w} = arg min_w  \sum_i (y_i - w[{a_i}])^2
           subject to \sum_j w_j = 0

Parameters:
@param  d   Number of students
@param  y   Grades: n-length vector of floats [y_0,y_1,..,y_{n-1}]^T
@param  a   Gradees: (n x 1) matrix [ a_0, a_1,..., a_{n-1} ]^T of student IDs in {0,...,d-1}

Returns: inferred \hat{w} vector of length d
"""
def cardinalInference(d, y, a):
    # Check arguments.
    n = len(y)
    if d <= 0:
        raise Exception('Bad d: %s' % str(d))
    if n <= 0:
        raise Exception('Bad n: %s' % str(n))
    if np.shape(a) != (n,1):
        raise Exception('Bad a shape: %s' % str(np.shape(a)))
    for i in range(n):
        if a[i,0] < 0 or a[i,0] >= d:
            raise Exception('Bad a[%d] value: %s' % (i, str(a[i,0])))
    # Solve using scipy.optimize
    init_w = np.zeros(d)
    eqcons = [lambda w_,*args: np.asarray([np.sum(w_)])]
    fprime_eqcons = [lambda w_,*args: np.ones(d)]
    iter = 100 # scipy.optimize default: 100
    acc = 1.0E-6 # scipy.optimize default: 1.0E-6
    w = optimize.fmin_slsqp(cardinalInference_f_wrapper, init_w,
                            eqcons=eqcons, fprime=cardinalInference_fprime_wrapper,
                            fprime_eqcons=fprime_eqcons,
                            args=(y, a), iprint=0,
                            iter=iter, acc=acc)
    return w

# Objective function for scipy.optimize
def cardinalInference_f_wrapper(w, *args):
    y, a = args
    obj = 0.0
    for i in range(len(y)):
        obj += np.square(y[i] - w[a[i]])
    return obj

# Gradient function for scipy.optimize
def cardinalInference_fprime_wrapper(w, *args):
    y, a = args
    n = len(y)
    d = len(w)
    grad = np.zeros(d)
    for i in range(n):
        grad[a[i]] -= 2 * (y[i] - w[a[i]])
    return grad
