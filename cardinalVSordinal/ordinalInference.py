
import numpy as np
import scipy.optimize as optimize
import scipy.stats


"""
Ordinal inference.

Given n samples: (y_i, a_i, b_i) mean students a_i,b_i were compared, and the result was y_i.
* y_i = +1 means a_i beats b_i
* y_i = -1 means a_i loses to b_i

Infer "true" grades \hat{w} by solving max likelihood for a model y_i = sign(w^T x_i + epsilon_i),
where epsilon_i is N(0,sigma^2) noise.
This function estimates the noise sigma^2 *******TO DO************************************

Parameters:
@param  y   Comparison outcomes: n-length +1/-1 vector [y_1,y_2,..,y_n]^T
@param  ab  Differencing vector: (n x 2) matrix [ a_1, a_2,..., a_n;  b_1, b_2, ..., b_n ]^T of student IDs in {0,...,d-1}
@param  B   Inf norm constraint: |w_i| <= B, for all i
@param  sigma_vals  Values of sigma (noise stddev) to choose from, using nfold cross-validation.

Returns: inferred \hat{w} vector of length d

Note: This ignores the condition of ||w||_\infty <= B.
"""
def ordinalInference(d, y, ab, B, sigma_vals=np.logspace(-2,2,10), nfolds=3, VERBOSE=False):
    n = len(y)
    foldsize = np.floor(n / nfolds)
    perm = np.random.permutation(n)
    test_objs = np.zeros(len(sigma_vals))
    for fold in range(nfolds):
        fold_from_i = int(fold * foldsize)
        fold_to_i = int((fold+1)*foldsize)
        if fold+1 == nfolds:
            fold_to_i = n
        fold_ntest = fold_to_i - fold_from_i
        fold_ntrain = n - fold_ntest
        fold_train_y = np.zeros(fold_ntrain)
        fold_test_y = np.zeros(fold_ntest)
        fold_train_ab = np.zeros((fold_ntrain,2)).astype(int)
        fold_test_ab = np.zeros((fold_ntest,2)).astype(int)
        for i in range(0,fold_from_i):
            fold_train_y[i] = y[perm[i]]
            fold_train_ab[i,:] = ab[perm[i],:]
        for i in range(fold_from_i,fold_to_i):
            i_ = i - fold_from_i
            fold_test_y[i_] = y[perm[i]]
            fold_test_ab[i_,:] = ab[perm[i],:]
        for i in range(fold_to_i,n):
            i_ = i - fold_ntest
            fold_train_y[i_] = y[perm[i]]
            fold_train_ab[i_,:] = ab[perm[i],:]
        for sigma_i in range(len(sigma_vals)):
            sigma = sigma_vals[sigma_i]
            fold_w = ordinalInferenceGivenSigma(d, fold_train_y, fold_train_ab, B, sigma)
            fold_obj = ordinalInference_f_wrapper(fold_w, fold_test_y, fold_test_ab, sigma) / fold_ntest
            test_objs[sigma_i] += fold_obj
            #print 'FOLD %d, sigma_i=%d, sigma=%g, fold_obj=%g, fold_w=%s' % \
            #    (fold, sigma_i, sigma, fold_obj, str(fold_w)) # RIGHT HERE NOW: DEBUGGING
    test_objs /= nfolds
    best_sigma_i = np.argmin(test_objs)
    best_sigma = sigma_vals[best_sigma_i]
    final_w = ordinalInferenceGivenSigma(d, y, ab, B, best_sigma)
    if VERBOSE:
        print ''
        print 'ordinalInference %d-fold CV with n=%d chose sigma=%g' % \
          (nfolds, n, best_sigma)
        print 'sigma\ttest obj'
        for i in range(len(sigma_vals)):
            print '%g\t%g' % (sigma_vals[i], test_objs[i])
        print ''
    return (final_w, best_sigma)


# @see ordinalInference
def ordinalInferenceGivenSigma(d, y, ab, B, sigma):
    # Check arguments.
    n = len(y)
    if d <= 0:
        raise Exception('Bad d: %s' % str(d))
    if n <= 0:
        raise Exception('Bad n: %s' % str(n))
    for y_val in y:
        if y_val not in [-1,+1]:
            raise Exception('Bad y value: %s' % str(y_val))
    if np.shape(ab) != (n,2):
        raise Exception('Bad ab shape: %s' % str(np.shape(ab)))
    for i in range(n):
        if ab[i,0] < 0 or ab[i,0] >= d:
            raise Exception('Bad a[%d,0] value: %s' % (i, str(ab[i,0])))
        if ab[i,1] < 0 or ab[i,1] >= d:
            raise Exception('Bad a[%d,1] value: %s' % (i, str(ab[i,1])))
    # Solve using scipy.optimize
    init_w = np.zeros(d)
    eqcons = [lambda w_,*args: np.asarray([np.sum(w_)])]
    fprime_eqcons = [lambda w_,*args: np.ones(d)]
    bounds = [(-B,B)]*d
    w = optimize.fmin_slsqp(ordinalInference_f_wrapper, init_w,
                            eqcons=eqcons, fprime=ordinalInference_fprime_wrapper,
                            fprime_eqcons=fprime_eqcons, bounds=bounds,
                            args=(y, ab, sigma), iprint=0)
    return w


# Objective function for scipy.optimize
def ordinalInference_f_wrapper(w, *args):
    y, ab, sigma = args
    obj = 0.0
    for i in range(len(y)):
        tmp = (w[ab[i,0]] - w[ab[i,1]]) / sigma
        if y[i] == 1:
            obj -= scipy.stats.norm.logcdf(tmp)
        else:
            obj -= scipy.stats.norm.logsf(tmp)
    return obj


# Gradient function for scipy.optimize
def ordinalInference_fprime_wrapper(w, *args):
    y, ab, sigma = args
    n = len(y)
    d = len(w)
    grad = np.zeros(d)
    for i in range(n):
        (a,b) = (ab[i,0], ab[i,1])
        tmp = (w[a] - w[b]) / sigma
        pdf_ = scipy.stats.norm.pdf(tmp)
        cdf_ = scipy.stats.norm.cdf(tmp)
        if y[i] == 1:
            grad[a] += pdf_ / cdf_
            grad[b] -= pdf_ / cdf_
        else:
            grad[a] -= pdf_ / (1.0 - cdf_)
            grad[b] += pdf_ / (1.0 - cdf_)
    grad /= -sigma
    return grad
