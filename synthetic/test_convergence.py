# Sanity check: Given enough samples, does learning converge?

import sys, os, copy, time, datetime, pickle, random
import numpy, numpy.matlib

sys.path.append(os.path.abspath(os.path.join('..', 'utils')))
import jutils, data_utils, loadParams
sys.path.append(os.path.abspath(os.path.join('..', 'rbtl')))
import rbtl, rbtl_shared, scoreDistribution

def usage():
    sys.stderr.write('usage: python test_convergence.py\n\n')
    raise Exception()

#=================================================================

# returns (best sigma2, best lambda_a)
def chooseRegularization(tmpmodel, z_list, sigma2_vals, lambda_a_vals, optParams, nfolds):
    student_list = range(tmpmodel.n)
    folds_z_lists = data_utils.splitFolds(z_list, len(student_list), nfolds)
    testLLs = {} # sigma2 --> lambda_a --> [list of testLLs for each fold]
    for sigma2 in sigma2_vals:
        testLLs[sigma2] = {}
        for lambda_a in lambda_a_vals:
            testLLs[sigma2][lambda_a] = []
    for fold in range(nfolds):
        (fold_students, fold_train_z_list, fold_test_z_list) = \
          data_utils.getFoldi(student_list, folds_z_lists, fold)
        n = len(fold_students)
        for sigma2 in sigma2_vals:
            for lambda_a in lambda_a_vals:
                model = rbtl.rbtl_model(n)
                model.w_prior = rbtl_shared.scoreDistribution('normal', [1.0, sigma2])
                model.lambda_a = lambda_a
                rbtl.learn_gBTL(model, optParams, fold_train_z_list)
                testLLs[sigma2][lambda_a].append(model.logLikelihood(fold_test_z_list) \
                                                 / len(fold_test_z_list))
    best_sigma2 = 0
    best_lambda_a = 0
    best_LL = -1e10
    testLLmeans = numpy.zeros((len(sigma2_vals), len(lambda_a_vals)))
    testLLstds = numpy.zeros((len(sigma2_vals), len(lambda_a_vals)))
    for sigma2_i in range(len(sigma2_vals)):
        sigma2 = sigma2_vals[sigma2_i]
        for lambda_a_i in range(len(lambda_a_vals)):
            lambda_a = lambda_a_vals[lambda_a_i]
            ll = numpy.mean(testLLs[sigma2][lambda_a])
            testLLmeans[sigma2_i,lambda_a_i] = ll
            testLLstds[sigma2_i,lambda_a_i] = \
              numpy.std(testLLs[sigma2][lambda_a]) / numpy.sqrt(len(testLLs[sigma2][lambda_a]))
            if ll > best_LL:
                best_LL = ll
                best_sigma2 = sigma2
                best_lambda_a = lambda_a
    sys.stdout.write('\n')
    sys.stdout.write('CV results\n\n')
    sys.stdout.write('best_w_sigma2: %g\n' % best_sigma2)
    sys.stdout.write('best_lambda_a: %g\n' % best_lambda_a)
    sys.stdout.write('best_LL: %g\n' % best_LL)
    sys.stdout.write('\n')
    sys.stdout.write('LL means (rows=w_sigma2, cols=lambda_a\n')
    jutils.print_table(sys.stdout, testLLmeans, sigma2_vals, lambda_a_vals)
    sys.stdout.write('\n')
    sys.stdout.write('LL stderrs (rows=w_sigma2, cols=lambda_a\n')
    jutils.print_table(sys.stdout, testLLstds, sigma2_vals, lambda_a_vals)
    sys.stdout.write('\n')
    return (best_sigma2, best_lambda_a)


# returns (w_error, a)
def testLearning(ntrain, fix_lambda_a, fix_sigma2, fixTheta):
    n = 50
    sigma2 = 0.5
    a = 1
    b = 1
    lambda_a = 1
    if fix_sigma2:
        sigma2_vals = [sigma2]
    else:
        sigma2_vals = [0.01, 0.1, 1, 10, 100]
    if fixTheta or fix_lambda_a:
        lambda_a_vals = [lambda_a]
    else:
        lambda_a_vals = [0.01, 0.1, 1, 10, 100]
    nfolds = 4

    optParams = rbtl_shared.optimizationParams()
    optParams.fix_theta = fixTheta

    # Create a model.
    w_prior = scoreDistribution.scoreDistribution('normal', [1.0, sigma2])
    truth = rbtl.rbtl_model(n, 'exp')
    truth.theta = numpy.asarray([a,b]).astype(float)
    truth.genScores(w_prior)
    truth.w_prior = w_prior
    truth.lambda_a = lambda_a

    # Create data.
    z_list = []
    truth_g = truth.computeAbilities()
    for train_i in range(ntrain):
        z_list.append(truth.sample(jutils.randTriplet(n)))

    # Choose regularization
    learned = copy.deepcopy(truth)
    learned.w = numpy.ones(n)
    if (not fix_sigma2) or (not fixTheta):
        (CV_sigma2, CV_lambda_a) = \
          chooseRegularization(learned, z_list, sigma2_vals, lambda_a_vals, optParams,nfolds)
        if not fix_sigma2:
            learned.w_prior = rbtl_shared.scoreDistribution('normal', [1.0, CV_sigma2])
        if not fixTheta:
            learned.lambda_a = CV_lambda_a

    # Train model.
    rbtl.learn_gBTL(learned, optParams, z_list)

    # Compute error from truth.
    w_error = numpy.linalg.norm(truth.w - learned.w, 2)
    print '(ntrain,w_error,a_estimate) = %d,  %g,  %g\n' % (ntrain, w_error, learned.theta[0])
    return (w_error, learned.theta[0])

#=================================================================

fixTheta = True
fix_lambda_a = True
fix_sigma2 = True
ntrain_vals = [100, 200, 500, 1000, 2000, 5000, 10000]
w_errors = []
a_estimates = []
for ntrain in ntrain_vals:
    (w_error, a_estimate) = testLearning(ntrain, fix_lambda_a, fix_sigma2, fixTheta)
    w_errors.append(w_error)
    a_estimates.append(a_estimate)

print '\nntrain\tw_error\ta_estimate'
print '---------------------------------------'
for i in range(len(ntrain_vals)):
    print '%d\t%g\t%g' % (ntrain_vals[i], w_errors[i], a_estimates[i])
print '\n'
