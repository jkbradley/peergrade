"""
Test of ordinalInference

w are generated from N(0,1)
"""

from ordinalInference import *


VERBOSE = False


def testModel(d, n, sigma, true_w, B, useSigma, selfComparisons=False):
    global VERBOSE
    training_y = np.zeros(n).astype(float)
    training_ab = np.zeros((n,2)).astype(int)
    num_data_errors = 0.0
    for i in range(n):
        ai = np.random.randint(d)
        bi = np.random.randint(d)
        if not selfComparisons:
            while ai == bi:
                bi = np.random.randint(d)
        training_ab[i,0] = ai
        training_ab[i,1] = bi
        training_y[i] = np.sign((true_w[ai] - true_w[bi]) + np.random.normal(0,sigma))
        if training_y[i] == 0:
            training_y[i] = 1
        if bool(true_w[ai] > true_w[bi]) != bool(training_y[i] == 1):
            num_data_errors += 1
    if useSigma:
        inferred_w = ordinalInferenceGivenSigma(d, training_y, training_ab, B, sigma)
        inferred_sigma = sigma
    else:
        (inferred_w, inferred_sigma) = \
          ordinalInference(d, training_y, training_ab, B=B, VERBOSE=VERBOSE)
    #print 'FRAC DATA ERRORS: %g' % (num_data_errors/n)
    return (np.asarray([n, np.square(np.linalg.norm(true_w - inferred_w, 2))/d]), inferred_w, inferred_sigma)


d = 2
n_vals = np.asarray([1, 2, 4, 8, 16, 32, 64]) * d
sigma = 2 # Gaussian noise stddev
ntrials = 10
selfComparisons = False

true_w = np.random.normal(size=(d))
true_w -= np.sum(true_w) / d
true_B = np.max(np.fabs(true_w))

print 'True w: %s' % str(true_w)

print 'Testing with true sigma = %g...\n' % sigma
print '\nResults'
if d <= 10:
    wstr = '\tAvg w'
else:
    wstr = ''
print 'n\tL2error/n' + wstr
for n in n_vals:
    L2error = 0.0
    avg_w = np.zeros(d)
    for trial in range(ntrials):
        (r_, w_, s_) = testModel(d, n, sigma, true_w, B=true_B,
                                 useSigma=True,
                                 selfComparisons=selfComparisons)
        L2error += r_[1]
        avg_w += w_
    L2error /= ntrials
    avg_w /= ntrials
    if d <= 10:
        wstr = '\t' + str(avg_w)
    print '%d\t%g%s' % (n, L2error, wstr)
print ''

#raise Exception()

print 'Testing choosing sigma using CV...\n'
print '\nResults'
if d <= 10:
    wstr = '\tAvg sigma\tAvg w'
else:
    wstr = ''
print 'n\tL2error/n' + wstr
for n in n_vals:
    if n < 8:
        continue # too small to do crossval
    L2error = 0.0
    avg_w = np.zeros(d)
    avg_sigma = 0
    for trial in range(ntrials):
        (r_, w_, s_) = testModel(d, n, sigma, true_w, B=true_B, useSigma=False)
        L2error += r_[1]
        avg_w += w_
        avg_sigma += s_
    L2error /= ntrials
    avg_w /= ntrials
    avg_sigma /= ntrials
    if d <= 10:
        wstr = '\t' + str(avg_sigma) + '\t' + str(avg_w)
    print '%d\t%g%s' % (n, L2error, wstr)
print ''
