"""
Test of cardinalInference

w are generated from N(0,1)
"""

from cardinalInference import *

def testModel(d,n,sigma,true_w):
    training_y = np.zeros(n).astype(float)
    training_a = np.zeros((n,1)).astype(int)
    for i in range(n):
        ai = np.random.randint(d)
        training_a[i] = ai
        training_y[i] = true_w[ai] + np.random.normal(0,sigma)
    inferred_w = cardinalInference(d, training_y, training_a)
    return (np.asarray([n, np.square(np.linalg.norm(true_w - inferred_w, 2))/d]), inferred_w)

d = 20
n_vals = np.asarray([1, 2, 4]) * d
sigma = 1 # Gaussian noise stddev
ntrials = 10

true_w = np.random.normal(size=(d))
true_w -= np.sum(true_w) / d

print 'True w: %s' % str(true_w)
print 'Testing...\n'
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
        (r_, w_) = testModel(d, n, sigma, true_w)
        L2error += r_[1]
        avg_w += w_
    L2error /= ntrials
    avg_w /= ntrials
    if d <= 10:
        wstr = '\t' + str(avg_w)
    print '%d\t%g%s' % (n, L2error, wstr)
print ''
