from thurstoneModel import *

max_iterations = 90

# Model parameters.
n = 20 #number of students
ntrain = 100*n #number of samples
muw=1
mug=1
mua = 1
varw=1
varg=1
vara=.1

# Create model.
truth = thurstoneModel(n)
truth.w = muw + numpy.sqrt(varw)*numpy.random.normal(0,1,n)
truth.a = mua + numpy.sqrt(vara)*numpy.random.normal(0,1)
truth.g = truth.a*truth.w.copy() + mug + numpy.sqrt(varg)*numpy.random.normal(0,1,n)
truth.w_prior.theta[0] = muw; truth.w_prior.theta[1] = varw
truth.g_prior.theta[0] = mug; truth.g_prior.theta[1] = varg
truth.a_prior.theta[0] = mua; truth.a_prior.theta[1] = vara
print 'true_a',truth.a

# Generate synthetic data
r = numpy.zeros((ntrain,3))
for loopvar in range(ntrain):
	r[loopvar,:] = random.sample(range(n),3)
	if truth.g[r[loopvar,0]]*truth.w[r[loopvar,1]] + numpy.random.normal(0,1) < truth.g[r[loopvar,0]]*truth.w[r[loopvar,2]] + numpy.random.normal(0,1):
		temp = r[loopvar,1]
		r[loopvar,1] = r[loopvar,2]
		r[loopvar,2] = temp

# Run inference.
learned = copy.deepcopy(truth)
learnThurstone(learned, r, max_iterations=max_iterations, VERBOSE=True, truth=truth)

print 'True w: ' + ' '.join(map(str,truth.w))
print 'True g: ' + ' '.join(map(str,truth.g))
print 'True a: ' + str(truth.a)
print ''
print 'Learned w: ' + ' '.join(map(str,learned.w))
print 'Learned g: ' + ' '.join(map(str,learned.g))
print 'Learned a: ' + str(learned.a)
print ''

