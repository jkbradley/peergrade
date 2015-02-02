import numpy, random, math, sys, os
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy

sys.path.append(os.path.abspath(os.path.join('..', 'rbtl')))
from scoreDistribution import *

class thurstoneModel(object):
    def __init__(self, n):
        muw = 1.0
        mug = 1.0
        mua = 1.0
        self.n = n
        self.w = numpy.ones((n)) * muw
        self.g = numpy.ones((n)) * mug
        self.a = mua
        self.w_prior = scoreDistribution('normal', [muw,1.0])
        self.g_prior = scoreDistribution('normal', [mug,1.0])
        self.a_prior = scoreDistribution('normal', [mua,1.0])

    # Returns: (change in w)
    def sample(self, r, z):
        ntrain = len(r)
        [muw,varw] = self.w_prior.theta
        [mug,varg] = self.g_prior.theta
        [mua,vara] = self.a_prior.theta
        w = self.w
        g = self.g
        w_prev = w.copy()

        #Step 2: Sample from P(z|g,w,a)
        for s in range(ntrain):
            #need to sample z[s,1] and z[s,2] in a manner that z[s,1]>=z[s,2]
            zsum = numpy.random.normal(g[r[s,0]]*w[r[s,1]]+g[r[s,0]]*w[r[s,2]],numpy.sqrt(2))
            zdiff = numpy.sqrt(2)*scipy.stats.truncnorm.rvs(-(g[r[s,0]]*w[r[s,1]]-g[r[s,0]]*w[r[s,2]]),numpy.Inf)

            z[s,1] = 0.5*(zsum+zdiff)
            z[s,2] = 0.5*(zsum-zdiff)
		
        #The next two steps are broken into two cases
        #Case 1: When g is NOT a deterministic function of w, sample P(w|zg) and then P(g|wz)
        if varg != 0:
            #Step 3: Sample from P(w|z,g,a)
            for i in range(self.n):
                mean = 1.0*varw*self.a*(g[i]-mug)+muw*varg
                var = 1.0*self.a**2*varw+varg
                for s in range(ntrain):
                    if r[s,1]==i:
                        mean+=g[r[s,0]]*z[s,1]*varw*varg
                        var += g[r[s,0]]**2*varw*varg
                    if r[s,2]==i:
                        mean+=g[r[s,0]]*z[s,2]*varw*varg
                        var += g[r[s,0]]**2*varw*varg
                w[i] = mean/var + numpy.sqrt(varw*varg/var)*numpy.random.normal(0,1)
				
            #Step 4: sample from P(g|z,w,a)
            for i in range(self.n):
                mean = 1.0*mug+self.a*w_prev[i]
                var = 1.0*varg
                for s in range(ntrain):
                    if r[s,0]==i:
                        mean+= w[r[s,1]]*z[s,1]*varg
                        mean+= w[r[s,2]]*z[s,2]*varg
                        var += w[r[s,1]]**2*varg
                        var += w[r[s,2]]**2*varg
                g[i] = mean/var+numpy.sqrt(varg/var)*numpy.random.normal(0,1)

        #Case 2:When g is a deterministic function of w, sample only P(w|z)
    	if varg == 0:
    		#Step 3: Sample from P(z|w,g,a)P(w) ignoring the relation between w and g
    		for i in range(self.n):
    			mean = 1.0*muw
    			var = 1.0*self.a**2*varw
    			for s in range(ntrain):
    				if r[s,1]==i:
    					mean+=g[r[s,0]]*z[s,1]*varw
    					var += g[r[s,0]]**2*varw
    				if r[s,2]==i:
    					mean+=g[r[s,0]]*z[s,2]*varw
    					var += g[r[s,0]]**2*varw
    			w[i] = mean/var + numpy.sqrt(varw/var)*numpy.random.normal(0,1)
    		#Step 4:Set g deterministically to be aw+b
    		g = a*w + mug
    
    	#Step 5: Sample P(a|wg) ~ P(wg|a)P(a)
    	mean = 1.0*mua*varg
    	var = 1.0*varg
    	for i in range(self.n):
    		mean += vara*(g[i]-mug)*w[i]
    		var += vara*w[i]**2
    	self.a = mean/var + numpy.sqrt(varg*vara/var)*numpy.random.normal(0,1)
    	#self.a = numpy.linalg.lstsq(numpy.vstack([w]).T,(g-mug))[0]
    	print 'a',self.a
        
    	#Stopping condition (I have arbitrarily chosen this)
    	return numpy.linalg.norm(w-w_prev)
        
def learnThurstone(model, r, max_iterations=100, VERBOSE=False, truth=None):
    ntrain = len(r)
    z = numpy.zeros((ntrain,3)) # for sampling
    model.g = numpy.ones((model.n)) * model.g_prior.theta[0]
    model.w = numpy.ones((model.n)) * model.w_prior.theta[0]
    model.a = model.a_prior.theta[0]
    for loopvar in range(max_iterations):
        norm_w_prev = numpy.linalg.norm(model.w)
        norm_w_diff = model.sample(r, z)
        if VERBOSE:
            print 'iter %d: norm_w_diff = %g' % (loopvar, norm_w_diff)
            if truth is not None:
                pearson = scipy.stats.pearsonr(truth.w, model.w)
                spearman = scipy.stats.spearmanr(truth.w, model.w)
                kendall = scipy.stats.kendalltau(truth.w, model.w)
                print 'w: (pearson,spearman,kendalltau) = (%g, %g, %g)' % (pearson[0],spearman[0],kendall[0])
                pearson = scipy.stats.pearsonr(truth.g, model.g)
                spearman = scipy.stats.spearmanr(truth.g, model.g)
                kendall = scipy.stats.kendalltau(truth.g, model.g)
                print 'g: (pearson,spearman,kendalltau) = (%g, %g, %g)' % (pearson[0],spearman[0],kendall[0])
                print ''
        if norm_w_diff < .01*norm_w_prev:
            break
