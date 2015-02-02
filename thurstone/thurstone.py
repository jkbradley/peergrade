import numpy, random, math
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy

d = 20 #number of students
n = 100*d #number of samples

num_iterations = 90

r = numpy.zeros((n,3))
z = numpy.zeros((n,3))
muw=1
mug=1
mua = 12
varw=1
varg=1
vara=.1

#Generate synthetic data
w_true = muw + numpy.sqrt(varw)*numpy.random.normal(0,1,d)
a_true = mua + numpy.sqrt(vara)*numpy.random.normal(0,1)
g_true = a_true*w_true.copy() + mug + numpy.sqrt(varg)*numpy.random.normal(0,1,d)
print 'true_a',a_true

for loopvar in range(n):
	r[loopvar,:] = random.sample(range(d),3)
	if g_true[r[loopvar,0]]*w_true[r[loopvar,1]] + numpy.random.normal(0,1) < g_true[r[loopvar,0]]*w_true[r[loopvar,2]] + numpy.random.normal(0,1):
		temp = r[loopvar,1]
		r[loopvar,1] = r[loopvar,2]
		r[loopvar,2] = temp

#Inference
#Step 1: Initialize
g=numpy.ones((d))*mug
w=numpy.ones((d))*muw
a=mua

for loopvar in range(num_iterations):
	w_prev = w.copy()
	#Step 2: Sample from P(z|g,w,a)
	for s in range(n):
		#need to sample z[s,1] and z[s,2] in a manner that z[s,1]>=z[s,2]
		zsum = numpy.random.normal(g[r[s,0]]*w[r[s,1]]+g[r[s,0]]*w[r[s,2]],numpy.sqrt(2))
		zdiff = numpy.sqrt(2)*scipy.stats.truncnorm.rvs(-(g[r[s,0]]*w[r[s,1]]-g[r[s,0]]*w[r[s,2]]),numpy.Inf)

		z[s,1] = 0.5*(zsum+zdiff)
		z[s,2] = 0.5*(zsum-zdiff)
		
	#The next two steps are broken into two cases		
	#Case 1: When g is NOT a deterministic function of w, sample P(w|zg) and then P(g|wz)
	if varg != 0:
		#Step 3: Sample from P(w|z,g,a)
		for i in range(d):
			mean = 1.0*varw*a*(g[i]-mug)+muw*varg
			var = 1.0*a**2*varw+varg
			for s in range(n):
				if r[s,1]==i:
					mean+=g[r[s,0]]*z[s,1]*varw*varg
					var += g[r[s,0]]**2*varw*varg
				if r[s,2]==i:
					mean+=g[r[s,0]]*z[s,2]*varw*varg
					var += g[r[s,0]]**2*varw*varg
			w[i] = mean/var + numpy.sqrt(varw*varg/var)*numpy.random.normal(0,1)
				
		#Step 4: sample from P(g|z,w,a)
		for i in range(d):
			mean = 1.0*mug+a*w_prev[i]
			var = 1.0*varg
			for s in range(n):
				if r[s,0]==i:
					mean+= w[r[s,1]]*z[s,1]*varg
					mean+= w[r[s,2]]*z[s,2]*varg
					var += w[r[s,1]]**2*varg
					var += w[r[s,2]]**2*varg
			g[i] = mean/var+numpy.sqrt(varg/var)*numpy.random.normal(0,1)

	#Case 2:When g is a deterministic function of w, sample only P(w|z)
	if varg == 0:
		#Step 3: Sample from P(z|w,g,a)P(w) ignoring the relation between w and g
		for i in range(d):
			mean = 1.0*muw
			var = 1.0*a**2*varw
			for s in range(n):
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
	for i in range(d):
		mean += vara*(g[i]-mug)*w[i]
		var += vara*w[i]**2
	a = mean/var + numpy.sqrt(varg*vara/var)*numpy.random.normal(0,1)
	#a = numpy.linalg.lstsq(numpy.vstack([w]).T,(g-mug))[0]
	print 'a',a

	#Stopping condition (I have arbitrarily chosen this)	
	if numpy.linalg.norm(w-w_prev)<.01*numpy.linalg.norm(w):
		break
	
plt.clf()
plt.scatter(w,w_true)
plt.xlabel('w vs w_true '+ str(numpy.linalg.norm(scipy.stats.rankdata(w_true)-scipy.stats.rankdata(w),1)/d))
if varg != 0:
	print 'pearson correlation between w and g', scipy.stats.pearsonr(w,g)
