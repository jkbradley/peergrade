# Unit test: Run RBTL variant on synthetic data.

import numpy

import sys, os
sys.path.append(os.path.abspath('..'))
from addPaths import *
addPaths('..')

from syntheticTester import *

#=================================================================

# True model
n = 30
w_mu = 1.0
w_sigma2 = 1.0
theta = [1,1]

# Data
ntrain = 10 * n

# Learning
learned_model_type = 'rbtlIndep'
learner_base_params = \
    rbtlModelLearner_base.rbtlModelLearnerParameters_base()
#learner_base_params.w_prior =
learner_base_CVparams = \
    rbtlModelLearner_base.rbtlModelLearnerCVParameters_base()
learner_base_CVparams.nfolds = 4
learner_base_CVparams.cv_choose_w_prior = True
if learner_base_CVparams.cv_choose_w_prior:
    learner_base_CVparams.cv_grid_w_sigma2 = [.001, .01, .1, 1, 10, 100]
else:
    learner_base_CVparams.cv_grid_w_sigma2 = [w_sigma2]
#   (for RBTL)
init_theta = [0,1]
rbtlLearner_params = \
    rbtlModelLearner.rbtlModelLearnerParameters(learner_base_params)
rbtlLearner_params.learn_alpha = 'grid'
rbtlLearner_CVparams = \
    rbtlModelLearner.rbtlModelLearnerCVParameters(learner_base_CVparams)
rbtlLearner_CVparams.cv_choose_lambda_a = True
#   (for RBTL-indep)
rbtlIndepLearner_params = \
    rbtlModelIndepLearner.rbtlModelIndepLearnerParameters(learner_base_params)
#rbtlIndepLearner_params.g_prior
rbtlIndepLearner_CVparams = \
    rbtlModelIndepLearner.rbtlModelIndepLearnerCVParameters(learner_base_CVparams)
rbtlIndepLearner_CVparams.cv_choose_g_sigma2 = True
rbtlIndepLearner_CVparams.cv_choose_lambda_wg = True  # If False, do not use w-g link.
# Optimization settings
optParams = optimizationParams()

#=================================================================

if rbtlLearner_CVparams.cv_choose_lambda_a:
    rbtlLearner_CVparams.cv_grid_lambda_a = [.001, .01, .1, 1, 10, 100]
else:
    rbtlLearner_CVparams.cv_grid_lambda_a = [.1]
if rbtlLearner_params.learn_alpha == 'grid':
    rbtlLearner_params.alpha_grid = [.001, .01, .1, 1, 10, 100]
if rbtlIndepLearner_CVparams.cv_choose_g_sigma2:
    rbtlIndepLearner_CVparams.cv_grid_g_sigma2 = [.001, .01, .1, 1, 10, 100]
else:
    true_g_sigma2 = \
        numpy.square(self.truth.theta[1] * numpy.std(self.truth.w))
    rbtlIndepLearner_CVparams.cv_grid_g_sigma2 = [true_g_sigma2]
if rbtlIndepLearner_CVparams.cv_choose_lambda_wg:
    rbtlIndepLearner_CVparams.cv_grid_lambda_wg = [.001, .01, .1, 1, 10, 100]
else:
    rbtlIndepLearner_CVparams.cv_grid_lambda_wg = [0]

#=================================================================
# Create model.
truth = rbtlModel.rbtlModel(n)
truth.setTheta(theta)
truth.w_prior = scoreDistribution('normal', [w_mu,w_sigma2])
truth.genScores(truth.w_prior)

#=================================================================
# Generate data.
z_list = []
for i in range(ntrain):
    z_list.append(truth.sample(jutils.randTriplet(n)))

#=================================================================
# Learn model.
if learned_model_type == 'rbtl':
    learned = rbtlModel.rbtlModel(n)
    if rbtlLearner_params.learn_alpha in ['opt', 'grid']:
        learned.setTheta(init_theta)
    else:
        learned.setTheta(theta)
    #
    learner = rbtlModelLearner.rbtlModelLearner(learned, optParams, rbtlLearner_params)
    cv_params = rbtlLearner_CVparams
elif learned_model_type == 'rbtlIndep':
    learned = rbtlModelIndep.rbtlModelIndep(n)
    learned.g = numpy.ones(n)
    #
    learner = rbtlModelIndepLearner.rbtlModelIndepLearner(learned, optParams, rbtlIndepLearner_params)
    learner.g_prior.theta[0] = numpy.mean(truth.getAbilities())
    cv_params = rbtlIndepLearner_CVparams
else:
    raise Exception()
learned.w = numpy.ones(n)
#
learner.w_prior.theta[0] = w_mu

learner.chooseRegularization(z_list, cv_params, \
                             log_fid=sys.stdout, VERBOSE=True)
learner.learn(z_list, log_fid=sys.stdout, VERBOSE=True)
learned = learner.model

#=================================================================
# Get results.
L2_w_error = numpy.linalg.norm(learned.w - truth.w, 2) / n
Linf_w_error = numpy.max(numpy.abs(learned.w - truth.w))
print '\nResults\n----------------------------------------\n'
print 'L2_w_error: %g' % L2_w_error
print 'Linf_w_error: %g' % Linf_w_error
if learned_model_type == 'rbtl':
    a_error = numpy.abs(learned.theta[0] - truth.theta[0])
    print 'learned alpha: %g' % learned.theta[0]
    print 'true alpha: %g' % truth.theta[0]
    print 'a_error: %g' % a_error
elif learned_model_type == 'rbtlIndep':
    L2_g_error = numpy.linalg.norm(learned.g - truth.getAbilities(), 2) / n
    Linf_g_error = numpy.max(numpy.abs(learned.g - truth.getAbilities()))
    print 'L2_g_error: %g' % L2_g_error
    print 'Linf_g_error: %g' % Linf_g_error
    if rbtlIndepLearner_CVparams.cv_choose_lambda_wg:
        print 'wg_alpha: %g' % learner.wg_alpha
else:
    raise Exception()

