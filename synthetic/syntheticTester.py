# Generate data from the RBTL model.
# Estimate using various models.
# Compare.

import copy, time, datetime, pickle
import numpy, numpy.matlib, math

import sys, os
sys.path.append(os.path.abspath('..'))
from addPaths import *
addPaths('..')

import jutils, data_utils, loadParams
import rbtlModel, rbtlModelIndep
import rbtlModelLearner_base, rbtlModelLearner, rbtlModelIndepLearner
from rbtlModel_base import *
from optimizationParams import *


# About parameters:
#   Model
#     w_nonneg: If true, w are made non-negative by adding abs(min negative w value).
class syntheticTesterParams(object):
    def __init__(self, testpath):
        params = loadParams.loadParams(testpath, VERBOSE=True)
        # Paths
        self.datapath_stem = loadParams.getParam(params, 'datapath_stem', str)
        self.outpath_stem = loadParams.getParam(params, 'outpath_stem', str)
        self.plotpath_stem = loadParams.getParam(params, 'plotpath_stem', str)
        #
        # True model
        self.n = loadParams.getParam(params, 'n', int)
        self.theta = loadParams.getParam(params, 'theta', list)
        if len(self.theta) != 2:
            raise Exception('Bad theta length: %d' % len(self.theta))
        self.w_distribution = loadParams.getParam(params, 'w_distribution', str)
        self.w_mu = loadParams.getParam(params, 'w_mu', float)
        self.w_sigma2 = loadParams.getParam(params, 'w_sigma2', float)
        self.w_nonneg = loadParams.getParam(params, 'w_nonneg', bool)
        self.w_L2norm = loadParams.getParam(params, 'w_L2norm', str)
        if self.w_L2norm == 'None':
            self.w_L2norm = None
        #
        # Learned model
        #  Use same as true model: w_distribution
        self.learned_model_type = None
        self.init_w_mu = loadParams.getParam(params, 'init_w_mu', float, 1.0)
        self.init_w_sigma2 = loadParams.getParam(params, 'init_w_sigma2', float, 1.0)
        self.learned_w_nonneg = loadParams.getParam(params, 'learned_w_nonneg', bool, False)
        self.learned_w_L2norm = loadParams.getParam(params, 'learned_w_L2norm', str, 'None')
        if self.learned_w_L2norm == 'None':
            self.learned_w_L2norm = None
        # Learner params
        tmp_learner_base_params = \
          rbtlModelLearner_base.rbtlModelLearnerParameters_base()
        tmp_learner_base_params.fix_w = False
        tmp_learner_base_params.w_prior = \
          scoreDistribution('normal', [self.init_w_mu,self.init_w_sigma2])
        if self.learned_w_nonneg:
            tmp_learner_base_params.w_bounds = [0,None]
        else:
            tmp_learner_base_params.w_bounds = None
        if self.learned_w_L2norm == 'n':
            tmp_learner_base_params.w_L2norm_constraint = self.n
        else:
            tmp_learner_base_params.w_L2norm_constraint = None
        # Learner params: For self.learned_model_type == 'rbtl':
        self.rbtlModelLearner_params = \
          rbtlModelLearner.rbtlModelLearnerParameters(tmp_learner_base_params)
        self.rbtlModelLearner_params.learn_alpha = \
          loadParams.getParam(params, 'learn_alpha', str, True)
        self.rbtlModelLearner_params.alpha_grid = \
          loadParams.getParam(params, 'alpha_grid', list, []) # Set by initCVvals() if empty.
        self.rbtlModelLearner_params.lambda_a = \
          loadParams.getParam(params, 'lambda_a', str, 0)
        self.learner_fixedTheta = None # Used if alpha is fixed. If None, use true value.
        # Learner params: For self.learned_model_type == 'rbtlIndep':
        self.rbtlModelIndepLearner_params = \
          rbtlModelIndepLearner.rbtlModelIndepLearnerParameters(tmp_learner_base_params)
        self.rbtlModelIndepLearner_params.fix_g = False
        self.init_g_mu = loadParams.getParam(params, 'init_g_mu', float, 1.0)
        self.init_g_sigma2 = loadParams.getParam(params, 'init_g_sigma2', float, 1.0)
        self.rbtlModelIndepLearner_params.g_prior = \
          scoreDistribution('normal', [self.init_g_mu,self.init_g_sigma2])
        self.rbtlModelIndepLearner_params.g_bounds = None
        self.rbtlModelIndepLearner_params.lambda_wg = 0
        self.rbtlModelIndepLearner_params.wg_alpha = 0
        # Learner CV params
        tmp_learner_base_CVparams = \
          rbtlModelLearner_base.rbtlModelLearnerCVParameters_base()
        tmp_learner_base_CVparams.nfolds = \
          loadParams.getParam(params, 'nfolds', int)
        tmp_learner_base_CVparams.cv_cleanComparisons = False
        tmp_learner_base_CVparams.cv_choose_w_prior = True
        tmp_learner_base_CVparams.cv_grid_w_sigma2 = \
            loadParams.getParam(params, 'cv_grid_w_sigma2', list, []) # Set by initCVvals() if empty.
        # Learner CV params: 'rbtl':
        self.rbtlModelLearner_CVparams = \
          rbtlModelLearner.rbtlModelLearnerCVParameters(tmp_learner_base_CVparams)
        self.rbtlModelLearner_CVparams.cv_choose_lambda_a = True
        self.rbtlModelLearner_CVparams.cv_grid_lambda_a = \
          loadParams.getParam(params, 'cv_grid_lambda_a', list, []) # Set by initCVvals() if empty.
        # Learner CV params: 'rbtlIndep':
        self.rbtlModelIndepLearner_CVparams = \
          rbtlModelIndepLearner.rbtlModelIndepLearnerCVParameters(tmp_learner_base_CVparams)
        self.rbtlModelIndepLearner_CVparams.cv_choose_g_prior = True
        self.rbtlModelIndepLearner_CVparams.cv_grid_g_sigma2 = \
          loadParams.getParam(params, 'cv_grid_g_sigma2', list, []) # Set by initCVvals() if empty.
        self.rbtlModelIndepLearner_CVparams.cv_choose_lambda_wg = True
        self.rbtlModelIndepLearner_CVparams.cv_grid_lambda_wg = \
          loadParams.getParam(params, 'cv_grid_lambda_wg', list, []) # Set by initCVvals() if empty.
        #---------
        # Test
        self.nruns = loadParams.getParam(params, 'nruns', int)
        self.max_ntrain = loadParams.getParam(params, 'max_ntrain', int)
        # Learning
        self.ntrain_vals = loadParams.getParam(params, 'ntrain_vals', list)
        # Optimization
        self.optParams = optimizationParams()
        if 'opt_max_iter' in params:
            self.optParams.max_iter = loadParams.getParam(params, 'opt_max_iter', int)
        if 'factr' in params:
            self.optParams.factr = loadParams.getParam(params, 'factr', float)
        #---------
        self.dataset_name = os.path.basename(self.datapath_stem)
        self.VERBOSE = False

    def __str__(self):
        s = 'syntheticTesterParams'
        s += jutils.printDictAsString(self.__dict__, prefix='  ')
        return s


# Parameters:
#   cleanComparisons   If True, then clean the train/test sets during CV:
#                      In train set, require each student to be a grader + gradee.
#                      In test set, remove students who do not appear in the train set.
class syntheticTester(object):
    def __init__(self, params):
        self.p = params
        """
        # Parameters from 'params':
        #   True model
        self.n = params.n
        self.w_distribution = params.w_distribution
        self.w_mu = params.w_mu
        self.w_sigma2 = params.w_sigma2
        self.w_nonneg = params.w_nonneg
        self.w_L2norm = params.w_L2norm
        #   Learned model
        #self.learned_model_type = params.learned_model_type
        #self.learned_w_bounds = params.learned_w_bounds
        #self.learned_w_L2norm_constraint = params.learned_w_L2norm_constraint
        #   Model: rbtlModelIndep only
        #self.g_mu = None
        #self.g_sigma2 = None
        #self.g_nonneg = None
        #   Learning
        #   Optimization
        #self.optParams = params.optParams

        # Options
        #self.fixSigma2=False # REPLACED BY cv_choose_w_prior
        #self.fix_g_Sigma2=False # for rbtlModelIndep only # REPLACED BY cv_choose_g_prior
        #self.fixLambda_a=False # REPLACED BY cv_choose_lambda_a
        #self.optParams.fix_theta=False
        #self.fixedTheta=None # If optParams.fix_theta==True, defaults to truth.theta
        #self.cleanComparisons = False
        #self.VERBOSE=False
        """

        # Internally set parameters:
        self.gotTruth = False
        self.gotData = False
        self.truth = None
        self.z_list = None
        #self.sigma2_vals = None
        #self.lambda_a_vals = None
        #self.g_sigma2_vals = None
        self.w_shift = 0 # set if w is shifted to make w non-negative (w_nonneg)
        self.w_scale = 1 # set if w is scaled to set L2norm(w) (w_L2norm)

    # Datapath, with the '.pkl' suffix
    def getDataPath(self, datapath_stem, n, w_mu, w_sigma2, a, b, run):
        return datapath_stem \
          + ('.n%d.mu%g.s%g.a%g.b%g.run%d.pkl' % (n, w_mu, w_sigma2, a, b, run))

    # Outpath, without the '.pkl' or '.log' suffix
    def getOutPathStem(self, outpath_stem, n, w_mu, w_sigma2, a, b, run, alg, ntrain):
        return outpath_stem \
          + ('.n%d.mu%g.s%g.a%g.b%g.run%d' % (n, w_mu, w_sigma2, a, b, run)) \
          + ('.%s.ntrain%d' % (alg, ntrain))
    
    def createTruth(self, theta):
        n = self.p.n
        w_distribution = self.p.w_distribution
        # Create true model.
        self.truth = rbtlModel.rbtlModel(n)
        self.truth.setTheta(theta)
        if w_distribution == 'normal':
            self.w_prior = \
              scoreDistribution('normal', [self.p.w_mu,self.p.w_sigma2])
            self.truth.genScores(self.w_prior)
        elif w_distribution in ['linear', 'hamspam']:
            self.w_prior = None
            if w_distribution == 'linear':
                self.truth.w = numpy.asarray(range(n)) + 1.0
            elif w_distribution == 'hamspam':
                self.truth.w = [0.1, 1.0] * (int(n/2))
                if len(self.truth.w) != n:
                    if len(self.truth.w) + 1 == n:
                        self.truth.w.append(0.1)
                    else:
                        raise Exception('SHOULD NEVER HAPPEN: w_distribution == hamspam initialization failed.')
            self.truth.w = numpy.asarray(self.truth.w)
            self.truth.w *= (numpy.sqrt(n) / numpy.linalg.norm(self.truth.w,2))
            if self.p.w_L2norm is None:
                raise Exception('ERROR: Learning will use incorrect regularization for prior for w (%s).  Not yet implemented.' % w_distribution)
        else:
            raise Exception('Bad w_distribution parameter: %s' % w_distribution)
        if self.p.w_nonneg or (self.p.w_L2norm is not None):
            if bool(self.p.w_nonneg) ^ bool(self.p.w_L2norm is not None):
                sys.stderr.write('\nWARNING: syntheticTester called with only one of w_nonneg, w_L2norm set (with values %s, %s). This will not use a prior on w and may not be consistent.\n\n' % (str(self.p.w_nonneg), str(self.p.w_L2norm)))
            self.w_prior = None
        if self.p.w_nonneg:
            minw = numpy.min(self.truth.w)
            if minw < 0:
                self.w_shift = numpy.abs(minw)
                self.truth.w += self.w_shift
        if self.p.w_L2norm is not None:
            if self.p.w_L2norm == 'n':
                self.w_scale = (numpy.sqrt(n) / numpy.linalg.norm(self.truth.w,2))
                self.truth.w *= self.w_scale
            else:
                raise Exception('Bad w_L2norm parameter: %s' % self.p.w_L2norm)
        self.gotTruth = True
        self.gotData = False

    def genData(self, ntrain):
        if not self.gotTruth:
            raise Exception('syntheticTester.genData() must be called AFTER createTruth()')
        self.ntrain = ntrain
        # Generate data.
        self.z_list = []
        for train_i in range(self.ntrain):
            self.z_list.append(self.truth.sample(jutils.randTriplet(self.truth.n)))
        self.gotData = True

    # Called by learn().
    def initCVvals(self):
        # Initial values:
        default_alpha_grid = [.01, .03, .1, .3, 1, 3, 10]
        default_grid_w_sigma2 = [.001, .01, .1, 1, 10, 100, 1000]
        default_grid_lambda_a = [.01, .1, 1, 10, 100]
        default_grid_g_sigma2 = [.001, .01, .1, 1, 10, 100, 1000]
        default_grid_lambda_wg = [.001, .01, .1, 1, 10, 100]
        # Set these value where needed.
        if self.p.rbtlModelLearner_params.learn_alpha == 'grid':
            self.p.rbtlModelLearner_params.alpha_grid = default_alpha_grid
        if (not self.p.rbtlModelLearner_CVparams.cv_choose_w_prior) or (self.w_prior is None):
            self.p.rbtlModelLearner_CVparams.cv_grid_w_sigma2 = [self.p.w_sigma2]
        else:
            self.p.rbtlModelLearner_CVparams.cv_grid_w_sigma2 = default_grid_w_sigma2
        if (not self.p.rbtlModelIndepLearner_CVparams.cv_choose_w_prior) or (self.w_prior is None):
            self.p.rbtlModelIndepLearner_CVparams.cv_grid_w_sigma2 = [self.p.w_sigma2]
        else:
            self.p.rbtlModelIndepLearner_CVparams.cv_grid_w_sigma2 = default_grid_w_sigma2
        if self.p.learned_model_type == 'rbtl':
            if (self.p.rbtlModelLearner_params.learn_alpha == 'fixed') or (not self.p.rbtlModelLearner_CVparams.cv_choose_lambda_a):
                self.p.rbtlModelLearner_CVparams.cv_grid_lambda_a = [1.0]
            else:
                self.p.rbtlModelLearner_CVparams.cv_grid_lambda_a = default_grid_lambda_a
        elif self.p.learned_model_type == 'rbtlIndep':
            if (not self.p.rbtlModelIndepLearner_CVparams.cv_choose_g_prior):
                true_g_sigma2 = \
                  numpy.square(self.w_scale * self.truth.theta[1] * numpy.std(self.truth.w))
                self.p.rbtlModelIndepLearner_CVparams.cv_grid_g_sigma2 = [true_g_sigma2]
            else:
                self.p.rbtlModelIndepLearner_CVparams.cv_grid_g_sigma2 =default_grid_g_sigma2
            if (not self.p.rbtlModelIndepLearner_CVparams.cv_choose_lambda_wg):
                self.p.rbtlModelIndepLearner_CVparams.cv_grid_lambda_wg = \
                  [self.p.rbtlModelIndepLearner_params.lambda_wg]
            else:
                self.p.rbtlModelIndepLearner_CVparams.cv_grid_lambda_wg = \
                  default_grid_lambda_wg
        else:
            raise Exception('Bad learned_model_type: %s' % self.p.learned_model_type)

    # Arguments:
    #   ntrain   If None, then use all existing training data.
    def learn(self, ntrain=None, log_fid=None):
        if not (self.gotTruth and self.gotData):
            raise Exception('syntheticTester.learn() must be called AFTER createTruth() and genData()')
        self.initCVvals()
        VERBOSE = (log_fid is not None) or self.p.VERBOSE
        n = self.p.n

        # Limit training data.
        if ntrain > self.ntrain:
            raise Exception('syntheticTester.learn() was called with ntrain larger' \
                            + ' than ntrain given to genData()')
        train_z_list = self.z_list[:ntrain]

        # Create empty model.
        if self.p.learned_model_type == 'rbtl':
            learned = rbtlModel.rbtlModel(n)
            if self.p.rbtlModelLearner_params.learn_alpha == 'opt':
                learned.setTheta([0,1])
            else:
                if self.p.learned_fixedTheta is None:
                    learned.setTheta(self.truth.theta)
                else:
                    learned.setTheta(self.p.learned_fixedTheta)
        elif self.p.learned_model_type == 'rbtlIndep':
            learned = rbtlModelIndep.rbtlModelIndep(n)
            learned.g = numpy.ones(n)
        else:
            raise Exception('Bad learned_model_type: %s' % self.p.learned_model_type)
        learned.w = numpy.ones(n)
        # Create learner, and choose regularization.
        if self.p.learned_model_type == 'rbtl':
            learner_type = rbtlModelLearner.rbtlModelLearner
            learner = learner_type(learned, self.p.optParams, \
                                   self.p.rbtlModelLearner_params)
            if self.w_prior is None:
                learner.w_prior = None
            else:
                learner.w_prior.theta[0] = self.w_prior.theta[0] # true mean
            learner.chooseRegularization(train_z_list, self.p.rbtlModelLearner_CVparams, \
                                         log_fid=log_fid, VERBOSE=VERBOSE)
        elif self.p.learned_model_type == 'rbtlIndep':
            learner_type = rbtlModelIndepLearner.rbtlModelIndepLearner
            learner = learner_type(learned, self.p.optParams, \
                                   self.p.rbtlModelIndepLearner_params)
            if self.w_prior is None:
                learner.w_prior = None
            else:
                learner.w_prior.theta[0] = self.w_prior.theta[0] # true mean
            # TO DO: fix_g, g_prior
            true_g_mu = numpy.mean(self.truth.getAbilities())
            learner.g_prior.theta[0] = true_g_mu
            learner.chooseRegularization(train_z_list, \
                                         self.p.rbtlModelIndepLearner_CVparams, \
                                         log_fid=log_fid, VERBOSE=VERBOSE)
        else:
            raise Exception('Bad learned_model_type: %s' % self.p.learned_model_type)
        
        # Learn model.
        start_time = time.time()
        learner.learn(train_z_list)
        runtime = time.time() - start_time
        learned = learner.model

        # Get results.
        L2_w_error = numpy.linalg.norm(learned.w - self.truth.w, 2) / n
        Linf_w_error = numpy.max(numpy.abs(learned.w - self.truth.w))
        w_kendalltau = scipy.stats.kendalltau(learned.w, self.truth.w)
        w_kendalltau = w_kendalltau[0]
        if math.isnan(w_kendalltau):
            w_kendalltau = 0
        if self.p.learned_model_type == 'rbtl':
            a_error = numpy.abs(learned.theta[0] - self.truth.theta[0])
        elif self.p.learned_model_type == 'rbtlIndep':
            L2_g_error = numpy.linalg.norm(learned.g - self.truth.getAbilities(), 2) / n
            Linf_g_error = numpy.max(numpy.abs(learned.g - self.truth.getAbilities()))
        else:
            raise Exception('Bad learned_model_type: %s' % self.p.learned_model_type)
        results = {}
        results['ntrain'] = ntrain
        results['CV_w_sigma2'] = learner.w_prior.theta[1]
        results['learned_model'] = learned
        results['L2_w_error'] = L2_w_error
        results['Linf_w_error'] = Linf_w_error
        results['w_kendalltau'] = w_kendalltau
        results['runtime'] = runtime
        if self.p.learned_model_type == 'rbtl':
            results['CV_lambda_a'] = learner.lambda_a
            results['a_error'] = a_error
        elif self.p.learned_model_type == 'rbtlIndep':
            results['CV_g_sigma2'] = learner.g_prior.theta[1]
            results['L2_g_error'] = L2_g_error
            results['Linf_g_error'] = Linf_g_error
            results['CV_lambda_wg'] = learner.lambda_wg
        else:
            raise Exception('Bad learned_model_type: %s' % self.p.learned_model_type)
        if VERBOSE:
            log_fid.write('runtime : %g\n' % runtime)
            log_fid.write('CV_w_sigma2 : %g\n' % learner.w_prior.theta[1])
            log_fid.write('L2_w_error : %g\n' % L2_w_error)
            log_fid.write('Linf_w_error : %g\n' % Linf_w_error)
            log_fid.write('w_kendalltau : %g\n' % w_kendalltau)
            if self.p.learned_model_type == 'rbtl':
                log_fid.write('CV_lambda_a : %g\n' % learner.lambda_a)
                log_fid.write('alpha : %g\n' % learned.theta[0])
                log_fid.write('alpha error : %g\n' % a_error)
            elif self.p.learned_model_type == 'rbtlIndep':
                log_fid.write('CV_g_sigma2 : %g\n' % learner.g_prior.theta[1])
                log_fid.write('L2_g_error : %g\n' % L2_g_error)
                log_fid.write('Linf_g_error : %g\n' % Linf_g_error)
                log_fid.write('CV_lambda_wg : %g\n' % learner.lambda_wg)
            else:
                raise Exception('Bad learned_model_type: %s' % self.p.learned_model_type)
        return results

    def __str__(self):
        s = 'syntheticTester'
        s += jutils.printDictAsString(self.__dict__, prefix='  ', ignore=['truth', 'z_list'])
        return s
